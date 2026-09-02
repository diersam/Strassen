// ============================================================================
//  bench_hsbm -- RI-K shaped benchmark: dense BLAS-3 vs BSM vs HSBM.
//
//  Builds synthetic operands with the block-sparsity structure that actually
//  occurs in resolution-of-the-identity exchange for a quasi-1D system (the
//  regime of the (AT)_16 fragment), then times
//
//      dense    one BLAS-3 call on the full matrices (MKL)
//      bsm      flat block-sparse multiplication      (matmult)
//      hsbm     peeled-core parallel HSBM             (matmult_hsbm)
//      hsbm1    the original single-threaded entry    (matmult_strassen_sparse)
//
//  and reports GFLOP/s plus relative RMSD against the dense result.
//
//  Test cases
//    transform  C(Npair x Naux) = A(Npair x Naux) . B(Naux x Naux)
//               the (mu nu|P)' = sum_Q (mu nu|Q) (Q|P)^-1/2 transform.  Tall
//               panel times a big square matrix: the shape where Strassen has
//               room to recurse.
//    occrik     C(Nbf x Nocc) = A(Nbf x Nbf) . B(Nbf x Nocc)
//               the dominant occ-RI-K contraction (mu i|P) = sum_nu (mu nu|P) C_nu,i
//               for one auxiliary function.  Depth is capped by Nocc/b.
//    square     C(N x N) = A(N x N) . B(N x N)
//               density-matrix-like, for comparison with the published numbers.
// ============================================================================
#include "matrix.h"
#include "matrix.hpp"
#include "blocksparsematrix.h"
#include "blocksparsematrix.hpp"
#include "strassen.hpp"
#include "hsbm_driver.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

size_t min_size_for_strassen;   // consumed by strassen.hpp

using clk = std::chrono::steady_clock;
static double secs(clk::time_point a, clk::time_point b){
    return 1e-9*(double)std::chrono::duration_cast<std::chrono::nanoseconds>(b-a).count();
}

// ---------------------------------------------------------------------------
//  Operand generation.  Element (r,c) gets  U(-1,1) * exp(-|pos_r - pos_c|/lam),
//  with positions spread uniformly over [0,1).  That is the exponential decay
//  with inter-atomic separation that makes AO-basis matrices block sparse, and
//  it reproduces the banded pattern of the paper's Figure 5.  lam is the decay
//  length as a fraction of the system: lam >= 1 is essentially dense, lam ~ 0.02
//  is strongly sparse.
// ---------------------------------------------------------------------------
struct XorShift {
    uint64_t s;
    explicit XorShift(uint64_t seed) : s(seed*0x9E3779B97F4A7C15ull + 0x1234567ull) { if(!s) s=1; }
    inline uint64_t next(){ s^=s<<13; s^=s>>7; s^=s<<17; return s; }
    inline double uniform(){ return 2.0*((double)(next()>>11)*0x1.0p-53) - 1.0; }
};

static void fill_decay(Matrix<double>& M, double lam, double diag_boost, uint64_t seed)
{
    const size_t nr = M.nrow(), nc = M.ncol();
    // exp lookup: distance in [0,1] on a 4096-point grid
    constexpr size_t NT = 4097;
    std::vector<double> tab(NT);
    for(size_t t=0;t<NT;++t) tab[t] = std::exp(-((double)t/(double)(NT-1))/lam);

    #pragma omp parallel for schedule(static)
    for(size_t c=0;c<nc;++c){
        XorShift rng(seed*1000003ull + c);
        const double yc = (nc>1) ? (double)c/(double)(nc-1) : 0.0;
        double* col = &M.elem(0,c);
        for(size_t r=0;r<nr;++r){
            const double xr = (nr>1) ? (double)r/(double)(nr-1) : 0.0;
            const double d  = std::fabs(xr-yc);
            const size_t t  = (size_t)(d*(double)(NT-1));
            col[r] = rng.uniform()*tab[t < NT ? t : NT-1];
        }
        if(diag_boost != 0.0 && c < nr) col[c] += diag_boost;
    }
    M.calc_frobenius_norm();
}

// ---------------------------------------------------------------------------
//  Relative RMSD of a block-sparse result against a dense reference, computed
//  block-wise so no full dense copy of the result is ever materialised.
// ---------------------------------------------------------------------------
static double rel_rmsd(const BlockSparseMatrix<double>& X, const Matrix<double>& ref)
{
    const size_t nr = X.nrow(), nc = X.ncol();
    const size_t brs = X.max_blocksize_row(), bcs = X.max_blocksize_col();
    const size_t nrb = X.nrowblocks(), ncb = X.ncolblocks();
    double num = 0.0, den = 0.0;
    #pragma omp parallel for schedule(dynamic) reduction(+:num,den) collapse(2)
    for(size_t cb=0; cb<ncb; ++cb)
        for(size_t rb=0; rb<nrb; ++rb){
            const auto& blk = X.block(rb,cb);
            const size_t r0=rb*brs, c0=cb*bcs;
            const size_t r1=std::min(r0+brs,nr), c1=std::min(c0+bcs,nc);
            const bool have = (blk.size()!=0);
            for(size_t c=c0;c<c1;++c)
                for(size_t r=r0;r<r1;++r){
                    const double v = ref.elem(r,c);
                    const double x = have ? blk.elem(r-r0,c-c0) : 0.0;
                    num += (x-v)*(x-v);
                    den += v*v;
                }
        }
    return (den>0.0) ? std::sqrt(num/den) : 0.0;
}

static double stored_fraction(const BlockSparseMatrix<double>& X){
    return (double)X.no_of_alloc_blocks()/(double)X.nblocks();
}

// ---------------------------------------------------------------------------
struct Cfg {
    std::string test = "transform";
    size_t npair = 16384, naux = 4096;
    size_t nbf   = 8192,  nocc = 1024;
    size_t n     = 8192;
    size_t bs    = 64;
    double lam   = 0.35;
    double tau   = 0.0;
    double thresh= 0.0;
    size_t S     = 0;
    size_t smin  = 0;
    int    reps  = 3;
    int    nthr  = 0;
    bool   do_dense = true, do_bsm = true, do_hsbm = true, do_hsbm1 = false;
};

static void usage(){
    printf(
"bench_hsbm [options]\n"
"  --test transform|occrik|square   (default transform)\n"
"  --npair N --naux N               transform dimensions   (16384, 4096)\n"
"  --nbf N --nocc N                 occrik dimensions      (8192, 1024)\n"
"  --n N                            square dimension       (8192)\n"
"  --bs B                           block size             (64)\n"
"  --lambda L                       decay length, fraction of system (0.35)\n"
"  --tau T                          block storage threshold (0)\n"
"  --thresh T                       screening threshold theta (0)\n"
"  --super-tile S                   S in blocks, 0 = auto  (0)\n"
"  --smin M                         minimum Strassen size in elements (0)\n"
"  --reps N                         repetitions            (3)\n"
"  --threads N                      OpenMP threads, 0 = default\n"
"  --no-dense --no-bsm --no-hsbm    skip a method\n"
"  --hsbm1                          also time the original single-threaded entry\n");
}

int main(int argc, char** argv)
{
    Cfg cfg;
    for(int a=1;a<argc;++a){
        const std::string k = argv[a];
        auto val = [&]() -> const char* { if(a+1>=argc){ usage(); std::exit(1);} return argv[++a]; };
        if      (k=="--test")       cfg.test  = val();
        else if (k=="--npair")      cfg.npair = std::stoul(val());
        else if (k=="--naux")       cfg.naux  = std::stoul(val());
        else if (k=="--nbf")        cfg.nbf   = std::stoul(val());
        else if (k=="--nocc")       cfg.nocc  = std::stoul(val());
        else if (k=="--n")          cfg.n     = std::stoul(val());
        else if (k=="--bs")         cfg.bs    = std::stoul(val());
        else if (k=="--lambda")     cfg.lam   = std::stod(val());
        else if (k=="--tau")        cfg.tau   = std::stod(val());
        else if (k=="--thresh")     cfg.thresh= std::stod(val());
        else if (k=="--super-tile") cfg.S     = std::stoul(val());
        else if (k=="--smin")       cfg.smin  = std::stoul(val());
        else if (k=="--reps")       cfg.reps  = std::stoi(val());
        else if (k=="--threads")    cfg.nthr  = std::stoi(val());
        else if (k=="--no-dense")   cfg.do_dense = false;
        else if (k=="--no-bsm")     cfg.do_bsm   = false;
        else if (k=="--no-hsbm")    cfg.do_hsbm  = false;
        else if (k=="--hsbm1")      cfg.do_hsbm1 = true;
        else { usage(); return (k=="-h"||k=="--help") ? 0 : 1; }
    }
    if(cfg.nthr>0) omp_set_num_threads(cfg.nthr);
    const int nthr = omp_get_max_threads();

    size_t M,K,N;
    if      (cfg.test=="transform"){ M=cfg.npair; K=cfg.naux; N=cfg.naux; }
    else if (cfg.test=="occrik")   { M=cfg.nbf;   K=cfg.nbf;  N=cfg.nocc; }
    else if (cfg.test=="square")   { M=cfg.n;     K=cfg.n;    N=cfg.n;    }
    else { fprintf(stderr,"unknown --test %s\n",cfg.test.c_str()); return 1; }

    const double gflop = 2.0*(double)M*(double)N*(double)K*1e-9;
    const double gib   = (double)sizeof(double)/(1024.0*1024.0*1024.0);
    printf("== %s :  C(%zu x %zu) = A(%zu x %zu) . B(%zu x %zu)\n",
           cfg.test.c_str(), M,N, M,K, K,N);
    printf("   b=%zu  lambda=%g  tau=%g  theta=%g  threads=%d  reps=%d\n",
           cfg.bs,cfg.lam,cfg.tau,cfg.thresh,nthr,cfg.reps);
    printf("   nominal work %.1f GFLOP;  dense operands %.2f GiB\n",
           gflop, ((double)M*(double)K + (double)K*(double)N + 2.0*(double)M*(double)N)*gib);
    fflush(stdout);

    // ---- operands ---------------------------------------------------------
    auto t0 = clk::now();
    Matrix<double> A(M,K), B(K,N);
    const double lam2 = (cfg.test=="transform") ? cfg.lam*3.0 : cfg.lam;
    fill_decay(A, cfg.lam, 0.0, 1);
    fill_decay(B, lam2, (cfg.test=="transform")?1.0:0.0, 2);
    printf("   generated operands in %.2f s\n", secs(t0,clk::now())); fflush(stdout);

    // ---- dense reference --------------------------------------------------
    Matrix<double> Cref(0.0,M,N);
    double t_dense = 0.0;
    if(cfg.do_dense){
        for(int r=0;r<cfg.reps;++r){
            const auto s = clk::now();
            matmult(Cref,A,false,B,false,1.0,0.0);
            const double dt = secs(s,clk::now());
            t_dense = (r==0)?dt:std::min(t_dense,dt);
        }
        printf("   %-6s %8.3f s  %9.1f GFLOP/s\n","dense",t_dense,gflop/t_dense);
        fflush(stdout);
    }

    // ---- block-sparse operands -------------------------------------------
    t0 = clk::now();
    BlockSparseMatrix<double> Ab(A,cfg.bs,cfg.bs,cfg.tau);
    BlockSparseMatrix<double> Bb(B,cfg.bs,cfg.bs,cfg.tau);
    Ab.calc_frobenius_norms();
    Bb.calc_frobenius_norms();
    A = Matrix<double>();   // release the dense operands
    B = Matrix<double>();
    printf("   blocked in %.2f s;  A stored %.1f%%, B stored %.1f%%\n",
           secs(t0,clk::now()), 1e2*stored_fraction(Ab), 1e2*stored_fraction(Bb));
    fflush(stdout);

    printf("\n   %-6s %8s  %9s  %9s  %s\n","method","time[s]","GFLOP/s","vs dense","rel. RMSD");
    if(cfg.do_dense)
        printf("   %-6s %8.3f  %9.1f  %9s  %s\n","dense",t_dense,gflop/t_dense,"1.00","-");

    // ---- BSM ---------------------------------------------------------------
    if(cfg.do_bsm){
        BlockSparseMatrix<double> Cb(M,N,cfg.bs,cfg.bs,cfg.tau);
        double t=0.0;
        for(int r=0;r<cfg.reps;++r){
            const auto s = clk::now();
            matmult(Cb,Ab,false,Bb,false,cfg.thresh,1.0,0.0);
            const double dt = secs(s,clk::now());
            t = (r==0)?dt:std::min(t,dt);
        }
        printf("   %-6s %8.3f  %9.1f  %9.2f  ","bsm",t,gflop/t,cfg.do_dense?t_dense/t:0.0);
        if(cfg.do_dense) printf("%.3e\n", rel_rmsd(Cb,Cref)); else printf("-\n");
        fflush(stdout);
    }

    // ---- HSBM, parallel peeled core ---------------------------------------
    if(cfg.do_hsbm){
        BlockSparseMatrix<double> Ch(M,N,cfg.bs,cfg.bs,cfg.tau);
        HSBMParams<double> par;
        par.thresh            = cfg.thresh;
        par.super_tile_blocks = cfg.S;
        par.min_size_strassen = cfg.smin;
        par.nthreads          = nthr;
        par.warm_pools        = true;
        if(const char* lv = getenv("HSBM_LEVEL_VALUE")) par.level_value    = atof(lv);
        if(const char* kg = getenv("HSBM_KSPLIT_GIB"))  par.k_split_max_gib = atof(kg);
        HSBMWorkspaceSet<double> wss(nthr);
        HSBMStats st{};
        double t=0.0;
        for(int r=0;r<cfg.reps;++r){
            Ch.zero();
            // Allocate the core output blocks outside the timer.  BSM never
            // pays this (it allocates lazily), and matmult_strassen_sparse gets
            // its fill_with_values outside the timer too, so charging it here
            // would be an unfair few percent.
            hsbm_prepare_output(Ch,par,Ab,Bb);
            const auto s = clk::now();
            st = matmult_hsbm(Ch,Ab,Bb,par,wss);
            const double dt = secs(s,clk::now());
            t = (r==0)?dt:std::min(t,dt);
        }
        printf("   %-6s %8.3f  %9.1f  %9.2f  ","hsbm",t,gflop/t,cfg.do_dense?t_dense/t:0.0);
        if(cfg.do_dense) printf("%.3e\n", rel_rmsd(Ch,Cref)); else printf("-\n");
        if(st.n_tiles*st.k_split < (size_t)nthr && st.used_core)
            printf("          WARNING: %zu tiles for %d threads -- %d thread(s) idle\n",
                   st.n_tiles*st.k_split, nthr, nthr-(int)(st.n_tiles*st.k_split));
        printf("          S=%zu (L=%zu)  core %zux%zux%zu blocks  coverage %.3f  "
               "tiles %zu x k%zu  strassen %zu/%zu  scratch %.2f GiB/thread"
               "  replicas %.2f GiB\n",
               st.super_tile, st.levels, st.core_ib, st.core_jb, st.core_kb,
               st.coverage, st.n_tiles, st.k_split, st.strassen_nodes, st.tree_nodes,
               st.scratch_gib_per_thread, st.replica_gib);
        fflush(stdout);
    }

    // ---- original single-threaded entry point ------------------------------
    if(cfg.do_hsbm1){
        if(M!=K || K!=N) printf("   %-6s  (skipped: needs square operands)\n","hsbm1");
        else {
            BlockSparseMatrix<double> Cs(M,N,cfg.bs,cfg.bs,cfg.tau);
            min_size_for_strassen = cfg.smin;
            double t=0.0;
            for(int r=0;r<cfg.reps;++r){
                Cs.fill_with_values(0.0);
                const auto s = clk::now();
                matmult_strassen_sparse(Cs,Ab,false,Bb,false,cfg.thresh,1.0,0.0);
                const double dt = secs(s,clk::now());
                t = (r==0)?dt:std::min(t,dt);
            }
            printf("   %-6s %8.3f  %9.1f  %9.2f  ","hsbm1",t,gflop/t,cfg.do_dense?t_dense/t:0.0);
            if(cfg.do_dense) printf("%.3e\n", rel_rmsd(Cs,Cref)); else printf("-\n");
        }
        fflush(stdout);
    }
    printf("\n");
    return 0;
}
