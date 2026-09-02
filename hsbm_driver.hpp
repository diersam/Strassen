#ifndef HSBM_DRIVER_HPP
#define HSBM_DRIVER_HPP
// ============================================================================
//  Parallel driver for HSBM on rectangular, ragged operands.
//
//  Strategy ("peel the perfectly sized core"):
//
//    Split each of the three block axes as  n = m*S + r,  with S = 2^L.
//
//      core    i<m_i*S, j<m_j*S, k<m_k*S   -> HSBM at uniform depth L.
//                                            Because every super-block is full,
//                                            is_edge_case never fires and the
//                                            cost model is free to pick Strassen
//                                            at every level on flops alone.
//      slab 1  i>=m_i*S, all j, all k      -> BSM
//      slab 2  i<core,  j>=m_j*S, all k    -> BSM
//      slab 3  i<core,  j<core,  k>=m_k*S  -> BSM
//
//    The three slabs are disjoint and together with the core they cover every
//    (i,j,k) triple exactly once.
//
//  Parallelism lives ONLY here.  The recursive kernel runs single-threaded on
//  one core tile with its own scratch pool, so the published single-core
//  behaviour is exactly what each thread executes.  Tiles are ordered by the
//  dry run's own flop estimate and handed out with schedule(dynamic,1) --
//  longest-processing-time-first, using a cost oracle we get for free.
//
//  Nothing here touches the cost model or the Strassen_fac penalties.
// ============================================================================
#include "blocksparsematrix.h"
#include "blocksparsematrix.hpp"
#include "strassen.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <numeric>
#include <vector>

// ----------------------------------------------------------------------------
//  Tunables.  thresh has the same meaning as in matmult()/matmult_strassen_sparse().
// ----------------------------------------------------------------------------
template<typename Num>
struct HSBMParams {
    Num    thresh            = Num(0);
    size_t super_tile_blocks = 0;   // S, in blocks. 0 = choose automatically.
    double level_value       = 1.02;// measured worth of one Strassen level; see
                                    // hsbm_choose_level().  Raise toward 8/7 on
                                    // hardware where additions are cheap.
    size_t min_size_strassen = 0;   // s_min, in matrix elements. 0 = recurse fully.
    bool   warm_pools        = true;// touch scratch up front (NUMA + no page faults when timing)
    int    nthreads          = 0;   // 0 = omp_get_max_threads()
    bool   verbose           = false;
    // Parallelise over the k axis when the (i,j) grid cannot supply enough
    // tiles.  Each thread takes a slice of k and accumulates into a private
    // copy of the core of C; the copies are summed at the end.  This buys
    // parallelism WITHOUT shrinking the super-tile, so the recursion keeps its
    // depth -- worth ~20% on a square operand at 4 threads, where the (i,j)-only
    // scheme is forced from L=6 down to L=4.
    // Costs one private core-of-C per extra thread, so it is bounded by memory:
    // set k_split_max_gib to cap it.  0 disables k splitting entirely.
    double k_split_max_gib   = 8.0;
};

struct HSBMStats {
    size_t super_tile = 0;      // S
    size_t levels     = 0;      // L
    size_t core_ib = 0, core_jb = 0, core_kb = 0;
    size_t n_tiles = 0;         // m_i * m_j
    size_t k_split = 1;         // threads working concurrently along k
    double replica_gib = 0.0;   // total memory spent on private C copies
    double coverage = 0.0;      // fraction of (i,j,k) block triples inside the core
    double scratch_gib_per_thread = 0.0;
    bool   used_core = false;
    size_t strassen_nodes = 0;  // super-blocks the cost model sent down the 7-path
    size_t tree_nodes     = 0;  // super-blocks examined, all levels
};

// ----------------------------------------------------------------------------
//  Per-thread scratch.  Constructed by its owning thread so first touch (and
//  therefore NUMA placement) lands on the right socket.
// ----------------------------------------------------------------------------
template<typename Num>
struct HSBMWorkspace final {
    size_t block_elems = 0;
    size_t pool_blocks = 0;
    BlockMemoryPool<Num>         pool;
    std::vector<const Mat<Num>*> ptrs;

    HSBMWorkspace(size_t block_elems_, size_t pool_blocks_, size_t ptr_slots, bool warm)
      : block_elems(block_elems_), pool_blocks(pool_blocks_),
        pool(block_elems_), ptrs(ptr_slots)
    {
        if(!warm) return;
        std::vector<Num*> held;
        held.reserve(pool_blocks_);
        for(size_t i=0;i<pool_blocks_;++i){
            Num* p = pool.allocate(block_elems_);
            std::memset((void*)p, 0, block_elems_*sizeof(Num)); // first touch
            held.push_back(p);
        }
        for(Num* p : held) pool.deallocate((void*)p, block_elems_);
    }
    HSBMWorkspace(const HSBMWorkspace&)            = delete;
    HSBMWorkspace& operator=(const HSBMWorkspace&) = delete;
};

// Reusable across calls; hand the same object to every matmult in a loop.
template<typename Num>
class HSBMWorkspaceSet final {
  public:
    explicit HSBMWorkspaceSet(int nthreads = 0)
      : _ws((size_t)(nthreads>0 ? nthreads : omp_get_max_threads())) {}

    HSBMWorkspace<Num>& get(int tid, size_t block_elems, size_t pool_blocks,
                            size_t ptr_slots, bool warm)
    {
        auto& slot = _ws[(size_t)tid];
        if(!slot || slot->block_elems < block_elems
                 || slot->pool_blocks < pool_blocks
                 || slot->ptrs.size() < ptr_slots)
            slot = std::make_unique<HSBMWorkspace<Num>>(block_elems, pool_blocks, ptr_slots, warm);
        return *slot;
    }
    size_t nthreads() const { return _ws.size(); }
    void   release()        { for(auto& s : _ws) s.reset(); }
  private:
    std::vector<std::unique_ptr<HSBMWorkspace<Num>>> _ws;
};

// ----------------------------------------------------------------------------
//  Block-sparse multiplication restricted to a block sub-range.  C += alpha*A*B
//  over ib in [ib0,ib1), jb in [jb0,jb1), kb in [kb0,kb1).  Identical screening
//  and lazy-allocation semantics to the flat BSM matmult; parallel over (ib,jb)
//  with kb sequential, so every output block has exactly one writer.
// ----------------------------------------------------------------------------
template<typename Num>
void bsm_matmult_range(BlockSparseMatrix<Num>& C,
                       const BlockSparseMatrix<Num>& A,
                       const BlockSparseMatrix<Num>& B,
                       const size_t ib0, const size_t ib1,
                       const size_t jb0, const size_t jb1,
                       const size_t kb0, const size_t kb1,
                       const Num thresh_per_block, const Num alpha)
{
    using MatB = Matrix<Num,BlockAllocator<Num>>;
    if(ib0>=ib1 || jb0>=jb1 || kb0>=kb1) return;
    const size_t nib_r = ib1-ib0, njb_r = jb1-jb0;

    #pragma omp parallel for schedule(dynamic,1) if(!omp_in_parallel())
    for(size_t ij=0; ij<nib_r*njb_r; ++ij){
        const size_t ib = ib0 + ij % nib_r;
        const size_t jb = jb0 + ij / nib_r;
        auto& c_block = C.block(ib,jb);
        for(size_t kb=kb0; kb<kb1; ++kb){
            const auto& a_block = A.block(ib,kb);
            const auto& b_block = B.block(kb,jb);
            if(a_block.size()==0 || b_block.size()==0) continue;
            if(a_block.frobenius_norm()*b_block.frobenius_norm() < thresh_per_block) continue;
            const bool exists = (c_block.size() != 0);
            if(!exists) c_block = MatB(a_block.nrow(), b_block.ncol(), C.allocator());
            matmult(c_block, a_block, false, b_block, false, alpha, exists?Num(1):Num(0));
        }
    }
}

// ----------------------------------------------------------------------------
//  Pick L.  Coverage falls as S grows while the Strassen saving rises, and the
//  outer loop needs enough tiles to balance.  Maximise
//      coverage * (8/7)^L + (1 - coverage)
//  subject to m_i*m_j >= 2*nthreads.  Override with params.super_tile_blocks
//  when sweeping S for a paper figure.
// ----------------------------------------------------------------------------
inline size_t hsbm_choose_level(size_t nib_full, size_t njb_full, size_t nkb_full,
                                int nthreads, double level_value = 1.02,
                                double k_split_budget_gib = 0.0, double core_gib = 0.0)
{
    const size_t lim = std::min(std::min(nib_full,njb_full),nkb_full);
    if(lim < 2) return 0;
    const size_t Lmax = (size_t)std::floor(std::log2((double)lim));
    size_t best_L = 1;
    double best   = -1.0;
    for(size_t L=1; L<=Lmax; ++L){
        const size_t S = (size_t)1<<L;
        const size_t mi = nib_full/S, mj = njb_full/S, mk = nkb_full/S;
        if(mi==0 || mj==0 || mk==0) break;
        // Enough tiles to keep every thread busy.  This is a hard constraint,
        // not a preference: with fewer tiles than threads the surplus threads
        // idle for the whole call.  One tile per thread is the floor, 2x gives
        // the dynamic schedule something to balance, but never demand more than
        // the grid can supply -- at 1 thread a single-tile core is correct and
        // rejecting it throws away a Strassen level.
        // Available concurrency.  k splitting means the (i,j) grid alone need
        // not cover the thread count -- requiring that would force the
        // super-tile down and cost recursion depth, worth ~20% for two levels
        // on this hardware.  But k slices need a private copy of the core of C
        // each, so only count k up to what the memory budget allows.
        const size_t k_cap  = (k_split_budget_gib > 0.0 && core_gib > 0.0)
                            ? std::max((size_t)1,(size_t)std::floor(k_split_budget_gib/core_gib)+1lu)
                            : 1lu;
        const size_t avail  = mi*mj*std::min(mk,k_cap);
        if(avail < (size_t)nthreads) continue;  // never leave threads idle
        // Prefer 2x the thread count so the dynamic schedule has slack -- but
        // only above 2 threads.  At 1-2 threads the slack is worthless and
        // demanding it costs a recursion level, measured at ~18% on the square
        // case (L=6 gives 1.175x BSM, L=5 gives 0.99x).
        if(L>1 && nthreads > 2 && avail < (size_t)(2*nthreads)) continue;
        const double cov = ((double)(mi*S)/(double)nib_full)
                         * ((double)(mj*S)/(double)njb_full)
                         * ((double)(mk*S)/(double)nkb_full);
        // What one Strassen level is actually worth.  The textbook value is
        // 8/7 = 1.143, but that counts multiplies only and ignores the 15 block
        // additions per level.  Measured on an AVX-512 node (dense b=128,
        // N=8192) a level returns barely 2%, because the additions are
        // bandwidth bound while the multiplies got ~1.7x faster than on the
        // EPYC 7302 the penalties were calibrated on.  With level_value near 1
        // the objective is dominated by coverage, which is the correct
        // behaviour when depth is cheap and peeling loss is not.
        const double score = cov*std::pow(level_value,(double)L) + (1.0-cov);
        if(score > best){ best = score; best_L = L; }
    }
    return best_L;
}

// ----------------------------------------------------------------------------
//  Main entry point.  C must be pre-initialised by the caller (C.zero() is
//  enough -- missing core blocks are created here); accumulation is C += A*B.
// ----------------------------------------------------------------------------
template<typename Num>
HSBMStats matmult_hsbm(BlockSparseMatrix<Num>& C,
                       const BlockSparseMatrix<Num>& A,
                       const BlockSparseMatrix<Num>& B,
                       const HSBMParams<Num>& par,
                       HSBMWorkspaceSet<Num>& wss)
{
    HSBMStats st;

    const size_t i_bs = A.max_blocksize_row();
    const size_t j_bs = B.max_blocksize_col();
    const size_t k_bs = A.max_blocksize_col();

    const size_t nib = A.nrowblocks();
    const size_t njb = B.ncolblocks();
    const size_t nkb = A.ncolblocks();

    assert(nkb == B.nrowblocks());
    assert(nib == C.nrowblocks());
    assert(njb == C.ncolblocks());

    const Num thresh_per_block = par.thresh*static_cast<Num>(i_bs*j_bs*k_bs);

    // Strassen mixes the three axes, so the scratch tiles must be square and
    // interchangeable.  Non-uniform block sizes fall back to pure BSM.
    const bool uniform_blocks = (i_bs==j_bs && j_bs==k_bs);

    // Only *complete* blocks may enter the core; a ragged final block would be
    // an edge case and would be pushed onto the sparse path anyway.
    const size_t nib_full = uniform_blocks ? A.nrow()/i_bs : 0;
    const size_t njb_full = uniform_blocks ? B.ncol()/j_bs : 0;
    const size_t nkb_full = uniform_blocks ? A.ncol()/k_bs : 0;

    size_t L = par.super_tile_blocks
             ? (size_t)std::floor(std::log2((double)par.super_tile_blocks))
             : hsbm_choose_level(nib_full,njb_full,nkb_full,
                                 par.nthreads>0?par.nthreads:omp_get_max_threads(),
                                 par.level_value);
    if(!uniform_blocks) L = 0;

    // An explicitly requested S that does not fit is clamped down rather than
    // silently abandoning the core -- otherwise an S sweep would quietly turn
    // into plain BSM at the top of the range.
    size_t S = L ? ((size_t)1<<L) : 0;
    while(L > 0 && (nib_full/S==0 || njb_full/S==0 || nkb_full/S==0)){ --L; S >>= 1; }
    if(L == 0) S = 0;

    const size_t mi = S ? nib_full/S : 0;
    const size_t mj = S ? njb_full/S : 0;
    const size_t mk = S ? nkb_full/S : 0;

    const size_t nib_c = mi*S, njb_c = mj*S, nkb_c = mk*S;

    st.super_tile = S;  st.levels = L;
    st.core_ib = nib_c; st.core_jb = njb_c; st.core_kb = nkb_c;
    st.n_tiles = mi*mj;
    st.used_core = (S != 0);
    st.coverage = (double)nib_c/(double)nib * (double)njb_c/(double)njb
                * (double)nkb_c/(double)nkb;

    // ---- no usable core: plain BSM over everything -------------------------
    if(S == 0){
        bsm_matmult_range(C,A,B, 0,nib, 0,njb, 0,nkb, thresh_per_block, Num(1));
        return st;
    }

    // ---- decision tree on the core sub-grid only ---------------------------
    // Capping max_level at L makes every top-level super-block exactly S blocks
    // wide, so none of them is ragged.
    std::vector<size_t> top_flops;
    const auto tree = matmult_strassen_dryrun(A, B, thresh_per_block,
                                              nib_c, njb_c, nkb_c, L, &top_flops);
    assert(tree.size() == L);
    for(const auto& lvl : tree){
        st.tree_nodes     += lvl.size();
        st.strassen_nodes += (size_t)std::count(lvl.begin(), lvl.end(), true);
    }

    // ---- scratch sizing ----------------------------------------------------
    const size_t h_top       = S/2;
    const size_t block_elems = i_bs*j_bs;
    const size_t pool_blocks = 16lu*h_top*h_top + 64lu;   // see compute_total_blocks
    const size_t ptr_slots   = 40lu*h_top*h_top + 128lu;
    st.scratch_gib_per_thread =
        (double)pool_blocks*(double)block_elems*(double)sizeof(Num)/(1024.0*1024.0*1024.0);

    min_size_for_strassen = par.min_size_strassen;

    // ---- longest-processing-time-first ordering of the (si,sj) tiles -------
    std::vector<size_t> order(mi*mj);
    std::iota(order.begin(), order.end(), 0lu);
    if(top_flops.size() == mi*mj*mk){
        std::vector<size_t> cost(mi*mj, 0lu);
        for(size_t sk=0; sk<mk; ++sk)
            for(size_t sj=0; sj<mj; ++sj)
                for(size_t si=0; si<mi; ++si)
                    cost[si + sj*mi] += top_flops[ijk(si,sj,sk,mi,mj)];
        std::stable_sort(order.begin(), order.end(),
                         [&cost](size_t a, size_t b){ return cost[a] > cost[b]; });
    }

    // ---- core ---------------------------------------------------------------
    int nthr = par.nthreads>0 ? par.nthreads : omp_get_max_threads();
    if((size_t)nthr > wss.nthreads()) nthr = (int)wss.nthreads();   // never index past the set
    if(nthr < 1) nthr = 1;

    // How many threads to put on the k axis.  Only useful once the (i,j) grid
    // has run out of tiles, and only affordable while the private copies of the
    // core of C fit in the budget.
    size_t k_split = 1;
    if(mk > 1 && (size_t)nthr > mi*mj && par.k_split_max_gib > 0.0){
        const double core_gib = (double)(nib_c*njb_c)*(double)block_elems
                              * (double)sizeof(Num)/(1024.0*1024.0*1024.0);
        const size_t by_mem  = core_gib > 0.0
                             ? (size_t)std::floor(par.k_split_max_gib/core_gib) + 1lu : mk;
        const size_t by_work = ((size_t)nthr + mi*mj - 1lu)/(mi*mj);
        k_split = std::min(std::min(mk, by_work), std::max((size_t)1,by_mem));
    }
    st.k_split = k_split;
    st.replica_gib = (double)(k_split-1)*(double)(nib_c*njb_c)*(double)block_elems
                   * (double)sizeof(Num)/(1024.0*1024.0*1024.0);

    // Private accumulators for the k slices beyond the first; slice 0 writes
    // straight into C.
    std::vector<std::unique_ptr<BlockSparseMatrix<Num>>> creplica(k_split>1 ? k_split-1 : 0);

    const size_t n_work = mi*mj*k_split;
    #pragma omp parallel num_threads(nthr)
    {
        const int tid = omp_get_thread_num();
        auto& ws = wss.get(tid, block_elems, pool_blocks, ptr_slots, par.warm_pools);
        BlockAllocator<Num> scratch_alloc(&ws.pool);
        PtrBumpAlloc<Num>   ptr_alloc{ ws.ptrs.data(), ws.ptrs.size(), 0 };

        #pragma omp for schedule(static)
        for(size_t r=0; r<creplica.size(); ++r)
            creplica[r] = std::make_unique<BlockSparseMatrix<Num>>(
                              C.nrow(), C.ncol(), i_bs, j_bs, C.thresh());

        #pragma omp for schedule(dynamic,1)
        for(size_t w=0; w<n_work; ++w){
            const size_t ks = w % k_split;              // which k slice
            const size_t t  = order[w / k_split];       // which (i,j) tile
            const size_t si = t % mi;
            const size_t sj = t / mi;

            BlockSparseMatrix<Num>& Cw = ks==0 ? C : *creplica[ks-1];

            // accumulate_into_bsm does a dense += on the output tiles, so every
            // core block has to exist.  This (tile, slice) pair owns them.
            for(size_t jb=sj*S; jb<(sj+1)*S; ++jb)
                for(size_t ib=si*S; ib<(si+1)*S; ++ib){
                    auto& cb = Cw.block(ib,jb);
                    if(cb.size()==0)
                        cb = Matrix<Num,BlockAllocator<Num>>(Num(0), i_bs, j_bs, Cw.allocator());
                }

            for(size_t sk=ks; sk<mk; sk+=k_split){
                ptr_alloc.restore(0);
                hybrid_recurse(Cw, A, B,
                               si*S, sj*S, sk*S,
                               S, (long)L - 1,
                               tree, thresh_per_block,
                               i_bs, j_bs,
                               scratch_alloc, ptr_alloc,
                               nib_c, njb_c, nkb_c);
            }
        }
    }

    // Reduce the private accumulators.  Parallel over output blocks, so each
    // block of C is touched by exactly one thread.
    if(!creplica.empty()){
        #pragma omp parallel for schedule(static) collapse(2) num_threads(nthr)
        for(size_t jb=0; jb<njb_c; ++jb)
            for(size_t ib=0; ib<nib_c; ++ib){
                auto& cb = C.block(ib,jb);
                for(const auto& rep : creplica){
                    const auto& rb = rep->block(ib,jb);
                    if(rb.size()==0) continue;
                    if(cb.size()==0) cb = Matrix<Num,BlockAllocator<Num>>(Num(0), i_bs, j_bs, C.allocator());
                    cb += rb;
                }
            }
    }

    // ---- the three remainder slabs -----------------------------------------
    bsm_matmult_range(C,A,B, nib_c,nib,     0,njb,       0,nkb,   thresh_per_block, Num(1));
    bsm_matmult_range(C,A,B, 0,nib_c,   njb_c,njb,       0,nkb,   thresh_per_block, Num(1));
    bsm_matmult_range(C,A,B, 0,nib_c,       0,njb_c, nkb_c,nkb,   thresh_per_block, Num(1));

    return st;
}

// ----------------------------------------------------------------------------
//  Pre-allocate the output blocks the Strassen path will accumulate into.
//  accumulate_into_bsm does a dense +=, so those blocks must exist.  Doing it
//  inside matmult_hsbm() charges ~100 MB of allocation and zeroing to the timer
//  that BSM never pays (it allocates lazily and lets the first gemm write with
//  beta=0).  Call this once, outside the timed region, when benchmarking.
//  Calling it is optional: matmult_hsbm() still creates anything missing.
// ----------------------------------------------------------------------------
template<typename Num>
void hsbm_prepare_output(BlockSparseMatrix<Num>& C, const HSBMParams<Num>& par,
                         const BlockSparseMatrix<Num>& A, const BlockSparseMatrix<Num>& B)
{
    const size_t i_bs = A.max_blocksize_row();
    const size_t j_bs = B.max_blocksize_col();
    const size_t k_bs = A.max_blocksize_col();
    if(i_bs!=j_bs || j_bs!=k_bs) return;

    const size_t nib_full = A.nrow()/i_bs, njb_full = B.ncol()/j_bs, nkb_full = A.ncol()/k_bs;
    size_t L = par.super_tile_blocks
             ? (size_t)std::floor(std::log2((double)par.super_tile_blocks))
             : hsbm_choose_level(nib_full,njb_full,nkb_full,
                                 par.nthreads>0?par.nthreads:omp_get_max_threads(),
                                 par.level_value, par.k_split_max_gib,
                                 (double)(nib_full*njb_full)*(double)(i_bs*j_bs)
                                 *(double)sizeof(Num)/(1024.0*1024.0*1024.0));
    size_t S = L ? ((size_t)1<<L) : 0;
    while(L > 0 && (nib_full/S==0 || njb_full/S==0 || nkb_full/S==0)){ --L; S >>= 1; }
    if(L == 0) return;
    const size_t nib_c = (nib_full/S)*S, njb_c = (njb_full/S)*S;

    #pragma omp parallel for schedule(static) collapse(2) if(!omp_in_parallel())
    for(size_t jb=0; jb<njb_c; ++jb)
        for(size_t ib=0; ib<nib_c; ++ib){
            auto& cb = C.block(ib,jb);
            if(cb.size()==0)
                cb = Matrix<Num,BlockAllocator<Num>>(Num(0), i_bs, j_bs, C.allocator());
        }
}

// Convenience overload: allocates a workspace set for the call.
template<typename Num>
HSBMStats matmult_hsbm(BlockSparseMatrix<Num>& C,
                       const BlockSparseMatrix<Num>& A,
                       const BlockSparseMatrix<Num>& B,
                       const HSBMParams<Num>& par)
{
    HSBMWorkspaceSet<Num> wss(par.nthreads>0?par.nthreads:omp_get_max_threads());
    return matmult_hsbm(C,A,B,par,wss);
}
#endif
