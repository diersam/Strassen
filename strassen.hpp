#ifndef STRASSEN_HPP
#define STRASSEN_HPP
#include "blocksparsematrix.h"
#include "blocksparsematrix.hpp"
#include "utils.hpp"
#include <cassert>
#include <cstdio>
#include <vector>
#include <cmath>
#include <array>

// ============================================================================
//  Hybrid Strassen-Sparse block matrix multiplication.
//
//  ONE recursive function consults a per-level dry-run decision tree to choose
//  between Winograd's 7-multiply scheme and a sparse-aware 8-multiply quadtree
//  descent at every super-block. Once Strassen is chosen at a node, the entire
//  subtree below it stays on the Strassen path (the tree is only consulted on
//  the 8-path -- "Option A").
//
//  Public entry points:
//    matmult_strassen_sparse(C, A, B, thresh, alpha, beta)   -- main hybrid call
//    matmult_strassen_dryrun(A, B, thresh)                   -- builds tree
// ============================================================================

template<typename Num>
using Mat = Matrix<Num, BlockAllocator<Num>>;

// ----------------------------------------------------------------------------
//  Lightweight matrix views used by the BLAS leaf
// ----------------------------------------------------------------------------
template<typename Num>
struct MatView      { Num*       data; size_t nrow, ncol, ld; };
template<typename Num>
struct ConstMatView { const Num* data; size_t nrow, ncol, ld; };

template<typename Num>
ConstMatView<Num> as_view(const Mat<Num>& m) { return { m.data_ptr(), m.nrow(), m.ncol(), m.nrow() }; }
template<typename Num>
MatView<Num>      as_view(Mat<Num>& m)       { return { m.data_ptr(), m.nrow(), m.ncol(), m.nrow() }; }

template<typename Num>
inline void gemm_block(MatView<Num>& C, const ConstMatView<Num>& A, const ConstMatView<Num>& B,
                       const Num alpha, const Num beta)
{
    const char tA='n', tB='n';
    const int m=(int)C.nrow, n=(int)C.ncol, k=(int)A.ncol;
    const int lda=(int)A.ld, ldb=(int)B.ld, ldc=(int)C.ld;
    any_gemm(&tA,&tB,&m,&n,&k,&alpha,A.data,&lda,B.data,&ldb,&beta,C.data,&ldc);
}

// ----------------------------------------------------------------------------
//  Block grids: a uniform [nrb x ncb] array of pointers / Mat objects laid
//  out column-major. ConstBlockGrid borrows pointers, OwnedBlockGrid owns
//  Mat scratch tiles allocated from a BlockAllocator pool.
// ----------------------------------------------------------------------------
template<typename Num>
struct ConstBlockGrid {
    const Mat<Num>* const* blocks;
    size_t nrb, ncb, stride;
    const Mat<Num>& block(size_t rb, size_t cb) const { return *blocks[rb + cb*stride]; }
    ConstBlockGrid quadrant(size_t rb0, size_t cb0, size_t h_r, size_t h_c) const {
        return { blocks + rb0 + cb0*stride, h_r, h_c, stride };
    }
};

template<typename Num>
struct PtrBumpAlloc {
    const Mat<Num>** buf;
    size_t           capacity, offset;
    const Mat<Num>** alloc(size_t n) {
        assert(offset + n <= capacity);
        const Mat<Num>** p = buf + offset;
        offset += n;
        return p;
    }
    size_t save()            const { return offset; }
    void   restore(size_t s)       { offset = s; }
};

template<typename Num>
struct OwnedBlockGrid {
    std::vector<Mat<Num>>  data;
    const Mat<Num>**       ptrs;
    size_t nrb, ncb;

    OwnedBlockGrid() = default;
    OwnedBlockGrid(const Num val, size_t nrb_, size_t ncb_,
                   size_t block_nr, size_t block_nc,
                   const BlockAllocator<Num>& alloc,
                   PtrBumpAlloc<Num>& ptr_alloc)
      : nrb(nrb_), ncb(ncb_)
    {
        data.reserve(nrb*ncb);
        for(size_t i=0;i<nrb*ncb;++i)
            data.emplace_back(val, block_nr, block_nc, alloc);
        ptrs = ptr_alloc.alloc(nrb*ncb);
        for(size_t i=0;i<nrb*ncb;++i) ptrs[i] = &data[i];
    }
    OwnedBlockGrid(size_t nrb_, size_t ncb_,
                   size_t block_nr, size_t block_nc,
                   const BlockAllocator<Num>& alloc,
                   PtrBumpAlloc<Num>& ptr_alloc)
      : nrb(nrb_), ncb(ncb_)
    {
        data.reserve(nrb*ncb);
        for(size_t i=0;i<nrb*ncb;++i)
            data.emplace_back(block_nr, block_nc, alloc);
        ptrs = ptr_alloc.alloc(nrb*ncb);
        for(size_t i=0;i<nrb*ncb;++i) ptrs[i] = &data[i];
    }
    Mat<Num>&       block(size_t rb, size_t cb)       { return data[rb + cb*nrb]; }
    const Mat<Num>& block(size_t rb, size_t cb) const { return data[rb + cb*nrb]; }
    ConstBlockGrid<Num> view() const { return { ptrs, nrb, ncb, nrb }; }
    ConstBlockGrid<Num> quadrant(size_t rb0, size_t cb0, size_t h_r, size_t h_c) const {
        return { ptrs + rb0 + cb0*nrb, h_r, h_c, nrb };
    }
};

// ----------------------------------------------------------------------------
//  Build a ConstBlockGrid view onto a [nrb x ncb] region of a BSM.
//  Pointer table is allocated from the bump-pool so it lives for the
//  duration of the call (no heap traffic per super-block).
// ----------------------------------------------------------------------------
template<typename Num>
ConstBlockGrid<Num> bsm_quadrant_view(const BlockSparseMatrix<Num>& M,
                                       size_t ib0, size_t jb0,
                                       size_t nrb, size_t ncb,
                                       PtrBumpAlloc<Num>& ptr_alloc)
{
    const Mat<Num>** p = ptr_alloc.alloc(nrb*ncb);
    for(size_t cb=0;cb<ncb;++cb)
        for(size_t rb=0;rb<nrb;++rb)
            p[rb + cb*nrb] = &M.block(ib0+rb, jb0+cb);
    return { p, nrb, ncb, nrb };
}

// ----------------------------------------------------------------------------
//  Grid-level add/sub: R = A op B over corresponding tiles.
//  (Each tile is itself a Mat<Num> dense block; += / -= are BLAS axpy.)
// ----------------------------------------------------------------------------
template<typename Num, typename GridA, typename GridB>
void subgrid_sub(OwnedBlockGrid<Num>& R, const GridA& A, const GridB& B)
{
    for(size_t cb=0;cb<R.ncb;++cb)
        for(size_t rb=0;rb<R.nrb;++rb){
            R.block(rb,cb)  = A.block(rb,cb);
            R.block(rb,cb) -= B.block(rb,cb);
        }
}


template<typename Num, typename Grid>
void strassen_prepare_A_side(
    OwnedBlockGrid<Num>& na_pc,OwnedBlockGrid<Num>& pc_pd,OwnedBlockGrid<Num>& na_pc_pd,OwnedBlockGrid<Num>& pa_pb_nc_nd,
    const Grid& a,const Grid& b,const Grid& c,const Grid& d)
{
  assert(na_pc.nrb*na_pc.ncb > 0);
  const size_t total_block_size = na_pc.block(0,0).size();
    for(size_t cb=0;cb<na_pc.ncb;++cb){
        for(size_t rb=0;rb<pc_pd.nrb;++rb){
            Num* __restrict__ na_pc_ptr       = na_pc.block(rb,cb).data_ptr();
            Num* __restrict__ pc_pd_ptr       = pc_pd.block(rb,cb).data_ptr();
            Num* __restrict__ na_pc_pd_ptr    = na_pc_pd.block(rb,cb).data_ptr();
            Num* __restrict__ pa_pb_nc_nd_ptr = pa_pb_nc_nd.block(rb,cb).data_ptr();

            const Num* __restrict__ a_ptr = a.block(rb,cb).data_ptr();
            const Num* __restrict__ b_ptr = b.block(rb,cb).data_ptr();
            const Num* __restrict__ c_ptr = c.block(rb,cb).data_ptr();
            const Num* __restrict__ d_ptr = d.block(rb,cb).data_ptr();
            #pragma GCC ivdep
            for(size_t i=0;i<total_block_size;++i){
                na_pc_ptr[i]    = c_ptr[i] - a_ptr[i];
                const Num pc_pd_val = c_ptr[i] + d_ptr[i];
                pc_pd_ptr[i]    = pc_pd_val;
                const Num na_pc_pd_val = pc_pd_val - a_ptr[i];
                na_pc_pd_ptr[i] = na_pc_pd_val;
                pa_pb_nc_nd_ptr[i] = b_ptr[i] - na_pc_pd_val;
            }
        }
    }
}

template<typename Num, typename Grid>
void strassen_prepare_B_side(
    OwnedBlockGrid<Num>& pC_nD,OwnedBlockGrid<Num>& nA_pC,OwnedBlockGrid<Num>& pA_nC_pD,OwnedBlockGrid<Num>& nA_pB_pC_nD,
    const Grid& A,const Grid& B,const Grid& C,const Grid& D)
{
  assert(pC_nD.nrb*pC_nD.ncb > 0);
  const size_t total_block_size = pC_nD.block(0,0).size();
    for(size_t cb=0;cb<pC_nD.ncb;++cb){
        for(size_t rb=0;rb<pC_nD.nrb;++rb){
            Num* __restrict__ pC_nD_ptr       = pC_nD.block(rb,cb).data_ptr();
            Num* __restrict__ nA_pC_ptr       = nA_pC.block(rb,cb).data_ptr();
            Num* __restrict__ pA_nC_pD_ptr    = pA_nC_pD.block(rb,cb).data_ptr();
            Num* __restrict__ nA_pB_pC_nD_ptr = nA_pB_pC_nD.block(rb,cb).data_ptr();

            const Num* __restrict__ A_ptr = A.block(rb,cb).data_ptr();
            const Num* __restrict__ B_ptr = B.block(rb,cb).data_ptr();
            const Num* __restrict__ C_ptr = C.block(rb,cb).data_ptr();
            const Num* __restrict__ D_ptr = D.block(rb,cb).data_ptr();
            #pragma GCC ivdep
            for(size_t i=0;i<total_block_size;++i){
              pC_nD_ptr[i]           = C_ptr[i] - D_ptr[i];
              const Num nA_pC_val    = C_ptr[i] - A_ptr[i];
              nA_pC_ptr[i]           = nA_pC_val;
              const Num pA_nC_pD_val = D_ptr[i] - nA_pC_val;
              pA_nC_pD_ptr[i]        = pA_nC_pD_val;
              nA_pB_pC_nD_ptr[i]     = B_ptr[i] - pA_nC_pD_val;
            }
        }
    }
}

template<typename Num>
void strassen_post_process_C_side(
    OwnedBlockGrid<Num>& C00,OwnedBlockGrid<Num>& C10,OwnedBlockGrid<Num>& C01,OwnedBlockGrid<Num>& C11)
{
  auto& nw = C11;//nw uses C11 as storage
  assert(nw.nrb*nw.ncb > 0);
  const size_t total_block_size = nw.block(0,0).size();
    for(size_t cb=0;cb<nw.ncb;++cb){
        for(size_t rb=0;rb<nw.nrb;++rb){
            Num* __restrict__ nw_ptr       = nw.block(rb,cb).data_ptr();
            Num* __restrict__ C00_ptr      = C00.block(rb,cb).data_ptr();
            Num* __restrict__ C01_ptr      = C01.block(rb,cb).data_ptr();
            Num* __restrict__ C10_ptr      = C10.block(rb,cb).data_ptr();
            #pragma GCC ivdep
            for(size_t i=0;i<total_block_size;++i){
              const double pC11_pC00 = nw_ptr[i] + C00_ptr[i];
              C10_ptr[i] += pC11_pC00;
              nw_ptr[i]  = C01_ptr[i] + C10_ptr[i];
              C01_ptr[i] += pC11_pC00;
            }
        }
    }
}

template<typename Num, typename GridA, typename GridB>
void subgrid_add(OwnedBlockGrid<Num>& R, const GridA& A, const GridB& B)
{
    for(size_t cb=0;cb<R.ncb;++cb)
        for(size_t rb=0;rb<R.nrb;++rb){
            R.block(rb,cb)  = A.block(rb,cb);
            R.block(rb,cb) += B.block(rb,cb);
        }
}

template<typename Num>
void accumulate_into_bsm(BlockSparseMatrix<Num>& M, const OwnedBlockGrid<Num>& G,
                         size_t ib0, size_t jb0)
{
    for(size_t cb=0;cb<G.ncb;++cb)
        for(size_t rb=0;rb<G.nrb;++rb)
            M.block(ib0+rb, jb0+cb) += G.block(rb,cb);
}

// ----------------------------------------------------------------------------
//  Pool sizing helper: bound on number of scratch Mat tiles needed for a
//  Strassen recursion of given depth on grids of side h. At each Winograd
//  level we add 13 grids of size (h_at_level)^2 (8 A/B temps + 4 C outputs
//  + 1 nw helper) and spawn 7 child calls.
// ----------------------------------------------------------------------------
inline size_t compute_total_blocks(size_t h, size_t depth)
{
    size_t total=0, grid_h=h, n_calls=1;
    for(size_t l=0;l<depth;++l){
        total  += (size_t)13 * n_calls * grid_h * grid_h;
        grid_h  = (grid_h > 1) ? grid_h/2 : 1;
        n_calls *= 7;
    }
    return total;
}

// ============================================================================
//  STRASSEN-PATH RECURSION
//
//  Once the tree votes Strassen at some super-block, this pair of mutually-
//  recursive functions takes over and stays on the Strassen path all the way
//  down to single-block dgemms. The tree is NOT consulted below this point.
// ============================================================================
template<typename Num, typename GridA, typename GridB>
void strassen_recurse(OwnedBlockGrid<Num>& C00, OwnedBlockGrid<Num>& C01,
                      OwnedBlockGrid<Num>& C10, OwnedBlockGrid<Num>& C11,
                      const GridA& a, const GridA& b, const GridA& c, const GridA& d,
                      const GridB& A, const GridB& C, const GridB& B, const GridB& D,
                      size_t block_nr, size_t block_nc,
                      const BlockAllocator<Num>& scratch_alloc,
                      PtrBumpAlloc<Num>& ptr_alloc);

// One step of the Strassen path: accumulate A_grid * B_grid into C_grid.
// Leaf: 1x1x1 -> single dgemm. Otherwise split 2-way per axis if possible
// and Winograd-recurse, else fall back to dense 8-product over kb (no sparse
// skip on this path -- operands may be sums whose norm cache is stale).
extern size_t min_size_for_strassen;
template<typename Num, typename GridA, typename GridB>
void strassen_matmult_acc(OwnedBlockGrid<Num>& C, const GridA& A, const GridB& B,
                          size_t block_nr, size_t block_nc,
                          const BlockAllocator<Num>& scratch_alloc,
                          PtrBumpAlloc<Num>& ptr_alloc)
{
    const size_t nrb=C.nrb, nkb=A.ncb, ncb=C.ncb;

    if(nrb==1 && nkb==1 && ncb==1){
        auto cv = as_view(C.block(0,0));
        const auto av = as_view(A.block(0,0));
        const auto bv = as_view(B.block(0,0));
        gemm_block(cv, av, bv, Num(1), Num(1));
        return;
    }
    const bool too_small_for_strassen = nrb*block_nr <= min_size_for_strassen;
    if(nrb%2!=0 || nkb%2!=0 || ncb%2!=0 || too_small_for_strassen){
        for(size_t cb=0;cb<ncb;++cb)
            for(size_t rb=0;rb<nrb;++rb){
                auto cv = as_view(C.block(rb,cb));
                for(size_t kb=0;kb<nkb;++kb){
                    const auto av = as_view(A.block(rb,kb));
                    const auto bv = as_view(B.block(kb,cb));
                    gemm_block(cv, av, bv, Num(1), Num(1));
                }
            }
        return;
    }

    const size_t h_r=nrb/2, h_k=nkb/2, h_c=ncb/2;
    const auto a_q=A.quadrant(0,   0,   h_r,h_k);
    const auto b_q=A.quadrant(0,   h_k, h_r,h_k);
    const auto c_q=A.quadrant(h_r, 0,   h_r,h_k);
    const auto d_q=A.quadrant(h_r, h_k, h_r,h_k);
    const auto A_q=B.quadrant(0,   0,   h_k,h_c);
    const auto C_q=B.quadrant(0,   h_c, h_k,h_c);
    const auto B_q=B.quadrant(h_k, 0,   h_k,h_c);
    const auto D_q=B.quadrant(h_k, h_c, h_k,h_c);

    const size_t ptr_mark=ptr_alloc.save();
    OwnedBlockGrid<Num> S00(0.e0,h_r,h_c,block_nr,block_nc,scratch_alloc,ptr_alloc);
    OwnedBlockGrid<Num> S01(0.e0,h_r,h_c,block_nr,block_nc,scratch_alloc,ptr_alloc);
    OwnedBlockGrid<Num> S10(0.e0,h_r,h_c,block_nr,block_nc,scratch_alloc,ptr_alloc);
    OwnedBlockGrid<Num> S11(0.e0,h_r,h_c,block_nr,block_nc,scratch_alloc,ptr_alloc);

    strassen_recurse(S00,S01,S10,S11,
                     a_q,b_q,c_q,d_q, A_q,C_q,B_q,D_q,
                     block_nr,block_nc,scratch_alloc,ptr_alloc);

    for(size_t cb=0;cb<h_c;++cb)
        for(size_t rb=0;rb<h_r;++rb){
            C.block(rb,    cb    ) += S00.block(rb,cb);
            C.block(rb,    cb+h_c) += S01.block(rb,cb);
            C.block(rb+h_r,cb    ) += S10.block(rb,cb);
            C.block(rb+h_r,cb+h_c) += S11.block(rb,cb);
        }
    ptr_alloc.restore(ptr_mark);
}

// Winograd's 7-multiply variant. Mapping: (a b; c d) and (A C; B D) are the
// quadrants of the A and B operands respectively (top-left, top-right,
// bot-left, bot-right). The 7 products are M1..M7 below, combined per the
// standard Winograd recombination.
template<typename Num, typename GridA, typename GridB>
void strassen_recurse(OwnedBlockGrid<Num>& C00, OwnedBlockGrid<Num>& C01,
                      OwnedBlockGrid<Num>& C10, OwnedBlockGrid<Num>& C11,
                      const GridA& a, const GridA& b, const GridA& c, const GridA& d,
                      const GridB& A, const GridB& C, const GridB& B, const GridB& D,
                      size_t block_nr, size_t block_nc,
                      const BlockAllocator<Num>& scratch_alloc,
                      PtrBumpAlloc<Num>& ptr_alloc)
{
    const size_t nrb=C00.nrb, nkb=a.ncb, ncb=C00.ncb;
    const size_t ptr_mark=ptr_alloc.save();

    auto make_A=[&](){ return OwnedBlockGrid<Num>(nrb,nkb,block_nr,block_nc,scratch_alloc,ptr_alloc); };
    auto make_B=[&](){ return OwnedBlockGrid<Num>(nkb,ncb,block_nr,block_nc,scratch_alloc,ptr_alloc); };

    auto na_pc      =make_A();
    auto pc_pd      =make_A();
    auto na_pc_pd   =make_A();
    auto pa_pb_nc_nd=make_A();
    strassen_prepare_A_side(na_pc,pc_pd,na_pc_pd,pa_pb_nc_nd,a,b,c,d);

    auto pC_nD      =make_B();
    auto nA_pC      =make_B();
    auto pA_nC_pD   =make_B();
    auto nA_pB_pC_nD=make_B();
    strassen_prepare_B_side(pC_nD,nA_pC,pA_nC_pD,nA_pB_pC_nD,A,B,C,D);

    auto PROD=[&](OwnedBlockGrid<Num>& Cout,
                  const auto& Ain, const auto& Bin){
        strassen_matmult_acc(Cout, Ain, Bin,
                             block_nr, block_nc, scratch_alloc, ptr_alloc);
    };

    PROD(C00, a, A);                           // M1 = a*A     -> C00
    PROD(C10, na_pc, pC_nD);                   // M2 = (c-a)(C-D)            -> C10
    PROD(C01, pc_pd, nA_pC);                   // M3 = (c+d)(C-A)            -> C01
    auto& nw=C11;//nw uses C11 as storage
    PROD(nw,  na_pc_pd, pA_nC_pD);             // M4 = (c+d-a)(A-C+D)        -> nw += M4

    strassen_post_process_C_side(C00,C10,C01,C11);

    PROD(C01, pa_pb_nc_nd.view(), D);          // M5 = (a+b-c-d)*D   -> C01 +=
    PROD(C10, d, nA_pB_pC_nD.view());          // M6 = d*(-A+B+C-D)  -> C10 +=
    PROD(C00, b, B);                           // M7 = b*B           -> C00 +=

    ptr_alloc.restore(ptr_mark);
}

// ============================================================================
//  HYBRID 8-PATH RECURSION
//
//  Walks the dry-run tree top-down. At each super-block:
//    - tree says "Strassen" and not edge: build 8 BSM-quadrant views and 4
//      output scratch grids, hand off to strassen_recurse, accumulate into
//      C_mat.
//    - otherwise (or if edge): split 8-way; each child recurses with
//      level-1 and the appropriate child super-block index.
//  At level<0 we hit a single-block leaf: one dgemm with sparse skip.
// ============================================================================
template<typename Num>
void hybrid_recurse(BlockSparseMatrix<Num>& C_mat,
                    const BlockSparseMatrix<Num>& A_mat,
                    const BlockSparseMatrix<Num>& B_mat,
                    size_t ib0, size_t jb0, size_t kb0,
                    size_t sb_size,           // super-block side in blocks (= 2^(level+1))
                    long   level,             // tree level; -1 means single-block leaves
                    const std::vector<std::vector<bool>>& tree,
                    const Num thresh,
                    size_t block_nr, size_t block_nc,
                    const BlockAllocator<Num>& scratch_alloc,
                    PtrBumpAlloc<Num>& ptr_alloc)
{
    const size_t nib = A_mat.nrowblocks();
    const size_t njb = B_mat.ncolblocks();
    const size_t nkb = A_mat.ncolblocks();

    if(level < 0){
        if(ib0 >= nib || jb0 >= njb || kb0 >= nkb) return;
        const auto& ab = A_mat.block(ib0,kb0);
        const auto& bb = B_mat.block(kb0,jb0);
        if(ab.size()==0 || bb.size()==0) return;
        if(ab.frobenius_norm() * bb.frobenius_norm() < thresh) return;
        auto cv = as_view(C_mat.block(ib0,jb0));
        const auto av = as_view(ab);
        const auto bv = as_view(bb);
        gemm_block(cv, av, bv, Num(1), Num(1));
        return;
    }

    const size_t n_si = integer_division_round_up(nib, sb_size);
    const size_t n_sj = integer_division_round_up(njb, sb_size);

    const size_t si = ib0 / sb_size;
    const size_t sj = jb0 / sb_size;
    const size_t sk = kb0 / sb_size;
    const bool go_strassen = tree[level][ijk(si,sj,sk,n_si,n_sj)];

    if(go_strassen){
        const size_t h = sb_size / 2;
        const size_t pre_view_mark = ptr_alloc.save();
        auto qa = bsm_quadrant_view(A_mat, ib0,   kb0,   h, h, ptr_alloc);
        auto qb = bsm_quadrant_view(A_mat, ib0,   kb0+h, h, h, ptr_alloc);
        auto qc = bsm_quadrant_view(A_mat, ib0+h, kb0,   h, h, ptr_alloc);
        auto qd = bsm_quadrant_view(A_mat, ib0+h, kb0+h, h, h, ptr_alloc);
        auto qA = bsm_quadrant_view(B_mat, kb0,   jb0,   h, h, ptr_alloc);
        auto qC = bsm_quadrant_view(B_mat, kb0,   jb0+h, h, h, ptr_alloc);
        auto qB = bsm_quadrant_view(B_mat, kb0+h, jb0,   h, h, ptr_alloc);
        auto qD = bsm_quadrant_view(B_mat, kb0+h, jb0+h, h, h, ptr_alloc);

        OwnedBlockGrid<Num> S00(0.e0,h,h,block_nr,block_nc,scratch_alloc,ptr_alloc);
        OwnedBlockGrid<Num> S01(0.e0,h,h,block_nr,block_nc,scratch_alloc,ptr_alloc);
        OwnedBlockGrid<Num> S10(0.e0,h,h,block_nr,block_nc,scratch_alloc,ptr_alloc);
        OwnedBlockGrid<Num> S11(0.e0,h,h,block_nr,block_nc,scratch_alloc,ptr_alloc);

        strassen_recurse(S00,S01,S10,S11,
                         qa,qb,qc,qd, qA,qC,qB,qD,
                         block_nr,block_nc,scratch_alloc,ptr_alloc);

        accumulate_into_bsm(C_mat, S00, ib0,   jb0  );
        accumulate_into_bsm(C_mat, S01, ib0,   jb0+h);
        accumulate_into_bsm(C_mat, S10, ib0+h, jb0  );
        accumulate_into_bsm(C_mat, S11, ib0+h, jb0+h);

        ptr_alloc.restore(pre_view_mark);
    }else{

      // 8-path descent: split each axis in half. At level 0, h=1, child_level=-1
      // and children become single-block leaves. Children that overshoot the
      // matrix are skipped (the leaf branch also guards against this).
      const size_t h = sb_size / 2;
      const long   child_level = level - 1;
      for(size_t dj=0; dj<2; ++dj){
          const size_t cj = jb0 + dj*h;
          if(cj >= njb) continue;
          for(size_t di=0; di<2; ++di){
              const size_t ci = ib0 + di*h;
              if(ci >= nib) continue;
              for(size_t dk=0; dk<2; ++dk){
                  const size_t ck = kb0 + dk*h;
                  if(ck >= nkb) continue;
                  hybrid_recurse(C_mat, A_mat, B_mat,
                                 ci, cj, ck, h, child_level,
                                 tree, thresh,
                                 block_nr, block_nc,
                                 scratch_alloc, ptr_alloc);
              }
          }
      }
    }
}

template<typename Num>
std::vector<std::vector<bool>> matmult_strassen_dryrun(
             const BlockSparseMatrix<Num>& A_mat, 
             const BlockSparseMatrix<Num>& B_mat, 
             const Num thresh)
{
  //no on-the-fly transformation yet
  constexpr bool transA = false;
  constexpr bool transB = false;
  assert(transA == false);
  assert(transB == false);

  //check dimensions
  const size_t ni = transA? A_mat.ncol() : A_mat.nrow();
  const size_t nj = transB? B_mat.nrow() : B_mat.ncol();
  const size_t nk1 = transA? A_mat.nrow() : A_mat.ncol();
  const size_t nk2 = transB? B_mat.ncol() : B_mat.nrow();

  assert(nk1 == nk2);
  const size_t nk = nk1;

  const size_t i_block_size = transA? A_mat.max_blocksize_col() : A_mat.max_blocksize_row();
  const size_t j_block_size = transB? B_mat.max_blocksize_row() : B_mat.max_blocksize_col();

  const size_t k_block_size1 = transA? A_mat.max_blocksize_row() : A_mat.max_blocksize_col();
  const size_t k_block_size2 = transB? B_mat.max_blocksize_col() : B_mat.max_blocksize_row();
  assert(k_block_size1 == k_block_size2);
  const size_t k_block_size = k_block_size1;

  const size_t nib = integer_division_round_up(ni,i_block_size);
  assert(nib == transA? A_mat.ncolblocks() : A_mat.nrowblocks());
  const size_t njb = integer_division_round_up(nj,j_block_size);
  assert(njb == transB? B_mat.nrowblocks() : B_mat.ncolblocks());
  const size_t nkb = integer_division_round_up(nk,k_block_size);
  assert(nkb == transA? A_mat.nrowblocks() : A_mat.ncolblocks());
  assert(nkb == transB? B_mat.ncolblocks() : B_mat.nrowblocks());

  //only quadratic so far
  assert(ni == nj);
  assert(ni == nk);
  assert(nib == njb);
  assert(nib == nkb);

  const size_t max_level = (size_t)std::log2((double)nib+0.5);
  std::vector<std::vector<size_t>> nflops_per_level(max_level+1);
  std::vector<std::vector<bool>> do_Strasssen_decision_tree(max_level);
  {//initiialize at level -1:
    nflops_per_level[0].resize(nib*njb*nkb);
    #pragma omp parallel for schedule(static)
    for(size_t ijkb=0;ijkb<nib*njb*nkb;++ijkb){
      const size_t ib = ijkb%nib;
      const size_t jb = (ijkb/nib)%njb;
      const size_t kb = ijkb/(nib*njb);
      const auto& a_block = transA? A_mat.block(kb,ib) : A_mat.block(ib,kb);
      const auto& b_block = transB? B_mat.block(jb,kb) : B_mat.block(kb,jb);
      const Num norm_a = a_block.frobenius_norm();
      const Num norm_b = b_block.frobenius_norm();
      const Num est = norm_a*norm_b;
      assert(nflops_per_level[0].size() > ijk(ib,jb,kb,nib,njb));
      const auto ni_act = transA? a_block.ncol() : a_block.nrow();
      const auto nj_act = transB? b_block.nrow() : b_block.ncol();
      const auto nk_act = transA? a_block.nrow() : a_block.ncol();
      const size_t nflops_per_block = 2lu*ni_act*nj_act*nk_act;
      nflops_per_level[0][ijk(ib,jb,kb,nib,njb)] = (est >= thresh ? 1lu :0lu)*nflops_per_block;
    }
    //number of FLops on this level (should never increase at higher levels)
    //const size_t n_flops_total = std::accumulate(nflops_per_level[0].cbegin(),nflops_per_level[0].cend(),0lu);
    //printf("total flops on level %lu: %lu\n",0lu, n_flops_total);
  }
  const size_t max_nflops_per_block_mult = 2lu*i_block_size*j_block_size*k_block_size;
  const size_t max_nflops_per_block_add = i_block_size*j_block_size;
  //additions are more expensive per op as mults, how much exactly depends on cache structure
  constexpr std::array<double,20> Strassen_fac = {10,10,10,25,35,35,35,35,35,35,35,35,35,35,35,35,35,35,35,35}; 
  //recursive bottom up to decide between Strassen and sparse at each level iterating the block MM count upwards
  for(size_t level=0;level<max_level;++level){// bottom up
    const size_t previous_super_block_size = (size_t)std::exp2(level);//no of blocks in previous level(1,2,4...)
    const size_t super_block_size = (size_t)std::exp2(level+1);//no of blocks in super-block (2,4,8...)
    const size_t n_flops_per_strassen_mult = (size_t)std::round(std::pow(7,level+1))*max_nflops_per_block_mult;
    const size_t n_flops_per_strassen_add_sub = (size_t)(15e0*Strassen_fac[level]*std::pow(4,level+1)*(double)max_nflops_per_block_add);//we trade 15 adds/subs for one mult
    const size_t n_flops_per_strassen = n_flops_per_strassen_mult + n_flops_per_strassen_add_sub;
    //printf("level: %lu size: %lu n_flops_per_strassen: %lu\n",level,super_block_size,n_flops_per_strassen);

    const size_t n_super_i = integer_division_round_up(nib,super_block_size);
    const size_t n_super_j = integer_division_round_up(njb,super_block_size);
    const size_t n_super_k = integer_division_round_up(nkb,super_block_size);

    const size_t previous_n_super_i = integer_division_round_up(nib,previous_super_block_size);
    const size_t previous_n_super_j = integer_division_round_up(njb,previous_super_block_size);
    //const size_t previous_n_super_k = integer_division_round_up(nkb,previous_super_block_size);

    nflops_per_level[level+1].resize(n_super_i*n_super_j*n_super_k);
    do_Strasssen_decision_tree[level].resize(n_super_i*n_super_j*n_super_k);
    #pragma omp parallel for schedule(static,64)//thread-work-size needs to be big enough to avoid race condition in std::vector<bool>
    for(size_t super_ijk=0;super_ijk<n_super_i*n_super_j*n_super_k;++super_ijk){//over 3d super-multiplication tensor on this level
      const size_t super_i = super_ijk%n_super_i;
      const size_t super_j = (super_ijk/n_super_i)%n_super_j;
      const size_t super_k = super_ijk/(n_super_i*n_super_j);

      //block indices we are starting at
      const size_t ib_start = super_i*super_block_size;
      const size_t jb_start = super_j*super_block_size;
      const size_t kb_start = super_k*super_block_size;

      //block indices we are ending at (accounts for edges with std::min)
      const size_t ib_end   = std::min(ib_start+super_block_size,nib);
      const size_t jb_end   = std::min(jb_start+super_block_size,njb);
      const size_t kb_end   = std::min(kb_start+super_block_size,nkb);

      //number of block indices in this super block
      const size_t ib_size  = ib_end - ib_start;
      const size_t jb_size  = jb_end - jb_start;
      const size_t kb_size  = kb_end - kb_start;

      //matrix indices we are starting at
      const size_t i_start  = ib_start*i_block_size;
      const size_t j_start  = jb_start*j_block_size;
      const size_t k_start  = kb_start*k_block_size;

      //matrix indices we are ending at (accounts for edges with std::min)
      const size_t i_end    = std::min(ib_end*i_block_size,ni);
      const size_t j_end    = std::min(jb_end*j_block_size,nj);
      const size_t k_end    = std::min(kb_end*k_block_size,nk);

      //number of matrix indices in this super block
      const size_t i_size   = i_end - i_start;
      const size_t j_size   = j_end - j_start;
      const size_t k_size   = k_end - k_start;

      //check for edge cases (edge cases if size is not perfect super-block size)
      const bool is_edge_case_i = i_size != i_block_size*super_block_size;
      const bool is_edge_case_j = j_size != j_block_size*super_block_size;
      const bool is_edge_case_k = k_size != k_block_size*super_block_size;
      //if any index is edge-case, we are in an edge case
      const bool is_edge_case = is_edge_case_i || is_edge_case_j || is_edge_case_k;
      //assert(is_edge_case == false);

      //loop over either 1 or 2 previous level super blocks (1 is only possible for the final one)
      const size_t i_sub_size = integer_division_round_up(ib_size,previous_super_block_size);
      const size_t j_sub_size = integer_division_round_up(jb_size,previous_super_block_size);
      const size_t k_sub_size = integer_division_round_up(kb_size,previous_super_block_size);

      //count number of mults for sparse
      size_t nflops = 0lu;
      //over all sub-mults in this 2x2x2 super-block
      for (size_t sub_ib=0;sub_ib<i_sub_size;++sub_ib){//mostly loop over {0,1}
        for (size_t sub_jb=0;sub_jb<j_sub_size;++sub_jb){//mostly loop over {0,1}
          for (size_t sub_kb=0;sub_kb<k_sub_size;++sub_kb){//mostly loop over {0,1}
            //MM counts from previous level
            assert(nflops_per_level[level].size() > ijk(2*super_i+sub_ib,2*super_j+sub_jb,2*super_k+sub_kb,previous_n_super_i,previous_n_super_j));
            nflops += nflops_per_level[level][ijk(2*super_i+sub_ib,2*super_j+sub_jb,2*super_k+sub_kb,previous_n_super_i,previous_n_super_j)];
          }
        }
      }
      //check for sparse or Strassen
      const bool do_Strassen = (nflops >= n_flops_per_strassen) && !is_edge_case;
      do_Strasssen_decision_tree[level][ijk(super_i,super_j,super_k,n_super_i,n_super_j)] = do_Strassen;
      if (do_Strassen){
        //upwards iteration of FLOPs-count
        nflops_per_level[level+1][ijk(super_i,super_j,super_k,n_super_i,n_super_j)] = n_flops_per_strassen;
      }else{//we do conventional
        //upwards iteration of FLOPs-count
        nflops_per_level[level+1][ijk(super_i,super_j,super_k,n_super_i,n_super_j)] = nflops;
      }
    }
    //number of FLops on this level (should never increase at higher levels)
    //const size_t n_flops_total = std::accumulate(nflops_per_level[level+1].cbegin(),nflops_per_level[level+1].cend(),0lu);
    //printf("total flops on level %lu: %lu\n",level+1, n_flops_total);
  }
  return do_Strasssen_decision_tree;
}

// ============================================================================
//  PUBLIC ENTRY POINT
//
//  C = alpha * A * B + beta * C
//  (alpha must be 1; transposes not supported. beta is applied to C up front.)
// ============================================================================
template<typename Num>
void matmult_strassen_sparse(BlockSparseMatrix<Num>& C_mat,
             const BlockSparseMatrix<Num>& A_mat, const bool transA,
             const BlockSparseMatrix<Num>& B_mat, const bool transB,
             const Num thresh, const Num& alpha, const Num& beta)
{
    assert(transA==false); assert(transB==false); assert(alpha==Num(1));
    // Caller is responsible for pre-initialising C (typically via
    // C_mat.fill_with_values(0.0)). We only accumulate. beta is unused;
    // it's kept for signature compatibility with the dense matmult.
    (void)beta;

    const size_t nib = A_mat.nrowblocks();
    const size_t njb = B_mat.ncolblocks();
    const size_t nkb = A_mat.ncolblocks();
    assert(nkb == B_mat.nrowblocks());
    assert(nib == C_mat.nrowblocks());
    assert(njb == C_mat.ncolblocks());
    assert(nib == njb && nib == nkb);

    const size_t i_bs = A_mat.max_blocksize_row();
    const size_t j_bs = B_mat.max_blocksize_col();

    // Build the dry-run tree fresh on every call (timed honestly).
    auto tree = matmult_strassen_dryrun(A_mat, B_mat, thresh);

    const size_t max_level = tree.size();
    const long   top_level = (long)max_level - 1;
    const size_t top_sb    = (size_t)std::exp2(max_level);

    // Top-level grid extents (handles non-power-of-2 nib).
    const size_t n_si_top = integer_division_round_up(nib, top_sb);
    const size_t n_sj_top = integer_division_round_up(njb, top_sb);
    const size_t n_sk_top = integer_division_round_up(nkb, top_sb);

    // Worst-case scratch sizing for Mat tiles + pointer tables on a
    // pure-Strassen path from a single top super-block.
    const size_t h_top      = top_sb / 2;
    const size_t depth      = max_level;
    const size_t pool_blocks = compute_total_blocks(h_top, depth) + 12*h_top*h_top + 64;
    const size_t block_elems = i_bs * j_bs;

    BlockMemoryPool<Num> scratch_pool(block_elems);
    {   // warm pool so its memory is resident before recursion starts
        std::vector<Num*> ptrs(pool_blocks);
        for(auto& p : ptrs) p = scratch_pool.allocate(block_elems);
        for(auto  p : ptrs) scratch_pool.deallocate(p, block_elems);
    }
    BlockAllocator<Num> scratch_alloc(&scratch_pool);
    std::vector<const Mat<Num>*> ptr_buf(pool_blocks);
    PtrBumpAlloc<Num> ptr_alloc{ ptr_buf.data(), pool_blocks, 0 };

    // Walk the top-level super-block grid.
    for(size_t sk=0; sk<n_sk_top; ++sk)
    for(size_t sj=0; sj<n_sj_top; ++sj)
    for(size_t si=0; si<n_si_top; ++si){
        ptr_alloc.restore(0);
        hybrid_recurse(C_mat, A_mat, B_mat,
                       si*top_sb, sj*top_sb, sk*top_sb,
                       top_sb, top_level,
                       tree, thresh,
                       i_bs, j_bs,
                       scratch_alloc, ptr_alloc);
    }
}

#endif
