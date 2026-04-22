#ifndef STRASSEN_HPP
#define STRASSEN_HPP
#include "blocksparsematrix.h"
#include "blocksparsematrix.hpp"
#include "utils.hpp"
#include <cassert>
#include <cstdio>
#include <vector>
#include <cmath>
//#define NDEBUG

template<typename Num>
using Mat = Matrix<Num,BlockAllocator<Num>>;

// MatView / ConstMatView — zero-copy submatrix views for BLAS calls
template<typename Num>
struct MatView {
    Num*   data;
    size_t nrow;
    size_t ncol;
    size_t ld;
    Num&       elem(size_t r, size_t c)       { return data[r + c*ld]; }
    const Num& elem(size_t r, size_t c) const { return data[r + c*ld]; }
};

template<typename Num>
struct ConstMatView {
    const Num* data;
    size_t     nrow;
    size_t     ncol;
    size_t     ld;
    const Num& elem(size_t r, size_t c) const { return data[r + c*ld]; }
};

template<typename Num>
ConstMatView<Num> as_view(const Mat<Num>& m) {
    return { m.data_ptr(), m.nrow(), m.ncol(), m.nrow() };
}
template<typename Num>
MatView<Num> as_view(Mat<Num>& m) {
    return { m.data_ptr(), m.nrow(), m.ncol(), m.nrow() };
}

// matmult overload for views — passes ld directly to BLAS
template<typename Num>
void matmult(MatView<Num>& C, const ConstMatView<Num>& A, const ConstMatView<Num>& B,
             const Num alpha = Num(1), const Num beta = Num(0))
{
    const char tA = 'n', tB = 'n';
    const int m   = (int)C.nrow;
    const int n   = (int)C.ncol;
    const int k   = (int)A.ncol;
    const int lda = (int)A.ld;
    const int ldb = (int)B.ld;
    const int ldc = (int)C.ld;
    any_gemm(&tA, &tB, &m, &n, &k, &alpha, A.data, &lda, B.data, &ldb, &beta, C.data, &ldc);
}

// ConstBlockGrid — non-owning view into a 2D array of Mat pointers.
// Stored column-major: block(rb,cb) = *blocks[rb + cb*stride]
// stride = leading dimension of the PARENT pointer array.
// Keeping parent stride means quadrant views work with just pointer offset:
//   quadrant(rb0,cb0,h_r,h_c) = { blocks + rb0 + cb0*stride, h_r, h_c, stride }
// No copying, no new allocation — same pointer array, different window.
template<typename Num>
struct ConstBlockGrid {
    const Mat<Num>* const* blocks;
    size_t nrb;
    size_t ncb;
    size_t stride;  // column stride of parent array (NOT necessarily == nrb)

    const Mat<Num>& block(size_t rb, size_t cb) const {
        return *blocks[rb + cb*stride];
    }

    // Zero-copy quadrant view — just offset the pointer
    ConstBlockGrid quadrant(size_t rb0, size_t cb0, size_t h_r, size_t h_c) const {
        return { blocks + rb0 + cb0*stride, h_r, h_c, stride };
    }
};

// PtrBumpAlloc — bump allocator for const Mat<Num>* pointer arrays.
// Replaces std::vector<const Mat<Num>*> in OwnedBlockGrid so pointer arrays
// come from a pre-allocated buffer instead of individual malloc calls.
template<typename Num>
struct PtrBumpAlloc {
    const Mat<Num>** buf;
    size_t           capacity;  // total pointers available
    size_t           offset;    // next free slot

    const Mat<Num>** alloc(size_t n) {
        assert(offset + n <= capacity);
        const Mat<Num>** p = buf + offset;
        offset += n;
        return p;
    }

    size_t save()           const { return offset; }
    void   restore(size_t s)      { offset = s; }
};

// OwnedBlockGrid — owns its pointer array, used for temporaries.
// Provides same interface as ConstBlockGrid.
template<typename Num>
struct OwnedBlockGrid {
    std::vector<Mat<Num>>  data;    // owns the Mat objects
    const Mat<Num>**       ptrs;    // pointer array — from PtrBumpAlloc, NOT owned
    size_t nrb;
    size_t ncb;

    OwnedBlockGrid() = default;

    OwnedBlockGrid(size_t nrb_, size_t ncb_,
                   size_t block_nr, size_t block_nc,
                   const BlockAllocator<Num>& alloc,
                   PtrBumpAlloc<Num>& ptr_alloc)
      : nrb(nrb_), ncb(ncb_)
    {
        data.reserve(nrb*ncb);
        for(size_t i=0;i<nrb*ncb;++i)
            data.emplace_back(Num(0), block_nr, block_nc, alloc);
        ptrs = ptr_alloc.alloc(nrb*ncb);
        for(size_t i=0;i<nrb*ncb;++i)
            ptrs[i] = &data[i];
    }

    Mat<Num>& block(size_t rb, size_t cb)             { return data[rb + cb*nrb]; }
    const Mat<Num>& block(size_t rb, size_t cb) const { return data[rb + cb*nrb]; }

    // View as ConstBlockGrid (non-owning)
    ConstBlockGrid<Num> view() const {
        return { ptrs, nrb, ncb, nrb };
    }

    // Zero-copy quadrant view into this grid
    ConstBlockGrid<Num> quadrant(size_t rb0, size_t cb0, size_t h_r, size_t h_c) const {
        return { ptrs + rb0 + cb0*nrb, h_r, h_c, nrb };
    }
};

// Sub-grid index helper
inline size_t sg_idx(size_t rb, size_t cb, size_t nrb){ return rb + cb*nrb; }

// subgrid_sub: R = A - B
template<typename Num, typename GridA, typename GridB>
void subgrid_sub(OwnedBlockGrid<Num>& R,
                 const GridA& A,
                 const GridB& B)
{
    for(size_t cb=0;cb<R.ncb;++cb)
        for(size_t rb=0;rb<R.nrb;++rb){
            R.block(rb,cb) = A.block(rb,cb);
            R.block(rb,cb) -= B.block(rb,cb);
        }
}

// subgrid_add: R = A + B
template<typename Num, typename GridA, typename GridB>
void subgrid_add(OwnedBlockGrid<Num>& R,
                 const GridA& A,
                 const GridB& B)
{
    for(size_t cb=0;cb<R.ncb;++cb)
        for(size_t rb=0;rb<R.nrb;++rb){
            R.block(rb,cb) = A.block(rb,cb);
            R.block(rb,cb) += B.block(rb,cb);
        }
}

// BLAS leaf: C += A * B  (triple loop, calls dgemm on each block pair)
template<typename Num, typename GridA, typename GridB>
void subgrid_matmult_acc(OwnedBlockGrid<Num>& C,
                          const GridA& A,
                          const GridB& B)
{
    for(size_t cb=0;cb<C.ncb;++cb)
        for(size_t rb=0;rb<C.nrb;++rb){
            auto cv = as_view(C.block(rb,cb));
            for(size_t kb=0;kb<A.ncb;++kb){
                const auto av = as_view(A.block(rb,kb));
                const auto bv = as_view(B.block(kb,cb));
                matmult(cv, av, bv, Num(1), Num(1));
            }
        }
}

// Accumulate OwnedBlockGrid back into BSM
template<typename Num>
void accumulate_into_bsm(BlockSparseMatrix<Num>& M,
                          const OwnedBlockGrid<Num>& G,
                          size_t ib0, size_t jb0)
{
    for(size_t cb=0;cb<G.ncb;++cb)
        for(size_t rb=0;rb<G.nrb;++rb)
            M.block(ib0+rb, jb0+cb) += G.block(rb,cb);
}

// Build a ConstBlockGrid view into a BSM quadrant — no data copy.
template<typename Num>
struct BSMQuadrant {
    std::vector<const Mat<Num>*> ptrs;
    ConstBlockGrid<Num> grid;
};

template<typename Num>
BSMQuadrant<Num> make_bsm_quadrant(const BlockSparseMatrix<Num>& M,
                                    size_t ib0, size_t jb0,
                                    size_t nrb, size_t ncb)
{
    BSMQuadrant<Num> q;
    q.ptrs.resize(nrb*ncb);
    for(size_t cb=0;cb<ncb;++cb)
        for(size_t rb=0;rb<nrb;++rb)
            q.ptrs[rb + cb*nrb] = &M.block(ib0+rb, jb0+cb);
    q.grid = { q.ptrs.data(), nrb, ncb, nrb };
    return q;
}

// compute_total_blocks — total Mat blocks needed across all recursion levels.
// Per winograd call at depth l: 9 grids of (h/2^l)^2 blocks, with 7^l calls.
inline size_t compute_total_blocks(size_t h, size_t depth)
{
    size_t total   = 0;
    size_t grid_h  = h;
    size_t n_calls = 1;
    for(size_t l = 0; l < depth; ++l){
        total  += (size_t)13 * n_calls * grid_h * grid_h;
        grid_h  /= 2;
        n_calls *= 7;
    }
    return total;
}

// compute_total_ptr_slots — total pointer slots needed for all OwnedBlockGrid ptrs arrays.
// Same count as blocks since each grid of g*g blocks needs g*g pointer slots.
inline size_t compute_total_ptr_slots(size_t h, size_t depth)
{
    return compute_total_blocks(h, depth);
}

// Forward declaration
template<typename Num, typename GridA, typename GridB>
void winograd_on_subgrids(
    OwnedBlockGrid<Num>& C00, OwnedBlockGrid<Num>& C01,
    OwnedBlockGrid<Num>& C10, OwnedBlockGrid<Num>& C11,
    const GridA& a, const GridA& b,
    const GridA& c, const GridA& d,
    const GridB& A, const GridB& C,
    const GridB& B, const GridB& D,
    size_t block_nr, size_t block_nc,
    const BlockAllocator<Num>& scratch_alloc,
    PtrBumpAlloc<Num>& ptr_alloc,
    size_t min_level, size_t current_level);

// recursive_matmult_acc: C += A * B
template<typename Num, typename GridA, typename GridB>
void recursive_matmult_acc(OwnedBlockGrid<Num>& C,
                            const GridA& A,
                            const GridB& B,
                            size_t block_nr, size_t block_nc,
                            const BlockAllocator<Num>& scratch_alloc,
                            PtrBumpAlloc<Num>& ptr_alloc,
                            size_t min_level, size_t current_level)
{
    const size_t nrb = C.nrb;
    const size_t nkb = A.ncb;
    const size_t ncb = C.ncb;

    if(current_level == min_level || nrb % 2 != 0 || nkb % 2 != 0 || ncb % 2 != 0){
        subgrid_matmult_acc(C, A, B);
        return;
    }

    const size_t h_r = nrb / 2;
    const size_t h_k = nkb / 2;
    const size_t h_c = ncb / 2;

    const auto a_q = A.quadrant(0,   0,   h_r, h_k);
    const auto b_q = A.quadrant(0,   h_k, h_r, h_k);
    const auto c_q = A.quadrant(h_r, 0,   h_r, h_k);
    const auto d_q = A.quadrant(h_r, h_k, h_r, h_k);

    const auto A_q = B.quadrant(0,   0,   h_k, h_c);
    const auto C_q = B.quadrant(0,   h_c, h_k, h_c);
    const auto B_q = B.quadrant(h_k, 0,   h_k, h_c);
    const auto D_q = B.quadrant(h_k, h_c, h_k, h_c);

    const size_t ptr_mark = ptr_alloc.save();
    OwnedBlockGrid<Num> C00(h_r, h_c, block_nr, block_nc, scratch_alloc, ptr_alloc);
    OwnedBlockGrid<Num> C01(h_r, h_c, block_nr, block_nc, scratch_alloc, ptr_alloc);
    OwnedBlockGrid<Num> C10(h_r, h_c, block_nr, block_nc, scratch_alloc, ptr_alloc);
    OwnedBlockGrid<Num> C11(h_r, h_c, block_nr, block_nc, scratch_alloc, ptr_alloc);

    winograd_on_subgrids(
        C00, C01, C10, C11,
        a_q, b_q, c_q, d_q,
        A_q, C_q, B_q, D_q,
        block_nr, block_nc, scratch_alloc, ptr_alloc,
        min_level, current_level - 1);

    for(size_t cb=0;cb<h_c;++cb)
        for(size_t rb=0;rb<h_r;++rb){
            C.block(rb,    cb    ) += C00.block(rb,cb);
            C.block(rb,    cb+h_c) += C01.block(rb,cb);
            C.block(rb+h_r,cb    ) += C10.block(rb,cb);
            C.block(rb+h_r,cb+h_c) += C11.block(rb,cb);
        }
    // C00..C11 destructors return Mat blocks to scratch pool;
    // ptr slots reclaimed via bump restore
    ptr_alloc.restore(ptr_mark);
}

// Winograd kernel
//
// A quadrants: a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1]
// B quadrants: A=B[0,0], C=B[0,1], B=B[1,0], D=B[1,1]
// Output C00..C11 pre-zeroed by caller.
template<typename Num, typename GridA, typename GridB>
void winograd_on_subgrids(
    OwnedBlockGrid<Num>& C00, OwnedBlockGrid<Num>& C01,
    OwnedBlockGrid<Num>& C10, OwnedBlockGrid<Num>& C11,
    const GridA& a, const GridA& b,
    const GridA& c, const GridA& d,
    const GridB& A, const GridB& C,
    const GridB& B, const GridB& D,
    size_t block_nr, size_t block_nc,
    const BlockAllocator<Num>& scratch_alloc,
    PtrBumpAlloc<Num>& ptr_alloc,
    size_t min_level, size_t current_level)
{
    const size_t nrb = C00.nrb;
    const size_t nkb = a.ncb;
    const size_t ncb = C00.ncb;

    const size_t ptr_mark = ptr_alloc.save();

    auto make_A = [&](){ return OwnedBlockGrid<Num>(nrb, nkb, block_nr, block_nc, scratch_alloc, ptr_alloc); };
    auto make_B = [&](){ return OwnedBlockGrid<Num>(nkb, ncb, block_nr, block_nc, scratch_alloc, ptr_alloc); };
    auto make_C = [&](){ return OwnedBlockGrid<Num>(nrb, ncb, block_nr, block_nc, scratch_alloc, ptr_alloc); };

    // A-side combinations
    auto na_pc       = make_A(); subgrid_sub(na_pc,       c, a);
    auto pc_pd       = make_A(); subgrid_add(pc_pd,       c, d);
    auto na_pc_pd    = make_A(); subgrid_sub(na_pc_pd,    pc_pd, a);
    auto pa_pb_nc_nd = make_A(); subgrid_sub(pa_pb_nc_nd, b, na_pc_pd);

    // B-side combinations
    auto pC_nD       = make_B(); subgrid_sub(pC_nD,       C, D);
    auto nA_pC       = make_B(); subgrid_sub(nA_pC,       C, A);
    auto pA_nC_pD    = make_B(); subgrid_sub(pA_nC_pD,    D, nA_pC);
    auto nA_pB_pC_nD = make_B(); subgrid_sub(nA_pB_pC_nD, B, pA_nC_pD);

    auto PROD = [&](OwnedBlockGrid<Num>& Cout,
                    const OwnedBlockGrid<Num>& Ain,
                    const OwnedBlockGrid<Num>& Bin){
        recursive_matmult_acc(Cout, Ain.view(), Bin.view(),
                              block_nr, block_nc, scratch_alloc, ptr_alloc, min_level, current_level);
    };
    auto PROD_G = [&](OwnedBlockGrid<Num>& Cout,
                      const ConstBlockGrid<Num>& Ain,
                      const ConstBlockGrid<Num>& Bin){
        recursive_matmult_acc(Cout, Ain, Bin,
                              block_nr, block_nc, scratch_alloc, ptr_alloc, min_level, current_level);
 };

    // 1. t = a*A → C00
    PROD_G(C00, a, A);

    // 2. nw = t  (separate copy)
    auto nw = make_C();
    for(size_t cb=0;cb<ncb;++cb)
        for(size_t rb=0;rb<nrb;++rb)
            nw.block(rb,cb) = C00.block(rb,cb);

    // 3. u = na_pc * pC_nD → C10
    PROD(C10, na_pc, pC_nD);

    // 4. v = pc_pd * nA_pC → C01
    PROD(C01, pc_pd, nA_pC);

    // 5. nw += na_pc_pd * pA_nC_pD
    PROD(nw, na_pc_pd, pA_nC_pD);

    // 5b–7. negate nw, update C01/C10/C11
    for(size_t cb=0;cb<ncb;++cb)
        for(size_t rb=0;rb<nrb;++rb){
            nw.block(rb,cb)  *= Num(-1);
            C01.block(rb,cb) -= nw.block(rb,cb);
            C10.block(rb,cb) -= nw.block(rb,cb);
            C11.block(rb,cb)  = nw.block(rb,cb);
            C11.block(rb,cb) += C10.block(rb,cb);
            C11.block(rb,cb) += C01.block(rb,cb);
        }

    // 8. C01 += pa_pb_nc_nd * D
    PROD_G(C01, pa_pb_nc_nd.view(), D);

    // 9. C10 += d * nA_pB_pC_nD
    PROD_G(C10, d, nA_pB_pC_nD.view());

    // 10. C00 += b * B
    PROD_G(C00, b, B);

    // Temporaries destroyed here — Mat blocks returned to scratch pool,
    // ptr slots reclaimed via bump restore
    ptr_alloc.restore(ptr_mark);
}

// min_level: 1=l1 only, 2=l1+l2, 3=l1+l2+l3
template<typename Num>
void matmult_strassen(BlockSparseMatrix<Num>& C_mat,
             const BlockSparseMatrix<Num>& A_mat, const bool transA,
             const BlockSparseMatrix<Num>& B_mat, const bool transB,
             const Num thresh, const Num& alpha, const Num& beta,
             const size_t min_level = 1)
{
    assert(transA == false);
    assert(transB == false);
    assert(alpha  == Num(1));

    if      (beta == Num(0)) C_mat.fill_with_values(Num(0));
    else if (beta != Num(1)) C_mat *= beta;

    const size_t nib = A_mat.nrowblocks();
    const size_t njb = B_mat.ncolblocks();
    const size_t nkb = A_mat.ncolblocks();
    assert(nkb == B_mat.nrowblocks());
    assert(nib == C_mat.nrowblocks());
    assert(njb == C_mat.ncolblocks());
    assert(nib == njb && nib == nkb);
    assert(nib % 2 == 0);

    const size_t i_bs = A_mat.max_blocksize_row();
    const size_t j_bs = B_mat.max_blocksize_col();
    assert(i_bs == C_mat.max_blocksize_row());
    assert(j_bs == C_mat.max_blocksize_col());

    const size_t h         = nib / 2;
    const size_t max_level = (size_t)std::log2((double)C_mat.nrowblocks()+0.5);
    const size_t depth     = max_level - min_level;
    //printf("max_level = %i\n",(int)max_level);

    // Build a scratch pool sized exactly for all temporaries.
    // Pre-warm it so no allocation happens during the recursive calls.
    const size_t block_elems  = i_bs * j_bs;
    const size_t total_blocks = (size_t)14 * h * h;
    BlockMemoryPool<Num> scratch_pool(block_elems);
    {
        std::vector<Num*> ptrs(total_blocks);
        for(auto& p : ptrs) p = scratch_pool.allocate(block_elems);
        for(auto  p : ptrs) scratch_pool.deallocate(p, block_elems);
    }
    BlockAllocator<Num> scratch_alloc(&scratch_pool);

    // Pointer bump allocator — eliminates all malloc calls for ptrs arrays.
    const size_t total_ptr_slots = (size_t)72 * h * h;
    std::vector<const Mat<Num>*> ptr_buf(total_ptr_slots);
    PtrBumpAlloc<Num> ptr_alloc{ ptr_buf.data(), total_ptr_slots, 0 };

    // Allocate zeroed output quadrants from scratch pool
    OwnedBlockGrid<Num> C00(h, h, i_bs, j_bs, scratch_alloc, ptr_alloc);
    OwnedBlockGrid<Num> C01(h, h, i_bs, j_bs, scratch_alloc, ptr_alloc);
    OwnedBlockGrid<Num> C10(h, h, i_bs, j_bs, scratch_alloc, ptr_alloc);
    OwnedBlockGrid<Num> C11(h, h, i_bs, j_bs, scratch_alloc, ptr_alloc);

    // Non-owning views into A and B quadrants — no data copy
    auto qa = make_bsm_quadrant(A_mat, 0, 0, h, h);
    auto qb = make_bsm_quadrant(A_mat, 0, h, h, h);
    auto qc = make_bsm_quadrant(A_mat, h, 0, h, h);
    auto qd = make_bsm_quadrant(A_mat, h, h, h, h);

    auto qA = make_bsm_quadrant(B_mat, 0, 0, h, h);
    auto qC = make_bsm_quadrant(B_mat, 0, h, h, h);
    auto qB = make_bsm_quadrant(B_mat, h, 0, h, h);
    auto qD = make_bsm_quadrant(B_mat, h, h, h, h);

    winograd_on_subgrids(
        C00, C01, C10, C11,
        qa.grid, qb.grid, qc.grid, qd.grid,
        qA.grid, qC.grid, qB.grid, qD.grid,
        i_bs, j_bs, scratch_alloc, ptr_alloc,
        min_level, max_level);

    // Accumulate results back into C_mat
    accumulate_into_bsm(C_mat, C00, 0, 0);
    accumulate_into_bsm(C_mat, C01, 0, h);
    accumulate_into_bsm(C_mat, C10, h, 0);
    accumulate_into_bsm(C_mat, C11, h, h);

    scratch_pool.release_all_memory();
}


// matmult_strassen
// min_level: 1=l1 only, 2=l1+l2, 3=l1+l2+l3
template<typename Num>
void matmult_strassen_4_parallel(BlockSparseMatrix<Num>& C_mat,
             BlockSparseMatrix<Num>& A_mat, const bool transA,
             BlockSparseMatrix<Num>& B_mat, const bool transB,
             const Num thresh, const Num& alpha, const Num& beta,
             const size_t min_level = 1)
{
  const size_t nisb = 2;
  const size_t njsb = 2;
  const size_t nksb = 2;
  std::vector<BlockSparseMatrix<Num>> A_mat_subs(nisb*nksb);
  std::vector<BlockSparseMatrix<Num>> B_mat_subs(nksb*njsb);

  #pragma omp parallel for schedule(static) collapse(2)
  for(size_t isb=0;isb<nisb;++isb){
    for(size_t ksb=0;ksb<nksb;++ksb){
      BlockSparseMatrix<Num>& A_mat_sub = A_mat_subs[ij(isb,ksb,nisb)];
      const size_t ni_sub = C_mat.nrow()/2;
      const size_t nk_sub = A_mat.ncol()/2;

      //generate sub-A block (cheap, just std::move blocks)
      A_mat_sub = BlockSparseMatrix<Num>(ni_sub,nk_sub,A_mat.max_blocksize_row(),A_mat.max_blocksize_col(),A_mat.thresh(),A_mat.mem_pool_ptr());
      const size_t nib_sub = A_mat_sub.nrowblocks();
      const size_t nkb_sub = A_mat_sub.ncolblocks();
      const size_t ib_start = isb*nib_sub;
      const size_t kb_start = ksb*nkb_sub;
      for(size_t kb_sub=0;kb_sub<nkb_sub;++kb_sub){
        const size_t kb = kb_start + kb_sub;
        for(size_t ib_sub=0;ib_sub<nib_sub;++ib_sub){
          const size_t ib = ib_start + ib_sub;
          A_mat_sub.block(ib_sub,kb_sub) = std::move(A_mat.block(ib,kb));
        }
      }
    }
  }

  #pragma omp parallel for schedule(static) collapse(2)
  for(size_t ksb=0;ksb<nksb;++ksb){
    for(size_t jsb=0;jsb<njsb;++jsb){
      BlockSparseMatrix<Num>& B_mat_sub = B_mat_subs[ij(ksb,jsb,nisb)];
      const size_t nj_sub = C_mat.ncol()/2;
      const size_t nk_sub = A_mat.ncol()/2;
      //generate sub-B block (cheap, just std::move blocks)
      B_mat_sub = BlockSparseMatrix<Num>(nk_sub,nj_sub,B_mat.max_blocksize_row(),B_mat.max_blocksize_col(),B_mat.thresh(),B_mat.mem_pool_ptr());
      const size_t nkb_sub = B_mat_sub.nrowblocks();
      const size_t njb_sub = B_mat_sub.ncolblocks();
      const size_t kb_start = ksb*nkb_sub;
      const size_t jb_start = jsb*njb_sub;
      for(size_t jb_sub=0;jb_sub<njb_sub;++jb_sub){
        const size_t jb = jb_start + jb_sub;
        for(size_t kb_sub=0;kb_sub<nkb_sub;++kb_sub){
          const size_t kb = kb_start + kb_sub;
          B_mat_sub.block(kb_sub,jb_sub) = std::move(B_mat.block(kb,jb));
        }
      }
    }
  }

  //parallel loop over 2x2 output matrix blocks
  #pragma omp parallel for schedule(static) collapse(2)
  for(size_t jsb=0;jsb<njsb;++jsb){
    for(size_t isb=0;isb<nisb;++isb){
      const size_t nj_sub = C_mat.ncol()/2;
      const size_t ni_sub = C_mat.nrow()/2;

      //generate sub-C block (cheap, just std::move blocks)
      BlockSparseMatrix<Num> C_mat_sub(ni_sub,nj_sub,C_mat.max_blocksize_row(),C_mat.max_blocksize_col(),C_mat.thresh(),C_mat.mem_pool_ptr());
      const size_t nib_sub = C_mat_sub.nrowblocks();
      const size_t njb_sub = C_mat_sub.ncolblocks();
      const size_t ib_start = isb*nib_sub;
      const size_t jb_start = jsb*njb_sub;
      for(size_t jb_sub=0;jb_sub<njb_sub;++jb_sub){
        const size_t jb = jb_start + jb_sub;
        for(size_t ib_sub=0;ib_sub<nib_sub;++ib_sub){
          const size_t ib = ib_start + ib_sub;
          C_mat_sub.block(ib_sub,jb_sub) = std::move(C_mat.block(ib,jb));
        }
      }
      //sequential over 2x accumulation index
      for(size_t ksb=0;ksb<nksb;++ksb){
        const size_t nk_sub = A_mat.ncol()/2;

        BlockSparseMatrix<Num>& A_mat_sub = A_mat_subs[ij(isb,ksb,nisb)];
        BlockSparseMatrix<Num>& B_mat_sub = B_mat_subs[ij(ksb,jsb,nisb)];

        //call Strassen for sub-blocks
        matmult_strassen(C_mat_sub, A_mat_sub, transA, B_mat_sub, transB, thresh,alpha,beta,min_level);

      }//end ksb loop
      //move C_sub back
      for(size_t jb_sub=0;jb_sub<njb_sub;++jb_sub){
        const size_t jb = jb_start + jb_sub;
        for(size_t ib_sub=0;ib_sub<nib_sub;++ib_sub){
          const size_t ib = ib_start + ib_sub;
          C_mat.block(ib,jb) = std::move(C_mat_sub.block(ib_sub,jb_sub));
        }
      }
    }//end isb loop
  }//end jsb loop

  //move A_sub back
  #pragma omp parallel for schedule(static) collapse(2)
  for(size_t isb=0;isb<nisb;++isb){
    for(size_t ksb=0;ksb<nksb;++ksb){
      BlockSparseMatrix<Num>& A_mat_sub = A_mat_subs[ij(isb,ksb,nisb)];
      const size_t ni_sub = C_mat.nrow()/2;
      const size_t nk_sub = A_mat.ncol()/2;
      const size_t nib_sub = A_mat_sub.nrowblocks();
      const size_t nkb_sub = A_mat_sub.ncolblocks();
      const size_t ib_start = isb*nib_sub;
      const size_t kb_start = ksb*nkb_sub;
      for(size_t kb_sub=0;kb_sub<nkb_sub;++kb_sub){
        const size_t kb = kb_start + kb_sub;
        for(size_t ib_sub=0;ib_sub<nib_sub;++ib_sub){
          const size_t ib = ib_start + ib_sub;
          A_mat.block(ib,kb) = std::move(A_mat_sub.block(ib_sub,kb_sub));
        }
      }
    }
  }

  #pragma omp parallel for schedule(static) collapse(2)
  for(size_t ksb=0;ksb<nksb;++ksb){
    for(size_t jsb=0;jsb<njsb;++jsb){
      BlockSparseMatrix<Num>& B_mat_sub = B_mat_subs[ij(ksb,jsb,nisb)];
      const size_t nj_sub = C_mat.ncol()/2;
      const size_t nk_sub = A_mat.ncol()/2;

      const size_t nkb_sub = B_mat_sub.nrowblocks();
      const size_t njb_sub = B_mat_sub.ncolblocks();
      const size_t kb_start = ksb*nkb_sub;
      const size_t jb_start = jsb*njb_sub;
      //move B_sub back
      for(size_t jb_sub=0;jb_sub<njb_sub;++jb_sub){
        const size_t jb = jb_start + jb_sub;
        for(size_t kb_sub=0;kb_sub<nkb_sub;++kb_sub){
          const size_t kb = kb_start + kb_sub;
          B_mat.block(kb,jb) = std::move(B_mat_sub.block(kb_sub,jb_sub));
        }
      }
    }
  }
}

#endif
