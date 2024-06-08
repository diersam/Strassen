#ifndef STRASSEN_HPP
#define STRASSEN_HPP
#include "blocksparsematrix.h"
#include "blocksparsematrix.hpp"
#include "utils.hpp"
#include <cassert>
#include <cstdio>
//#define NDEBUG

template<typename Num>
void matmult_strassen(BlockSparseMatrix<Num>& C, 
             const BlockSparseMatrix<Num>& A, const bool transA,
             const BlockSparseMatrix<Num>& B, const bool transB,
             const Num thresh, const Num& alpha, const Num& beta)
{
  //only these options so far
  assert(alpha == Num(1));
  assert(beta == Num(1));

  //no on-the-fly transformation yet
  assert(transA == false);
  assert(transB == false);

  //using Mat = Matrix<Num,BlockAllocator<Num>>;
  if (beta == Num(1)){
    //nothing to do
  }else if (beta == Num(0)){
    C.zero();
  }else{
    C *= beta;
  }

  //check dimensions
  const size_t ni = transA? A.ncol() : A.nrow();
  const size_t nj = transB? B.nrow() : B.ncol();
  const size_t nk1 = transA? A.nrow() : A.ncol();
  const size_t nk2 = transB? B.ncol() : B.nrow();

  assert(nk1 == nk2);
  const size_t nk = nk1;

  const size_t i_block_size = transA? A.max_blocksize_col() : A.max_blocksize_row();
  const size_t j_block_size = transB? B.max_blocksize_row() : B.max_blocksize_col();

  assert(i_block_size == C.max_blocksize_row());
  assert(j_block_size == C.max_blocksize_col());
  assert(ni == C.nrow());
  assert(nj == C.ncol());

  const size_t k_block_size1 = transA? A.max_blocksize_row() : A.max_blocksize_col();
  const size_t k_block_size2 = transB? B.max_blocksize_col() : B.max_blocksize_row();
  assert(k_block_size1 == k_block_size2);
  const size_t k_block_size = k_block_size1;

  const size_t nib = integer_division_round_up(ni,i_block_size);
  assert(nib == C.nrowblocks());
  assert(nib == transA? A.ncolblocks() : A.nrowblocks());
  const size_t njb = integer_division_round_up(nj,j_block_size);
  assert(njb == C.ncolblocks());
  assert(njb == transB? B.nrowblocks() : B.ncolblocks());
  const size_t nkb = integer_division_round_up(nk,k_block_size);
  assert(nkb == transA? A.nrowblocks() : A.ncolblocks());
  assert(nkb == transB? B.ncolblocks() : B.nrowblocks());

  //only quadratic so far
  assert(ni == nj);
  assert(ni == nk);
  assert(nib == njb);
  assert(nib == nkb);

  const size_t max_level = (size_t)std::log2(nib);
  std::vector<std::vector<size_t>> nflops_per_level(max_level+1);
  {//initiialize at level -1:
    nflops_per_level[0] = std::vector<size_t>(nib*njb*nkb);
    #pragma omp parallel for schedule(static)
    for(size_t ijkb=0;ijkb<nib*njb*nkb;++ijkb){
      const size_t ib = ijkb%nib;
      const size_t jb = (ijkb/nib)%njb;
      const size_t kb = ijkb/(nib*njb);
      const auto& a_block = transA? A.block(kb,ib) : A.block(ib,kb);
      const auto& b_block = transB? B.block(jb,kb) : B.block(kb,jb);
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
    const size_t n_flops_total = std::accumulate(nflops_per_level[0].cbegin(),nflops_per_level[0].cend(),0lu);
    printf("total flops on level %lu: %lu\n",0lu, n_flops_total);
  }
  const size_t max_nflops_per_block = 2lu*i_block_size*j_block_size*k_block_size;
  //recursive bottom up to decide between Strassen and sparse at each level iterating the block MM count upwards
  for(size_t level=0;level<max_level;++level){// bottom up
    const size_t previous_super_block_size = (size_t)std::exp2(level);//no of blocks in previous level(1,2,4...)
    const size_t super_block_size = (size_t)std::exp2(level+1);//no of blocks in super-block (2,4,8...)
    const size_t Strassen_thresh = (size_t)std::round(std::pow(7,level+1))*max_nflops_per_block;
    printf("level: %lu size: %lu Strassen_thresh: %lu\n",level,super_block_size,Strassen_thresh);

    const size_t n_super_i = integer_division_round_up(nib,super_block_size);
    const size_t n_super_j = integer_division_round_up(njb,super_block_size);
    const size_t n_super_k = integer_division_round_up(nkb,super_block_size);

    const size_t previous_n_super_i = integer_division_round_up(nib,previous_super_block_size);
    const size_t previous_n_super_j = integer_division_round_up(njb,previous_super_block_size);
    //const size_t previous_n_super_k = integer_division_round_up(nkb,previous_super_block_size);

    nflops_per_level[level+1] = std::vector<size_t>(n_super_i*n_super_j*n_super_k);
    #pragma omp parallel for schedule(static)
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
      const bool do_Strassen = (nflops >= Strassen_thresh) && !is_edge_case;
      if (do_Strassen){
        //upwards iteration of FLOPs-count
        nflops_per_level[level+1][ijk(super_i,super_j,super_k,n_super_i,n_super_j)] = Strassen_thresh;
      }else{//we do conventional
        //upwards iteration of FLOPs-count
        nflops_per_level[level+1][ijk(super_i,super_j,super_k,n_super_i,n_super_j)] = nflops;
      }
    }
    //number of FLops on this level (should never increase at higher levels)
    const size_t n_flops_total = std::accumulate(nflops_per_level[level+1].cbegin(),nflops_per_level[level+1].cend(),0lu);
    printf("total flops on level %lu: %lu\n",level+1, n_flops_total);
  }
}
#endif

