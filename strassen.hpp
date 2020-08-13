#ifndef STRASSEN_HPP
#define STRASSEN_HPP
#include "blocksparsematrix.h"
#include "blocksparsematrix.hpp"
#include "utils.hpp"
#include <cassert>
#include <cstdio>


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

  //only dense so far
  assert(thresh == Num(0));
  assert(C.no_of_alloc_blocks() == C.nblocks());
  assert(A.no_of_alloc_blocks() == A.nblocks());
  assert(B.no_of_alloc_blocks() == B.nblocks());

  using Mat = Matrix<Num,BlockAllocator<Num>>;
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

  const size_t max_level = std::log2(nib);
  for(size_t level=0;level<max_level;++level){
    //printf("level: %lu size: %lu\n",level,(size_t)std::exp2(level));
#if 0
    const auto I01 = A11 + A22;
    const auto I02 = B11 + B22;
    const auto M1 = I01*I02;
    const auto I03 = A21 + A22;
    const auto I04 = B11;
    const auto M2 = I03*I04;
    const auto I05 = A11;
    const auto I06 = B12 - B22;
    const auto M3 = I05*I06;
    const auto I07 = A22;
    const auto I08 = B12 - B22;
    const auto M4 = I07*I08;
    const auto I09 = A11 + A12;
    const auto I10 = B22;
    const auto M5 = I09*I10;
    const auto I11 = A21 - A11;
    const auto I12 = B11 + B12;
    const auto M6 = I11*I12;
    const auto I13 = A12 - A22;
    const auto I14 = B21 + B22;
    const auto M7 = I13*I14;

    C11 += M1 + M4 - M5 + M7;
    C12 += M3 + M5;
    C21 += M2 + M4;
    C22 += M1 - M2 + M3 + M6;
#endif
  }
}
#endif

