#ifndef BLOCKSPARSE3D_TENSOR_H
#define BLOCKSPARSE3D_TENSOR_H

#include <vector>
#include "blocksparsematrix.h"

template<typename Num>
using BlockSparse3DTensor = std::vector<BlockSparseMatrix<Num>>;

//computes tensor operation:
//C_{ij} = \sum_{kl} transA(A)_{ik}^l transB(B)_{kj}^l
template<typename Num>
void tensormult(BlockSparseMatrix<Num>& C,
             const BlockSparse3DTensor<Num>& A, const bool transA,
             const BlockSparse3DTensor<Num>& B, const bool transB,
             const Num thresh, const Num& alpha, const Num& beta);

//computes tensor operation of the Form:
//transC(C)_{ij}^l = \sum_k transA(A)_{ik} transB(B)_{kj}^l
template<typename Num>
void tensormult(BlockSparse3DTensor<Num>& C, const bool transC,
             const BlockSparseMatrix<Num>& A, const bool transA,
             const BlockSparse3DTensor<Num>& B, const bool transB,
             const Num thresh, const Num& alpha, const Num& beta);

//computes tensor operation of the Form:
//C_{ij}^l = \sum_k transA(A)_{ik} transB(B)_{kj}^l
template<typename Num>
void tensormult(BlockSparseMatrix<Num>& C,
             const BlockSparse3DTensor<Num>& A, const bool transA,
             const BlockSparse3DTensor<Num>& B, const bool transB,
             const Num thresh, const Num& alpha, const Num& beta){
   tensormult(C,false,A,transA,B,transb,thresh,alpha,beta);
}
//computes tensor operation of the Form:
//C_{ij}^l = transA(A)_{ik} transB(B)_{kj}^l
template<typename Num>
void tensormult(BlockSparse3DTensor<Num>& C,
             const BlockSparseMatrix<Num>& A, const bool transA,
             const BlockSparse3DTensor<Num>& B, const bool transB,
             const Num thresh, const Num& alpha, const Num& beta){
   tensormult(C,true,A,!transA,B,!transb,thresh,alpha,beta);
}

#endif

