#include "matrix.h"
#include "matrix.hpp"
#include "blocksparsematrix.h"
#include "blocksparsematrix.hpp"
#include <chrono>
#include <cstdio>


int main(int argc, char** argv){
#if 0
  if (argc < 5){
    puts("Usage: <howoften> <ni> <nj> <nk>");
    die("Too few arguments");
  }
  const size_t howoften = (size_t) std::atoi(argv[1]);
  const size_t ni = (size_t) std::atoi(argv[2]);
  const size_t nj = (size_t) std::atoi(argv[3]);
  const size_t nk = (size_t) std::atoi(argv[4]);
  std::vector<double> A(nk*ni);
  std::vector<double> B(nk*nj);
  std::vector<double> C(ni*nj);
  fill_randomly<double>(A.begin(),A.end());
  fill_randomly<double>(B.begin(),B.end());
  fill_randomly<double>(C.begin(),C.end());

  {
    const std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    for (size_t i=0; i < howoften;++i){
      matmult(C.data(),ni,nj,A.data(),nk,ni,B.data(),nk,nj,12);
    }
    const std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    const double musec = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    const double average_musec = musec/((double)howoften);
    const double nflop =  2.e0*(double)(ni*nj*nk);
    const double flops = 1e6*nflop/average_musec;
    printf(" %lu multiplications of two %lux%lu matrices took %f us in total (average = %f, flops = %e)\n",
        howoften,ni,nj,musec,average_musec,flops);
  }
  DMat mat1(2,2);
  mat1.elem(0,0) = 1.e0;
  mat1.elem(0,1) = 2.e0;
  mat1.elem(1,0) = 3.e0;
  mat1.elem(1,1) = 4.e0;
  DMat mat2(2,2);
  mat2.elem(0,0) = -4.e0;
  mat2.elem(0,1) = 3.e0;
  mat2.elem(1,0) = -2.e0;
  mat2.elem(1,1) = 1.e0;
  const DMat mat3 = mat1*mat2;
  mat1.print("mat1");
  mat2.print("mat2");
  mat3.print("mat3");
  BlockSparseMatrix<double> bsmat1(mat1,1,1,0.e0);
  BlockSparseMatrix<double> bsmat2(mat2,1,1,0.e0);
  BlockSparseMatrix<double> bsmat3(2,2,1,1,0.e0);
  matmult(bsmat3,bsmat1,false,bsmat2,false,0.e0);
  bsmat1.print("bsmat1");
  bsmat2.print("bsmat2");
  bsmat3.print("bsmat3");
#endif
  Matrix<double> mat1(10000,10000);
  Matrix<double> mat2(10000,10000);
  Matrix<double> mat3(10000,10000);
  {
    const std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    matmult(mat3,mat1,false,mat2,false,0.e0);
    const std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    const double musec = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    printf("Matmult (dense) took %2.4f seconds\n",1e-6*musec);
  }
  {
    BlockSparseMatrix<double> bsmat1(mat1,96,96,0.e0);
    BlockSparseMatrix<double> bsmat2(mat2,96,96,0.e0);
    BlockSparseMatrix<double> bsmat3(10000,10000,96,96,0.e0);
    const std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    matmult(bsmat3,bsmat1,false,bsmat2,false,0.e0);
    const std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    const double musec = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    printf("Matmult (BSMat) took %2.4f seconds\n",1e-6*musec);
  }
}

