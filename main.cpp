#include "matrix.h"
#include "matrix.hpp"
#include "blocksparsematrix.h"
#include "blocksparsematrix.hpp"
#include "blocksparsematrix_naive.h"
#include "blocksparsematrix_naive.hpp"
#include "strassen.hpp"
#include <chrono>
#include <cstdio>

int main(int argc, char** argv){
  if(argc <5){
    puts("Usage <program> <N_mat> <blocksize> <min_level> <N_test>");
    exit(1);
  }
  const size_t N = (size_t)std::stoi(argv[1]);
  const size_t blocksize = (size_t)std::stoi(argv[2]);
  const size_t min_level = (size_t)std::stoi(argv[3]);
  const int N_test = std::stoi(argv[4]);

  Matrix<double> mat1(N,N);
  mat1.fill_with_uniform_pseudo_random_numbers(0,1,42);
  Matrix<double> mat2(N,N);
  mat2.fill_with_uniform_pseudo_random_numbers(0,1,43);
#if 0
  Matrix<double> mat1(11230,11230);
  mat1.read_from_file("S_dna16_hf_svp");
  Matrix<double> mat2(11230,11230);
  mat2.read_from_file("density_dna16_hf_svp");
#endif
#if 1 //test dense multiplication
  //Matrix<double> mat1(N,N);
  //Matrix<double> mat2(N,N);
  Matrix<double> mat3(N,N);
  for(int i=0;i<N_test;++i)
  {
    mat3.scale(0.e0);
    const std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    matmult(mat3,mat1,false,mat2,false,1.e0,1.e0);
    const std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    const double musec = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    assert(mat3.nrow() == mat1.nrow());
    assert(mat3.ncol() == mat2.ncol());
    assert(mat1.ncol() == mat2.nrow());
    const double GFLOPs = 2e-3*(double)(mat3.nrow()*mat3.ncol()*mat1.ncol())/(musec);
    printf("Matmult (dense) took %2.4f seconds (%2.4f GFLOPs)\n",1e-6*musec,GFLOPs);
  }
#endif
#if 1 //test sparse multiplication
  const size_t bs1 = blocksize;
  const size_t bs2 = blocksize;
  const size_t bs3 = blocksize;
    //const double thresh = 1e-10;
  BlockSparseMatrix<double> bsmat1(mat1,bs1,bs2,0.e0);
  BlockSparseMatrix<double> bsmat2(mat2,bs2,bs3,0.e0);
  BlockSparseMatrix<double> bsmat3(Matrix<double>(N,N),bs1,bs3,0.e0);
 const double thresh = 0.e0;
  //bsmat1.create_block_pixmap("mat1.xpm");
  //bsmat2.create_block_pixmap("mat2.xpm");
  for(int i=0;i<N_test;++i)
  {
    bsmat3.scale(0.e0);
    const std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    matmult(bsmat3,bsmat1,false,bsmat2,false,thresh,1.e0,1.e0);
    const std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    const double musec = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    const double GFLOPs = 2e-3*(double)(mat1.nrow()*mat1.ncol()*mat2.ncol())/(musec);
    printf("Matmult (BSMat) took %2.4f seconds (%2.4f GFLOPs)\n",1e-6*musec,GFLOPs);
    const auto mat3_bs = bsmat3.to_matrix();
    printf("average RMSD = %e\n",(mat3_bs - mat3).calc_frobenius_norm()/(N*N));
  }
#endif
#if 1 //test Strassen multiplication
  for(int i=0;i<N_test;++i)
  {
    bsmat3.scale(0.e0);
    const std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    //matmult_strassen(bsmat3,bsmat1,false,bsmat2,false,thresh,1.e0,1.e0,min_level);
    matmult_strassen_4_parallel(bsmat3,bsmat1,false,bsmat2,false,thresh,1.e0,1.e0,min_level);
    const std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    const double musec = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    const double GFLOPs = 2e-3*(double)(mat1.nrow()*mat1.ncol()*mat2.ncol())/(musec);
    printf("Matmult (Strassen) took %2.4f seconds (%2.4f GFLOPs)\n",1e-6*musec,GFLOPs);
    const auto mat3_bs = bsmat3.to_matrix();
    printf("average RMSD = %e\n",(mat3_bs - mat3).calc_frobenius_norm()/(N*N));
  }
#endif
}

