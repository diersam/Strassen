#include "matrix.h"
#include "matrix.hpp"
#include "blocksparsematrix.h"
#include "blocksparsematrix.hpp"
#include "strassen.hpp"
#include <chrono>
#include <cstdio>

size_t min_size_for_strassen;
#if 0
int main(int argc, char** argv){
  if(argc < 5){
    puts("Usage: <program> <blocksize> <strassen_min> <thresh> <n_runs>");
    return 1;
  }
  const size_t bs     = (size_t)std::stoi(argv[1]);
  min_size_for_strassen = (size_t)std::stoi(argv[2]);
  const double thresh = std::stod(argv[3]);
  const int    n_runs = std::stoi(argv[4]);

  constexpr size_t N_file = 11230;
  constexpr size_t N = N_file;

  Matrix<double> mat1(N, N), mat2(N, N);
  mat1.read_from_file("PS_dna16_hf_svp");
  mat2.read_from_file("density_dna16_hf_svp");
  Matrix<double> mat3_dense(N, N);
  printf("--- Dense DGEMM (%zu x %zu) ---\n", N, N);
  for(int i=0;i<n_runs;++i){
    const auto t0=std::chrono::steady_clock::now();
    matmult(mat3_dense,mat1,false,mat2,false,1.0,0.0);
    const auto t1=std::chrono::steady_clock::now();
    const double us=(double)std::chrono::duration_cast<std::chrono::microseconds>(t1-t0).count();
    printf("  [%d] %.4f s  (%.4f GFLOPs)\n",i+1,1e-6*us,2e-3*(double)(N*N*N)/us);
  }
  printf("\n");
  const double L2_norm_of_output = mat3_dense.calc_frobenius_norm();

  BlockSparseMatrix<double> bsmat1(mat1,bs,bs,0.0);
  BlockSparseMatrix<double> bsmat2(mat2,bs,bs,0.0);
  BlockSparseMatrix<double> bsmat3(Matrix<double>(N,N),bs,bs,0.0);

  printf("--- BSM matmult (thresh=%.2e) ---\n",thresh);
  for(int i=0;i<n_runs;++i){
    bsmat3.fill_with_values(0.0);
    const auto t0=std::chrono::steady_clock::now();
    matmult(bsmat3,bsmat1,false,bsmat2,false,thresh,1.0,0.0);
    const auto t1=std::chrono::steady_clock::now();
    const double us=(double)std::chrono::duration_cast<std::chrono::microseconds>(t1-t0).count();
    printf("  [%d] %.4f s  (%.4f GFLOPs)",i+1,1e-6*us,2e-3*(double)(N*N*N)/us);
    printf("relative RMSD = %e\n",(bsmat3.to_matrix()-mat3_dense).calc_frobenius_norm()/L2_norm_of_output);
  }
  printf("\n");

  printf("--- Strassen-Sparse (thresh=%.2e) ---\n",thresh);
  for(int i=0;i<n_runs;++i){
    bsmat3.fill_with_values(0.0);
    const auto t0=std::chrono::steady_clock::now();
    matmult_strassen_sparse(bsmat3,bsmat1,false,bsmat2,false,thresh,1.0,0.0);
    const auto t1=std::chrono::steady_clock::now();
    const double us=(double)std::chrono::duration_cast<std::chrono::microseconds>(t1-t0).count();
    printf("  [%d] %.4f s  (%.4f GFLOPs)",i+1,1e-6*us,2e-3*(double)(N*N*N)/us);
    printf("relative RMSD = %e\n",(bsmat3.to_matrix()-mat3_dense).calc_frobenius_norm()/L2_norm_of_output);
  }
  printf("\n");

  return 0;
}
#endif
