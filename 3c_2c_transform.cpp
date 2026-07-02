#include "matrix.h"
#include "matrix.hpp"
#include "blocksparsematrix.h"
#include "blocksparsematrix.hpp"
#include "strassen.hpp"
#include <chrono>
#include <cstdio>

#if 0
int main(int argc, char** argv){
  if(argc < 8){
    puts("Usage: <program> <N_bf> <N_aux> <N_vec2> <3c filename> <2c filename> <blocksize> <threshold>");
    return 1;
  }
  const size_t N_bf = (size_t)std::stol(argv[1]);
  const size_t N_aux = (size_t)std::stol(argv[2]);
  const size_t N_vec2 = (size_t)std::stol(argv[3]);
  const std::string fn_3c = std::string(argv[4]);
  const std::string fn_2c = std::string(argv[5]);
  const size_t bs = (size_t)std::stol(argv[6]);
  const double thresh_mult = std::stod(argv[7]);

  Matrix<double> ints_3c(N_vec2, N_aux);
  ints_3c.read_from_file(fn_3c.c_str());
  Matrix<double> ints_2c(N_aux, N_aux);
  ints_2c.read_from_file(fn_2c.c_str());

  //(mu nu |P)^' = \sum_Q (mu nu|Q) (Q|P)^{-1/2}
  Matrix<double> ints_3c_transformed(N_vec2, N_aux);

  printf("--- dense matmult  ---\n");
  for(int i=0;i<1;++i)
  {
    const auto start=std::chrono::steady_clock::now();
    matmult(ints_3c_transformed,ints_3c,false,ints_2c,false,1.0,0.0);
    const auto end=std::chrono::steady_clock::now();
    const double us=(double)std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();
    printf("  [%d] %.4f s  (%.4f GFLOPs)\n",i+1,1e-6*us,2e-3*(double)(N_vec2*N_aux*N_aux)/us);
  }
  const double L2_norm_of_output = ints_3c_transformed.calc_frobenius_norm();
  printf("L2 norm of output = %e\n",L2_norm_of_output);


  BlockSparseMatrix<double> ints_3c_bs(ints_3c,bs,bs,0.0);
  ints_3c = Matrix<double>();
  BlockSparseMatrix<double> ints_2c_bs(ints_2c,bs,bs,0.0);
  ints_2c = Matrix<double>();
  BlockSparseMatrix<double> ints_3c_transformed_bs(N_vec2, N_aux,bs,bs,0.0);

  printf("\n--- BSM matmult (thresh=%.2e) ---\n",thresh_mult);

  for(int i=0;i<1;++i)
  {
    ints_3c_transformed_bs.fill_with_values(0.e0);
    const auto start=std::chrono::steady_clock::now();
    matmult(ints_3c_transformed_bs,ints_3c_bs,false,ints_2c_bs,false,thresh_mult,1.0,1.0);
    const auto end=std::chrono::steady_clock::now();
    const double us=(double)std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();
    printf("  [%d] %.4f s  (%.4f GFLOPs)",i+1,1e-6*us,2e-3*(double)(N_vec2*N_aux*N_aux)/us);
    printf("relative RMSD = %e\n",(ints_3c_transformed_bs.to_matrix()-ints_3c_transformed).calc_frobenius_norm()/L2_norm_of_output);
  }

  return 0;
}
#endif

