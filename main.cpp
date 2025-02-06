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

#if 1
  Matrix<double> mat1(11230,11230);
  mat1.read_from_file("PS_dna16_hf_svp");
  Matrix<double> mat2(11230,11230);
  mat2.read_from_file("density_dna16_hf_svp");
#endif
#if 1 //test dense multiplication
  Matrix<double> mat3(11230,11230);//zero output matrix
  //for(int i=0;i<5;++i)
  {
    const std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    matmult(mat3,mat1,false,mat2,false,1.e0,0.e0);
    const std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    const double musec = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    const double GFLOPs = 2e-3*(double)(mat3.nrow()*mat3.ncol()*mat1.ncol())/(musec);
    printf("Matmult (dense) took %2.4f seconds (%2.4f GFLOPs)\n",1e-6*musec,GFLOPs);
  }
#endif
#if 1 //test sparse multiplication
  //for(int i=0;i<5;++i)
  {
    constexpr size_t bs = 48;
    BlockSparseMatrix<double> bsmat1(mat1,bs,bs,0.e0);
    BlockSparseMatrix<double> bsmat2(mat2,bs,bs,0.e0);
    BlockSparseMatrix<double> bsmat3(Matrix<double>(11230,11230),bs,bs,0.e0);
    //bsmat1.create_block_pixmap("mat1.xpm");
    //bsmat2.create_block_pixmap("mat2.xpm");
 

    const std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    const double thresh = 1e-10;
    //const double thresh = 0.e0;
    matmult(bsmat3,bsmat1,false,bsmat2,false,thresh,1.e0,1.e0);
    //matmult_strassen(bsmat3,bsmat1,false,bsmat2,false,thresh,1.e0,1.e0);
    const std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    const double musec = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    const double GFLOPs = 2e-3*(double)(mat1.nrow()*mat1.ncol()*mat2.ncol())/(musec);
    printf("Matmult (BSMat) took %2.4f seconds (%2.4f GFLOPs)\n",1e-6*musec,GFLOPs);
    //const auto mat3_bs = bsmat3.to_matrix();
    //printf("RMSD = %e\n",(mat3_bs - mat3).calc_frobenius_norm());
  }
#endif
}

