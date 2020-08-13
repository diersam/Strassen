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
#if 0
  DMat mat1(2,2);
  mat1.elem(0,0) = 1.e0;
  mat1.elem(0,1) = 2.e0;
  mat1.elem(1,0) = 3.e0;
  mat1.elem(1,1) = 4.e0;
  mat1.write_to_file("mat1");
  DMat mat2(2,2);
  mat2.read_from_file("mat1");
  mat1.print("mat1");
  mat2.print("mat2");
  mat2.create_pixmap("mat2.xpm");
  BlockSparseMatrix<double> bsmat2(mat1,2,2,0.e0);
  bsmat2.create_block_pixmap("bsmat2.xpm");
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

  Matrix<double> mat1(11230,11230);
  //Matrix<double> mat1(11880,11880);
  mat1.read_from_file("PS_dna16_svp");
  //mat1.create_pixmap("density_dna16_svp_cart.xpm");
  Matrix<double> mat2(mat1);
  Matrix<double> mat3(11230,11230);
  {
    const std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    matmult(mat3,mat1,false,mat2,false,1.e0,1.e0);
    const std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    const double musec = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    const double GFLOPs = 2e-3*(double)(mat3.nrow()*mat3.ncol()*mat1.ncol())/(musec);
    printf("Matmult (dense) took %2.4f seconds (%2.4f GFLOPs)\n",1e-6*musec,GFLOPs);
  }
#if 1
  {
    BlockSparseMatrix<double> bsmat1(mat1,96,96,0.e0);
    BlockSparseMatrix<double> bsmat2(mat2,96,96,0.e0);
    BlockSparseMatrix<double> bsmat3(Matrix<double>(11230,11230),96,96,0.e0);

    const std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    //matmult_strassen(bsmat3,bsmat1,false,bsmat2,false,0.e0,1.e0,1.e0);
    matmult(bsmat3,bsmat1,false,bsmat2,false,0.e0,1.e0,1.e0);
    const std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    const double musec = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    const double GFLOPs = 2e-3*(double)(mat3.nrow()*mat3.ncol()*mat1.ncol())/(musec);
    printf("Matmult (BSMat) took %2.4f seconds (%2.4f GFLOPs)\n",1e-6*musec,GFLOPs);
    //const auto mat3_bs = bsmat3.to_matrix();
    //printf("RMSD = %e\n",(mat3_bs - mat3).calc_frobenius_norm());
  }
#endif
}

