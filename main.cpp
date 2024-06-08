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

  //Matrix<double> mat1(2746,2746);
  //Matrix<double> mat1(11230,11230);
  //mat1.read_from_file("temp");
  //mat1.create_pixmap("temp.xpm");
  //BlockSparseMatrix<double> bsmat1(mat1,48,48,0.e0);
  //bsmat1.create_block_pixmap("temp.xpm");
  //exit(1);
#if 1
  //mat1.create_pixmap("density_dna16_svp_full.xpm");
  //mat1.read_from_file("dna4_hf_svp_density");
  //mat1.apply([](const auto& val){return std::pow(std::abs(val),1.5e0);});
  //mat1.create_pixmap("dna4_hf_svp_density.xpm");
  //mat1.read_from_file("dna4_hf_svp_overlap");
  //mat1.apply([](const auto& val){return std::pow(std::abs(val),1.5e0);});
  //mat1.create_pixmap("dna4_hf_svp_overlap.xpm");
  Matrix<double> mat1(11230,11230);
  mat1.read_from_file("PS_dna16_hf_svp");
  //BlockSparseMatrix<double> bsmat1(mat1,48,48,0.e0);
  //bsmat1.create_block_pixmap("density_dna16_hf_svp.xpm");
  Matrix<double> mat2(11230,11230);
  mat2.read_from_file("density_dna16_hf_svp");
  //BlockSparseMatrix<double> bsmat2(mat2,48,48,0.e0);
  //bsmat2.create_block_pixmap("overlap_dna16_hf_svp.xpm");
  //Matrix<double> mat3 (11230,11230);
  //mat3.read_from_file("PS_dna16_hf_svp");
  //BlockSparseMatrix<double> bsmat3(mat3,48,48,0.e0);
  //bsmat3.create_block_pixmap("PS_dna16_hf_svp.xpm");
  //matmult(mat3,mat1,false,mat2,false,1.e0,0.e0);
  //mat3.write_to_file("PS_dna16_hf_svp");
  //auto mat1_padded = create_padded_matrix(mat1,12287,12287);
  //auto mat2_padded = create_padded_matrix(mat2,12287,12287);
  //auto mat3_padded = create_padded_matrix(mat3,12288,12288);
#endif
#if 0
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
#if 1
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
    //matmult(bsmat3,bsmat1,false,bsmat2,false,thresh,1.e0,1.e0);
    matmult_strassen(bsmat3,bsmat1,false,bsmat2,false,thresh,1.e0,1.e0);
    const std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    const double musec = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    const double GFLOPs = 2e-3*(double)(mat1.nrow()*mat1.ncol()*mat2.ncol())/(musec);
    printf("Matmult (BSMat) took %2.4f seconds (%2.4f GFLOPs)\n",1e-6*musec,GFLOPs);
    //const auto mat3_bs = bsmat3.to_matrix();
    //printf("RMSD = %e\n",(mat3_bs - mat3).calc_frobenius_norm());
  }
#endif
}

extern "C"{
  int mkl_serv_intel_cpu_true() {
    return 1;
  }
}

