#include "matrix.h"
#include "matrix.hpp"
#include "blocksparsematrix.h"
#include "blocksparsematrix.hpp"
#include "strassen.hpp"
#include <chrono>
#include <cstdio>
//decompress from sig-shellpair storage to Nbf^2 matrix storage (only upper triange)
template<typename T>
void decompress_integrals(Matrix<T>& ints_3c_decompressed, const size_t P, const Matrix<T>& ints_3c_compressed, const Matrix<size_t>& v2m_keys);


#if 1
int main(int argc, char** argv){
  if(argc < 10){
    puts("Usage: <program> <N_bf> <N_aux> <N_occ> <N_vec2> <3c filename> <MO filename> <v2m_key_filename> <blocksize> <threshold>");
    return 1;
  }
  const size_t N_bf = (size_t)std::stol(argv[1]);
  const size_t N_aux = (size_t)std::stol(argv[2]);
  const size_t N_occ = (size_t)std::stol(argv[3]);
  const size_t N_vec2 = (size_t)std::stol(argv[4]);
  const std::string fn_3c = std::string(argv[5]);
  const std::string fn_ocC_mo = std::string(argv[6]);
  const std::string fn_v2m_key = std::string(argv[7]);
  const size_t bs = (size_t)std::stol(argv[8]);
  const double thresh_mult = std::stod(argv[9]);

  Matrix<double> ints_3c_compressed(N_vec2,N_aux);
  ints_3c_compressed.read_from_file(fn_3c.c_str());

  Matrix<double> K_mu_i(0.e0,N_bf,N_occ);
  Matrix<double> occ_MOs(N_bf,N_occ);
  occ_MOs.read_from_file(fn_ocC_mo.c_str());
  Matrix<size_t> v2m_keys(N_vec2,1);
  v2m_keys.read_from_file(fn_v2m_key.c_str());

  printf("--- dense occ-RIK  ---\n");
  auto start=std::chrono::steady_clock::now();
  //K_{mj} = \sum_{nlsiPQ} (mn|P) (P|Q)^{-1} (Q|ls) C_{ni} C_{li} C_{sj}
  //K_{mj} = \sum_{nlsiP} (mn|P)^' (P|ls)^' C_{ni} C_{li} C_{sj}
  #pragma omp parallel
  {
    Matrix<double> ints_3c_decompressed(0.e0,N_bf,N_bf);
    Matrix<double> mui_P(N_bf,N_occ);
    Matrix<double> ij_P(N_occ,N_occ);
    Matrix<double> ij_P_sym(N_occ,N_occ);
    Matrix<double> K_mu_i_sub(0.e0,N_bf,N_occ);
    #pragma omp for schedule(static)
    for(size_t P=0; P<N_aux;++P){
      //decompress from sig-shellpair storage to Nbf^2 matrix storage (only upper triange)
      decompress_integrals(ints_3c_decompressed,P,ints_3c_compressed,v2m_keys);

      //(mu i|P) = \sum_nu (mu nu|P) C_{nu i}
      matmult(mui_P,ints_3c_decompressed,false,occ_MOs,false,1.0,0.0);
      //(ij|P) = \sum_mu  C_{mu i} (mu j|P)
      matmult(ij_P,occ_MOs,true,mui_P,false,1.0,0.0);
      //symmetrize (ij|P) + (ji|P) to account for missing upper triangle of (mn|P)
      ij_P_sym = ij_P;
      ij_P_sym.add_transpose(ij_P);
      //K_{mu j} = \sum_{iP} (mu i|P) (ij|P)
      matmult(K_mu_i_sub,mui_P,false,ij_P_sym,false,1.0,1.0);
    }
    //collect output of all threads
    #pragma omp critical
    {
      K_mu_i += K_mu_i_sub;
    }

  }
  auto end=std::chrono::steady_clock::now();
  double us=(double)std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();
  const size_t total_flops = 2*(N_bf*N_bf*N_occ*N_aux + 2*N_bf*N_occ*N_occ*N_aux);
  printf("  %.4f s  (%.4f GFLOPs)\n",1e-6*us,2e-3*(double)(total_flops)/us);

  const double L2_norm_of_output = K_mu_i.calc_frobenius_norm();
  printf("L2 norm of output = %e\n",L2_norm_of_output);


  printf("--- sparse occ-RIK  ---\n");
  start=std::chrono::steady_clock::now();
  BlockSparseMatrix<double> K_mu_i_bsm(N_bf,N_occ,bs,bs,0.0);
  K_mu_i_bsm.fill_with_values(0.e0);
  BlockSparseMatrix<double> occ_MOs_bsm(occ_MOs,bs,bs,0.0);
  #pragma omp parallel
  {
    Matrix<double> ints_3c_decompressed(0.e0,N_bf,N_bf);
    BlockSparseMatrix<double> ints_3c_decompressed_bsm(N_bf,N_bf,bs,bs,0.e0);
    BlockSparseMatrix<double> mui_P_bsm(N_bf,N_occ,bs,bs,0.0);
    BlockSparseMatrix<double> ij_P_bsm(N_occ,N_occ,bs,bs,0.0);
    BlockSparseMatrix<double> ij_P_bsm_sym(ij_P_bsm);
    BlockSparseMatrix<double> K_mu_i_sub_bsm(N_bf,N_occ,bs,bs,0.0);
    K_mu_i_sub_bsm.fill_with_values(0.e0);
    #pragma omp for schedule(static)
    for(size_t P=0; P<N_aux;++P){
      //decompress from sig-shellpair storage to Nbf^2 matrix storage (only upper triange)
      decompress_integrals(ints_3c_decompressed,P,ints_3c_compressed,v2m_keys);
      //transform into bs format
      ints_3c_decompressed_bsm.copy_from_input_matrix(ints_3c_decompressed);

      //(mu i|P) = \sum_nu (mu nu|P) C_{nu i}
      matmult(mui_P_bsm,ints_3c_decompressed_bsm,false,occ_MOs_bsm,false,thresh_mult,1.0,0.0);
      mui_P_bsm.calc_frobenius_norms();
      //(ij|P) = \sum_mu  C_{mu i} (mu j|P)
      matmult(ij_P_bsm,occ_MOs_bsm,true,mui_P_bsm,false,thresh_mult,1.0,0.0);
      //symmetrize (ij|P) + (ji|P) to account for missing upper triangle of (mn|P)
      ij_P_bsm_sym=ij_P_bsm;
      ij_P_bsm_sym.add_transpose(ij_P_bsm);
      //K_{mu j} = \sum_{iP} (mu i|P) (ij|P)
      //ij_P_bsm.calc_frobenius_norms();
      matmult(K_mu_i_sub_bsm,mui_P_bsm,false,ij_P_bsm_sym,false,thresh_mult,1.0,1.0);
    }
    //collect output of all threads
    #pragma omp critical
    {
      K_mu_i_bsm += K_mu_i_sub_bsm;
    }

  }
  end=std::chrono::steady_clock::now();
  us=(double)std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();
  printf("  %.4f s  (%.4f GFLOPs)\n",1e-6*us,2e-3*(double)(total_flops)/us);
  printf("relative RMSD = %e\n",(K_mu_i_bsm.to_matrix()-K_mu_i).calc_frobenius_norm()/L2_norm_of_output);

}

//decompress from sig-shellpair storage to Nbf^2 matrix storage (only upper triange)
template<typename T>
void decompress_integrals(Matrix<T>& ints_3c_decompressed, const size_t P, const Matrix<T>& ints_3c_compressed, const Matrix<size_t>& v2m_keys)
{
  //make sure we have the right number of keys
  assert(ints_3c_compressed.nrow() == v2m_keys.size());
  //make sure no key is too big
  assert(*std::max_element(v2m_keys.data().cbegin(),v2m_keys.data().cend()) <= ints_3c_decompressed.size());
  T* __restrict__ decompressed_ptr = ints_3c_decompressed.data_ptr();
  const T* __restrict__ compressed_ptr = &ints_3c_compressed.elem(0,P);
  for (size_t id_compressed=0;id_compressed < v2m_keys.size();++id_compressed){
    decompressed_ptr[v2m_keys.elem(id_compressed,0)] = compressed_ptr[id_compressed];
  }
}
#endif

