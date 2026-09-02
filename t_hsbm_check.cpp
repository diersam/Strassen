#include "matrix.h"
#include "matrix.hpp"
#include "blocksparsematrix.h"
#include "blocksparsematrix.hpp"
#include "strassen.hpp"
#include "hsbm_driver.hpp"
#include <cstdio>
#include <random>

size_t min_size_for_strassen;

static void fill_decay(Matrix<double>& M, double lambda, unsigned seed){
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> u(-1.0,1.0);
  const double nr=(double)M.nrow(), nc=(double)M.ncol();
  for(size_t j=0;j<M.ncol();++j)
    for(size_t i=0;i<M.nrow();++i){
      const double d = std::fabs((double)i/nr - (double)j/nc);
      M.elem(i,j) = u(rng)*std::exp(-d/lambda);
    }
  M.calc_frobenius_norm();
}

static double reldiff(const Matrix<double>& a, const Matrix<double>& b){
  double n=0,d=0;
  for(size_t i=0;i<a.size();++i){ const double x=a.data_ptr()[i]-b.data_ptr()[i]; n+=x*x; d+=b.data_ptr()[i]*b.data_ptr()[i]; }
  return std::sqrt(n/d);
}

static int fails = 0;

static void one(const char* tag, size_t M, size_t K, size_t N, size_t bs,
                double lambda, double tau, double thresh, size_t S)
{
  Matrix<double> A(M,K), B(K,N), Cref(0.0,M,N);
  fill_decay(A,lambda,11); fill_decay(B,lambda,22);
  matmult(Cref,A,false,B,false,1.0,0.0);

  BlockSparseMatrix<double> Ab(A,bs,bs,tau), Bb(B,bs,bs,tau);
  Ab.calc_frobenius_norms(); Bb.calc_frobenius_norms();

  BlockSparseMatrix<double> Cb(M,N,bs,bs,tau);
  matmult(Cb,Ab,false,Bb,false,thresh,1.0,0.0);
  const double e_bsm = reldiff(Cb.to_matrix(),Cref);

  BlockSparseMatrix<double> Ch(M,N,bs,bs,tau);
  Ch.zero();
  HSBMParams<double> par;
  par.thresh = thresh;
  par.super_tile_blocks = S;
  par.warm_pools = false;
  const auto st = matmult_hsbm(Ch,Ab,Bb,par);
  const double e_hsbm = reldiff(Ch.to_matrix(),Cref);

  // With screening on, both methods legitimately carry O(theta) error, and HSBM
  // is often *more* accurate because no screening happens on the Strassen path.
  // So the criterion is relative to BSM, with an absolute floor for theta=0.
  const bool ok = (e_hsbm <= std::max(e_bsm*10.0, 1e-12));
  if(!ok) ++fails;
  printf("%-26s %5zux%-5zux%-5zu bs=%-3zu th=%-7.0e S=%-4zu L=%zu cov=%.3f tiles=%-4zu "
         "| str %zu/%zu | BSM %.2e HSBM %.2e %s\n",
         tag,M,K,N,bs,thresh,st.super_tile,st.levels,st.coverage,st.n_tiles,
         st.strassen_nodes,st.tree_nodes,
         e_bsm,e_hsbm, ok?"ok":"  <<< FAIL");
}

int main(){
  printf("== square, perfectly sized ==\n");
  one("square exact",        128,128,128, 8, 0.30, 0.0, 0.0,  0);
  one("square exact S=4",    128,128,128, 8, 0.30, 0.0, 0.0,  4);
  one("square exact S=8",    128,128,128, 8, 0.30, 0.0, 0.0,  8);
  one("square exact S=16",   128,128,128, 8, 0.30, 0.0, 0.0, 16);

  printf("\n== square, ragged block grid ==\n");
  one("square ragged 100b",  100* 6,100*6,100*6, 6, 0.30, 0.0, 0.0, 0);
  one("square odd dim",      130,130,130, 8, 0.30, 0.0, 0.0, 0);
  one("square partial block",125,125,125, 8, 0.30, 0.0, 0.0, 0);

  printf("\n== rectangular (RI-K shapes) ==\n");
  one("transform-like",      512, 128, 128, 8, 0.30, 0.0, 0.0, 0);
  one("transform ragged",    500, 130, 130, 8, 0.30, 0.0, 0.0, 0);
  one("occ-RI-K first",      256, 256,  64, 8, 0.30, 0.0, 0.0, 0);
  one("occ-RI-K skinny",     256, 256,  32, 8, 0.30, 0.0, 0.0, 0);
  one("wide",                 64, 256, 512, 8, 0.30, 0.0, 0.0, 0);
  one("k-dominant",          128, 512, 128, 8, 0.30, 0.0, 0.0, 0);

  printf("\n== screening on ==\n");
  one("sparse th=1e-10",     256,256,256, 8, 0.05, 0.0, 1e-10, 0);
  one("sparse th=1e-6",      256,256,256, 8, 0.05, 0.0, 1e-6,  0);
  one("sparse tau=1e-10",    256,256,256, 8, 0.05, 1e-10, 1e-10, 0);
  one("rect sparse",         512,256,128, 8, 0.05, 1e-12, 1e-10, 0);

  printf("\n== block sizes ==\n");
  one("bs=4",                256,256,128, 4, 0.30, 0.0, 0.0, 0);
  one("bs=16",               256,256,128,16, 0.30, 0.0, 0.0, 0);
  one("bs=32",               256,256,128,32, 0.30, 0.0, 0.0, 0);

  printf("\n== realistic block size (b=64): Strassen path must fire ==\n");
  one("b=64 N=512",          512, 512, 512, 64, 0.30, 0.0, 0.0, 0);
  one("b=64 N=1024",        1024,1024,1024, 64, 0.30, 0.0, 0.0, 0);
  one("b=64 rect",          2048, 512, 512, 64, 0.30, 0.0, 0.0, 0);
  one("b=64 ragged",        1000,1000,1000, 64, 0.30, 0.0, 0.0, 0);
  one("b=64 sparse",        1024,1024,1024, 64, 0.05, 1e-12, 1e-10, 0);
  one("b=96 N=1152",        1152,1152, 768, 96, 0.30, 0.0, 0.0, 0);

  printf("\n== degenerate ==\n");
  one("tiny",                 16, 16, 16, 8, 0.30, 0.0, 0.0, 0);
  one("no core (k=1 block)", 128,  8,128, 8, 0.30, 0.0, 0.0, 0);
  one("single block",          8,  8,  8, 8, 0.30, 0.0, 0.0, 0);

  printf("\n%s (%d failure%s)\n", fails?"FAILED":"ALL PASSED", fails, fails==1?"":"s");
  return fails ? 1 : 0;
}
