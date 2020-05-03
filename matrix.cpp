#include "matrix.h"
#include "matrix.hpp"

//BLAS/LAPACK header double-precision
extern "C" {
  void dgemm_(const char*,const char*,const int*,const int*,const int*,const double*,const double*,const int*,const double*,const int*,const double*,double*,const int*);
  //void daxpy_(const int*,const double*,const double*,const int*,double*,const int*);
  double ddot_(const int*,const double*,const int*,const double*,const int*);
  double dnrm2_(const int*,const double*,const int*);
}

template<>
double Matrix<double,std::allocator<double>>::calc_frobenius_norm() {
  const int isize = (int)_data.size();
  const int iincx = 1;
  _frobenius_norm = dnrm2_(&isize,this->data_ptr(),&iincx);
  return _frobenius_norm;
}

//template<typename Allocator1,typename Allocator2,typename Allocator3>
//void matmult(Matrix<double,Allocator1>& C, const Matrix<double,Allocator2>& A, const bool transA, 
    //const Matrix<double, Allocator3>& B, const bool transB, const double& alpha, const double& beta){
template<>
void matmult(Matrix<double,std::allocator<double>>& C, const Matrix<double,std::allocator<double>>& A, const bool transA, 
    const Matrix<double, std::allocator<double>>& B, const bool transB, const double& alpha, const double& beta){
  const char ctransA = transA ? 't' : 'n';
  const char ctransB = transB ? 't' : 'n';

  const size_t m1 = C.nrow();
  const size_t m2 = transA ? A.ncol() : A.nrow();
  assert(m1 == m2);
  const int m = (int)m1;

  const size_t n1 = C.ncol();
  const size_t n2 = transB ? B.nrow() : B.ncol();
  assert(n1 == n2);
  const int n = (int)n1;

  const size_t k1 = transA ? A.nrow() : A.ncol();
  const size_t k2 = transB ? B.ncol() : B.nrow();
  assert(k1 == k2);
  const int k = (int)k1;

  const int lda = (int)A.nrow();
  const int ldb = (int)B.nrow();
  const int ldc = (int)C.nrow();

  dgemm_(&ctransA,&ctransB,&m,&n,&k,&alpha,A.data_ptr(),&lda,B.data_ptr(),&ldb,&beta,C.data_ptr(),&ldc);
}


//template<typename Allocator>
//double dot(const Matrix<double>& A, const Matrix<double>& B){
template<>
double dot(const Matrix<double,std::allocator<double>>& A, const Matrix<double,std::allocator<double>>& B){
  assert_sizes(A,B);
  const int isize = (int)A.size();
  const int iinca = 1;
  const int iincb = 1;
  return ddot_(&isize,A.data_ptr(),&iinca,B.data_ptr(),&iincb);
}

//template<typename Allocator>
//void Matrix<double,Allocator>::print(const char* name, const char* format, size_t n_per_row) const {
template<>
void Matrix<double,std::allocator<double>>::print(const char* name, const char* format, size_t n_per_row) const {
  puts(name);
  for (size_t bcol=0;bcol<_ncol;bcol+=n_per_row){
    const size_t start = bcol;
    const size_t end = std::min(start + n_per_row, _ncol);
    for (size_t col=start;col<end;++col){
      printf("        %4lu",col+1);//column header
    }
    puts("");
    for (size_t row=0;row<_nrow;++row){
      for (size_t col=start;col<end;++col){
        printf(format,  this->elem(row,col));
      }
      puts("");
    }
    puts("");
  }
  puts("");
}

//instantiate

