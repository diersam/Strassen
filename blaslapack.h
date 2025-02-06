#ifndef BLASLAPACK_H
#define BLASLAPACK_H

//generic blas header
template <typename Num>
void any_axpy(const int* N ,const Num* A, const Num* X, const int* INCX, Num* Y,const int* INCY);

template <typename Num>
Num any_dot(const int*,const Num*,const int*,const Num*,const int*);

template <typename Num>
Num any_nrm2(const int*,const Num*,const int*);

template <typename Num>
void any_gemm(const char* transa,const char* transb ,const int* m,const int* n,const int* k,
    const Num* alpha,const Num* A,const int* lda,const Num* B,const int* ldb,const Num* beta ,Num* C,const int* ldc);

//BLAS/LAPACK header double-precision
extern "C" {
void saxpy_(const int* N ,const float* A, const float* X, const int* INCX, float* Y,const int* INCY);
void daxpy_(const int* N ,const double* A, const double* X, const int* INCX, double* Y,const int* INCY);
  void sgemm_(const char*,const char*,const int*,const int*,const int*,const float*,const float*,const int*,const float*,const int*,const float*,float*,const int*);
  void dgemm_(const char*,const char*,const int*,const int*,const int*,const double*,const double*,const int*,const double*,const int*,const double*,double*,const int*);
  //void daxpy_(const int*,const double*,const double*,const int*,double*,const int*);
  double ddot_(const int*,const double*,const int*,const double*,const int*);
  double dnrm2_(const int*,const double*,const int*);
}
#endif

