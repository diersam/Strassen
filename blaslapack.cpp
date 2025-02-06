#include "blaslapack.h"

//specialize generic blas header

template<>
void any_axpy<float>(const int* N ,const float* A, const float* X, const int* INCX,float* Y,const int* INCY)
{
  saxpy_(N,A,X,INCX,Y,INCY);
}

template<>
void any_axpy<double>(const int* N ,const double* A, const double* X, const int* INCX,double* Y,const int* INCY)
{
  daxpy_(N,A,X,INCX,Y,INCY);
}

template<> 
void any_gemm<float>(const char* transa,const char* transb ,const int* m,const int* n,const int* k,
    const float* alpha,const float* A,const int* lda,const float* B,const int* ldb,const float* beta ,float* C,const int* ldc){
  sgemm_(transa,transb,m,n,k,alpha,A,lda,B,ldb,beta,C,ldc);
}

template<> 
void any_gemm<double>(const char* transa,const char* transb ,const int* m,const int* n,const int* k,
    const double* alpha,const double* A,const int* lda,const double* B,const int* ldb,const double* beta ,double* C,const int* ldc){
  dgemm_(transa,transb,m,n,k,alpha,A,lda,B,ldb,beta,C,ldc);
}

template<>
double any_dot<double>(const int* size,const double* A,const int* inca,const double* B,const int* incb){
  return ddot_(size,A,inca,B,incb);
}

template<>
double any_nrm2<double>(const int* size,const double* A,const int* inca){
  return dnrm2_(size,A,inca);
}

extern "C"{
  int mkl_serv_intel_cpu_true() {
    return 1;
  }
}


