#ifndef MATRIX_HPP
#define MATRIX_HPP
#include "matrix.h"
#include "blaslapack.h"
#include <algorithm>
#include <numeric>
#include <functional>
#include <cmath>
#include <cstdio>
#include <cstddef>
#include <cassert>
#include <random>
#include "utils.hpp"

template<typename Num,typename Allocator>
Matrix<Num,Allocator>::Matrix(const Allocator& alloc)
 : _data(alloc){}

template<typename Num,typename Allocator>
Matrix<Num,Allocator>::Matrix(size_t nrow, size_t ncol, const Allocator& alloc)
 : _nrow(nrow),
   _ncol(ncol),
   _data(nrow*ncol,Num(),alloc) {}

//initialize all elements with val
template<typename Num,typename Allocator>
Matrix<Num,Allocator>::Matrix(const Num val, size_t nrow, size_t ncol, const Allocator& alloc)
 : _nrow(nrow),
   _ncol(ncol),
   _data(nrow*ncol,val,alloc) 
{
   _frobenius_norm = sqrt((double)(nrow*ncol))*val;
}

//initialize with copies of vals
template<typename Num,typename Allocator>
Matrix<Num,Allocator>::Matrix(const Num* vals, size_t nrow, size_t ncol, const Allocator& alloc)
 : _nrow(nrow),
   _ncol(ncol),
   _data(vals, vals+nrow*ncol,alloc)
{
   this->calc_frobenius_norm();
}

template<typename Num,typename Allocator>
Num Matrix<Num,Allocator>::calc_min() const {
  return *std::min_element(_data.cbegin(),_data.cend());
}
template<typename Num,typename Allocator>
Num Matrix<Num,Allocator>::calc_max() const {
  return *std::max_element(_data.cbegin(),_data.cend());
}

template<typename Num,typename Allocator>
void Matrix<Num,Allocator>::symmetrize_ave(){
  for(size_t r=0;r<_nrow;++r){
    for(size_t c=r+1;c<_ncol;++c){
      const double ave = 0.5e0*(elem(r,c)+elem(c,r));
      elem(r,c) = elem(c,r) = ave;
    }
  }
}
template<typename Num1, typename Num2, typename Allocator1, typename Allocator2>
void assert_sizes(const Matrix<Num1,Allocator1>& lhs,const Matrix<Num2,Allocator2>& rhs){
  if(lhs.nrow() != rhs.nrow() || lhs.ncol() != rhs.ncol()){
    printf("  lhs.nrow() = %lu \t rhs.nrow() = %lu \t lhs.ncol() = %lu \t rhs.ncol() = %lu",
    lhs.nrow(),rhs.nrow(),lhs.ncol(),rhs.ncol());
    puts("Matrix-dimensions do not match");
    exit(1);
  }
}

template<typename Num, typename Allocator>
Matrix<Num,Allocator> operator*(const Matrix<Num,Allocator>& lhs, const Num& rhs){
  Matrix<Num,Allocator> retval = lhs;
  retval*=rhs;
  return retval;
}


template<typename Num,typename Allocator>
Matrix<Num,Allocator> operator+(const Matrix<Num,Allocator>& lhs,const Matrix<Num,Allocator>& rhs){
  assert_sizes(lhs,rhs);
  Matrix<Num,Allocator> retval(lhs);
  retval += rhs;
  return retval;
}

template<typename Num,typename Allocator>
Matrix<Num,Allocator> operator-(const Matrix<Num,Allocator>& lhs,const Matrix<Num,Allocator>& rhs){
  assert_sizes(lhs,rhs);
  Matrix<Num,Allocator> retval(lhs);
  retval -= rhs;
  return retval;
}

template<typename Num,typename Allocator>
Num dot(const Matrix<Num,Allocator>& lhs, const Matrix<Num,Allocator>& rhs){
  assert_sizes(lhs,rhs);
  return std::inner_product(lhs.data().cbegin(),lhs.data().cend(),rhs.data().cbegin(),0.e0);
}

template<typename Num,typename Allocator>
void Matrix<Num,Allocator>::axpy(const Matrix& rhs, const Num scale) & 
{
  assert_sizes(*this,rhs);
  const int isize = (int)this->size();
  const int incx  = 1;
  const int incy  = 1;
  Num* __restrict__ lhs_data = &this->_data[0];
  const Num* __restrict__ rhs_data = &rhs._data[0];
  const Num A = scale;
  any_axpy(&isize,&A,rhs_data,&incx,lhs_data,&incy);
}

template<typename Num,typename Allocator>
Matrix<Num,Allocator>& Matrix<Num,Allocator>::operator+=(const Matrix<Num,Allocator>& rhs) & 
{
  this->axpy(rhs,1.e0);
  return *this;
}

template<typename Num,typename Allocator>
Matrix<Num,Allocator>& Matrix<Num,Allocator>::operator-=(const Matrix<Num,Allocator>& rhs) & {
  this->axpy(rhs,-1.e0);
  return *this;
}

template<typename Num,typename Allocator>
Matrix<Num,Allocator>& Matrix<Num,Allocator>::operator*=(const Num& rhs) & {
  const auto scale_func = [&rhs](const Num& input){return input*rhs;};
  std::transform(_data.cbegin(),_data.cend(),_data.begin(),scale_func);
  //this->apply(scale_func);
  return *this;
}

template<typename Num,typename Allocator>
template<class Functype>
Matrix<Num,Allocator>& Matrix<Num,Allocator>::apply(Functype f) & {
  std::transform(_data.cbegin(),_data.cend(),_data.begin(),f);
  return *this;
}

template<typename Num,typename Allocator>
void Matrix<Num,Allocator>::scale(const Num scale){
  const auto scale_func = [scale](const Num in){return scale*in;};
  this->apply(scale_func);
}

template<typename Num,typename Allocator>
template<typename Num2,typename Allocator2>
Matrix<Num,Allocator>::operator Matrix<Num2,Allocator2>() const{
  Matrix<Num2,Allocator2> retval(_nrow,_ncol);
  std::copy(_data.cbegin(),_data.cend(),retval.data().begin());
  return retval;
}

template<typename Num,typename Allocator>
Matrix<Num,Allocator> operator*(const Matrix<Num,Allocator>& lhs, const Matrix<Num,Allocator>& rhs){
  Matrix<Num,Allocator> retval(lhs.nrow(),rhs.ncol(),rhs.allocator());
  matmult(retval,lhs,false,rhs,false);
  return retval;
}

template<typename Num, typename Allocator>
Matrix<Num,Allocator> transpose(const Matrix<Num,Allocator>& to_transpose){
  Matrix<Num,Allocator> retval(to_transpose.ncol(),to_transpose.nrow());
  for(size_t c=0;c<retval.ncol();++c){
    for(size_t r=0;r<retval.nrow();++r){
      retval.elem(r,c) = to_transpose.elem(c,r);
    }
  }
  return retval;
}

template<typename Num, typename Allocator>
template<typename Num2,typename Allocator2>
void Matrix<Num,Allocator>::add_transpose(const Matrix<Num2,Allocator2>& to_add){
  for(size_t c=0;c<_ncol;++c){
    for(size_t r=0;r<_nrow;++r){
      this->elem(r,c) += to_add.elem(c,r);
    }
  }
}

template<typename Num, typename Allocator>
Num Matrix<Num,Allocator>::calc_frobenius_norm(){
  _frobenius_norm = nrm2(*this);
  return _frobenius_norm;
}

template<typename T, typename Allocator>
void Matrix<T,Allocator>::print(const char* name, const char* format, size_t n_per_row) const {
  ::print(this->data_ptr(),_nrow,_ncol,name,format,n_per_row);
} 

template<typename T, typename Allocator>
void Matrix<T,Allocator>::read_from_file(const char* filename){
  FILE* file_handle = fopen(filename,"rb");
  if(file_handle == nullptr){
    printf("File %s not available\n",filename);
    exit(1);
  }
  const size_t size = get_file_size(file_handle)/sizeof(T);
  assert(size == this->size());
  fseek(file_handle,0,SEEK_SET);
  const auto retval = fread(this->data_ptr(),sizeof(T),size,file_handle);
  assert(retval != 0);
  fclose(file_handle);
}

template<typename T, typename Allocator>
void Matrix<T,Allocator>::write_to_file(const char* filename) const {
  FILE* file_handle = fopen(filename,"wb");
  fseek(file_handle,0,SEEK_SET);
  fwrite(this->data_ptr(),sizeof(T),this->size(),file_handle);
  fclose(file_handle);
}

template<typename Num,typename Allocator1,typename Allocator2,typename Allocator3>
void matmult(Matrix<Num,Allocator1>& C, const Matrix<Num,Allocator2>& A, const bool transA, 
    const Matrix<Num,Allocator3>& B, const bool transB, const Num& alpha, const Num& beta){

  const char ctransA = transA ? 't' : 'n';
  const char ctransB = transB ? 't' : 'n';

  const size_t m1 = C.nrow();
#ifndef NDEBUG
  const size_t m2 = transA ? A.ncol() : A.nrow();
  if(m1 != m2){
    puts("m1 != m2 in matmult()");
    print_stack_trace();
    exit(1);
  }
#endif
  const int m = (int)m1;

  const size_t n1 = C.ncol();
#ifndef NDEBUG
  const size_t n2 = transB ? B.nrow() : B.ncol();
  if(n1 != n2){
    puts("n1 != n2 in matmult()");
    print_stack_trace();
    exit(1);
  }
#endif
  const int n = (int)n1;

  const size_t k1 = transA ? A.nrow() : A.ncol();
  const size_t k2 = transB ? B.ncol() : B.nrow();
  assert(k1 == k2);
#ifndef NDEBUG
  if(k1 != k2){
    puts("k1 != k2 in matmult()");
    print_stack_trace();
    exit(1);
  }
#endif
  const int k = (int)k1;

  const int lda = (int)A.nrow();
  const int ldb = (int)B.nrow();
  const int ldc = (int)C.nrow();

  any_gemm(&ctransA,&ctransB,&m,&n,&k,&alpha,A.data_ptr(),&lda,B.data_ptr(),&ldb,&beta,C.data_ptr(),&ldc);
}


template<typename Num, typename Alloc>
Num nrm2(const Matrix<Num,Alloc>& mat){
  const int isize = (int)mat.size();
  const int iincx = 1;
  return any_nrm2(&isize,mat.data_ptr(),&iincx);
}

template<typename Num, typename Alloc1, typename Alloc2>
Num dot(const Matrix<Num,Alloc1>& A, const Matrix<Num,Alloc1>& B){
  assert_sizes(A,B);
  const int isize = (int)A.size();
  const int iinca = 1;
  const int iincb = 1;
  return any_dot(&isize,A.data_ptr(),&iinca,B.data_ptr(),&iincb);
}

template<typename Num, typename Allocator>
Matrix<Num,Allocator> create_padded_matrix(const Matrix<Num,Allocator>& to_padd, const size_t row_size, const size_t col_size)
{
  //intialize with zeros 
  Matrix<Num,Allocator> retval(0.e0,row_size,col_size);
  const size_t min_nrow = std::min(retval.nrow(),to_padd.nrow());
  const size_t min_ncol = std::min(retval.ncol(),to_padd.ncol());
  //will all the available elements (padding will remain as zeros)
  #pragma omp parallel for schedule(static) collapse(2)
  for(size_t c=0;c<min_ncol;++c){
    for(size_t r=0;r<min_nrow;++r){
      retval.elem(r,c) = to_padd.elem(r,c);
    }
  }
  return retval;
}

template<typename Num, typename Alloc>
void Matrix<Num,Alloc>::create_pixmap(const char* file_name) const{

  FILE* output_file = fopen(file_name,"w");
  fprintf(output_file,"/* XPM */\nstatic char * matrix_xpm[] = {\n\"%i %i 12 1\",\n",(int)_ncol,(int)_nrow);
  fprintf(output_file,"\"a\tc #ff0000\",\n");   // > 1
  fprintf(output_file,"\"b\tc #ff3300\",\n"); // > e-01
  fprintf(output_file,"\"c\tc #ff3333\",\n"); // > e-02
  fprintf(output_file,"\"d\tc #ff6633\",\n"); // > e-03
  fprintf(output_file,"\"e\tc #ff6666\",\n"); // > e-04
  fprintf(output_file,"\"f\tc #ff9966\",\n"); // > e-05
  fprintf(output_file,"\"g\tc #ff9999\",\n"); // > e-06
  fprintf(output_file,"\"h\tc #ffbb99\",\n"); // > e-07
  fprintf(output_file,"\"i\tc #ffbbbb\",\n"); // > e-08
  fprintf(output_file,"\"j\tc #ffeebb\",\n"); // > e-09
  fprintf(output_file,"\"k\tc #ffeeee\",\n"); // > e-10
  fprintf(output_file,"\".\tc #ffffff\",\n"); // zero

  for(size_t row=0;row<_nrow;++row){
    fprintf(output_file,"\"");
    for(size_t col=0;col<_ncol;++col){
      const auto absval = std::abs(this->elem(row,col));
      if      (absval >= Num(1.e+00)) fprintf(output_file,"a");
      else if (absval >= Num(1.e-01)) fprintf(output_file,"b");
      else if (absval >= Num(1.e-02)) fprintf(output_file,"c");
      else if (absval >= Num(1.e-03)) fprintf(output_file,"d");
      else if (absval >= Num(1.e-04)) fprintf(output_file,"e");
      else if (absval >= Num(1.e-05)) fprintf(output_file,"f");
      else if (absval >= Num(1.e-06)) fprintf(output_file,"g");
      else if (absval >= Num(1.e-07)) fprintf(output_file,"h");
      else if (absval >= Num(1.e-08)) fprintf(output_file,"i");
      else if (absval >= Num(1.e-09)) fprintf(output_file,"j");
      else if (absval >= Num(1.e-10)) fprintf(output_file,"k");
      else fprintf(output_file,".");
    }
    fprintf(output_file,"\",\n");
  }

  fprintf(output_file,"};");
  fclose(output_file);

}
template<class Numeric>
void fill_with_uniform_pseudo_random_numbers(Numeric* __restrict__ data, const Numeric min, const Numeric max, const size_t size, const int seed=42);

// Allocates ana array with random Numeric entries.
template<class Numeric>
void fill_with_uniform_pseudo_random_numbers(Numeric* __restrict__ data, const Numeric min, const Numeric max, const size_t size, const int seed) {
  std::uniform_real_distribution<Numeric> distribution(min,max);
  #pragma omp parallel
  {
    std::default_random_engine generator(seed);
    #pragma omp for schedule(static)
    for (size_t i = 0; i < size; ++i) data[i] = distribution(generator);
  }
}

template<typename Num, typename Alloc>
void Matrix<Num,Alloc>::fill_with_uniform_pseudo_random_numbers(const Num min, const Num max, const int seed) & {
  ::fill_with_uniform_pseudo_random_numbers(this->data_ptr(),min,max,this->size(),seed);
}

#endif

