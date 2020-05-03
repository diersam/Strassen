#include "matrix.h"
#include <algorithm>
#include <numeric>
#include <functional>
#include <cmath>
#include <cstdio>
#include <cstddef>
#include <cassert>

template<typename Num,typename Allocator>
Matrix<Num,Allocator>::Matrix(size_t nrow, size_t ncol, const Allocator& alloc)
 : _data(nrow*ncol,alloc),
   _nrow(nrow),
   _ncol(ncol) {}

//initialize with copies of vals
template<typename Num,typename Allocator>
Matrix<Num,Allocator>::Matrix(const Num* vals, size_t nrow, size_t ncol, const Allocator& alloc)
 : _data(vals, vals+nrow*ncol,alloc),
   _nrow(nrow),
   _ncol(ncol) {}

template<typename Num,typename Allocator>
Num Matrix<Num,Allocator>::calc_min() const {
  return *std::min_element(_data.cbegin(),_data.cend());
}
template<typename Num,typename Allocator>
Num Matrix<Num,Allocator>::calc_max() const {
  return *std::max_element(_data.cbegin(),_data.cend());
}

const auto abs_compare = [](const auto lhs, const auto rhs){return std::abs(lhs) < std::abs(rhs);};
template<typename Num,typename Allocator>
Num Matrix<Num,Allocator>::calc_abs_min() const {
  _abs_max = std::abs(*std::min_element(_data.cbegin(),_data.cend(),abs_compare));
  return _abs_max;
}

template<typename Num,typename Allocator>
Num Matrix<Num,Allocator>::calc_abs_max() {
  _abs_max = std::abs(*std::max_element(_data.cbegin(),_data.cend(),abs_compare));
  return _abs_max;
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
Matrix<Num,Allocator>& Matrix<Num,Allocator>::operator+=(const Matrix<Num,Allocator>& rhs) & {
  assert_sizes(*this,rhs);
  std::transform(rhs._data.cbegin(),rhs._data.cend(),this->_data.cbegin(),this->_data.begin(),std::plus<>());
  return *this;
}

template<typename Num,typename Allocator>
Matrix<Num,Allocator>& Matrix<Num,Allocator>::operator-=(const Matrix<Num,Allocator>& rhs) & {
  assert_sizes(*this,rhs);
  std::transform(rhs._data.cbegin(),rhs._data.cend(),this->_data.cbegin(),this->_data.begin(),std::minus<>());
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
Matrix<Num,Allocator>& Matrix<Num,Allocator>::apply(Num func(Num)) & {
  std::transform(_data.cbegin(),_data.cend(),_data.begin(),func);
  return *this;
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
  Matrix<Num,Allocator> retval(lhs.nrow(),rhs.ncol());
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

