#ifndef MATRIX_H
#define MATRIX_H
#include <vector>
#include <cstdio>


//Dense matrix class for different numeric types (float,double,complex<float>,complex<double>)
//Allows for userdefined Allocators
template<typename Num, typename Allocator = std::allocator<Num>>
class Matrix final{
  public:
  explicit Matrix(const Allocator& alloc = Allocator());
  explicit Matrix(size_t nrow, size_t ncol, const Allocator& alloc = Allocator());
  //initialize all elements with val
  explicit Matrix(const Num val, size_t nrow, size_t ncol, const Allocator& alloc = Allocator());
  //initialize with copies of vals
  explicit Matrix(const Num* vals, size_t nrow, size_t ncol, const Allocator& alloc = Allocator());

  std::vector<Num,Allocator>& data() & {return _data;}
  const std::vector<Num,Allocator>& data() const & {return _data;}
  Num* data_ptr() & {return _data.data();}
  const Num* data_ptr() const & {return _data.data();}
  size_t nrow() const {return _nrow;}
  size_t ncol() const {return _ncol;}
  size_t size() const {return _nrow*_ncol;}

  void print(const char* name = "", const char* format="%10.5f", size_t n_per_row=6) const;
  Matrix& operator-=(const Matrix& rhs) & ;
  Matrix& operator+=(const Matrix& rhs) & ;
  Matrix& operator*=(const Num& rhs) & ;
  Matrix& apply(Num func(Num)) & ;
  template<typename Num2,typename Allocator2>
  void add_transpose(const Matrix<Num2,Allocator2>& to_add);

  Num calc_frobenius_norm();
  Num& frobenius_norm()             & {return _frobenius_norm;};
  const Num& frobenius_norm() const & {return _frobenius_norm;};

  Num calc_min() const;
  Num calc_max() const;
  Num& elem(size_t row, size_t col) & {return _data[row + col*_nrow];}
  const Num& elem(size_t row, size_t col) const & {return _data[row + col*_nrow];}

  template<typename Num2, typename Allocator2>
  explicit operator Matrix<Num2,Allocator2>() const;

  void symmetrize_ave();

  private:

  size_t _nrow = 0;
  size_t _ncol = 0;
  Num _frobenius_norm = Num();
  std::vector<Num,Allocator> _data;
};

using DMat = Matrix<double,std::allocator<double>>;

template<typename Num, typename Allocator>
Matrix<Num,Allocator> operator*(const Matrix<Num,Allocator>& lhs, const Num& rhs);
template<typename Num, typename Allocator>
Matrix<Num,Allocator> operator*(const Num& lhs,const Matrix<Num,Allocator>& rhs){return rhs*lhs;}
template<typename Num, typename Allocator>
Matrix<Num,Allocator> operator*(const Matrix<Num,Allocator>& lhs, const Matrix<Num,Allocator>& rhs);
template<typename Num, typename Allocator>
Matrix<Num,Allocator> operator+(const Matrix<Num,Allocator>& lhs,const Matrix<Num,Allocator>& rhs);
template<typename Num, typename Allocator>
Matrix<Num,Allocator> operator-(const Matrix<Num,Allocator>& lhs,const Matrix<Num,Allocator>& rhs);

template<typename Num>
void print(const Num* mat_ptr, const size_t nrow, const size_t ncol, 
           const char* name = "", const char* format="%10.5f", size_t n_per_row=6);

template<typename Num,typename Allocator1,typename Allocator2,typename Allocator3>
void matmult(Matrix<Num,Allocator1>& C, const Matrix<Num,Allocator2>& A, const bool transA, 
    const Matrix<Num,Allocator3>& B, const bool transB, const Num& alpha = Num(1), const Num& beta = Num(0));

template<typename Num1, typename Num2, typename Allocator1, typename Allocator2>
void assert_sizes(const Matrix<Num1,Allocator1>& lhs,const Matrix<Num2,Allocator2>& rhs);

template<typename Num, typename Allocator>
Num dot(const Matrix<Num,Allocator>& lhs, const Matrix<Num,Allocator>& rhs);

template<typename Num, typename Allocator>
Matrix<Num,Allocator> transpose(const Matrix<Num,Allocator>& to_transpose);


template<typename Num, typename Alloc>
Num nrm2(const Matrix<Num,Alloc>& mat);//L2 norm
#endif

