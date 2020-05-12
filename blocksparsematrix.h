#ifndef BLOCKSPARSEMATRIX_H
#define BLOCKSPARSEMATRIX_H

#include <vector>
#include "matrix.h"
#include "blockallocator.h"
#include "blockallocator.hpp"

//BlockSparseMatrix with blockallocator
template<typename Num>
class BlockSparseMatrix final{
  using Mat = Matrix<Num,BlockAllocator<Num>>;
  public:
    explicit BlockSparseMatrix() = default;//plain matrix
    //empty Matrix (but parameters set)
    explicit BlockSparseMatrix(size_t nr, size_t nc, size_t target_blocksize_row, size_t target_blocksize_col, Num thresh_in);
    //initialize from input array
    explicit BlockSparseMatrix(const Num* input_vals, size_t nr, size_t nc, size_t target_blocksize_row, size_t target_blocksize_col, Num thresh_in);
    //initialize from dense Matrix (irrespective of its allocator)
    template <class Allocator>
    explicit BlockSparseMatrix(const Matrix<Num,Allocator>& in, 
        size_t target_blocksize_row, size_t target_blocksize_col, Num thresh_in);

    //generate from input array
    void copy_from_input_array(const Num* __restrict__ input_vals);
    //generate from input matrix
    template <class Allocator>
    void copy_from_input_matrix(const Matrix<Num,Allocator>& in);

    //print significant blocks as pixmap
    void create_block_pixmap(const char* file_name) const;

    void scale(Num scale);
    void zero();
    //write into output array
    void to_pointer(Num* __restrict__ output_ptr) const;
    //print in dense format
    void print(const char* name = "", const char* format="%10.5f", size_t n_per_row=6) const;

    size_t nrow() const {return _nrow;}
    size_t ncol() const {return _ncol;}
    size_t nrowblocks() const {return _nrowblocks;}
    size_t ncolblocks() const {return _ncolblocks;}
    size_t max_blocksize_row() const {return _max_blocksize_row;}
    size_t max_blocksize_col() const {return _max_blocksize_col;}
    Num thresh() const {return _thresh;}

    size_t nblocks() const {return _blocks.size();}
    size_t no_of_alloc_blocks() const;//count no of allocated blocks
    size_t calc_alloc_dim() const;//sums actual allocation size of all blocks

    BlockSparseMatrix& operator-=(const BlockSparseMatrix& rhs);
    BlockSparseMatrix& operator+=(const BlockSparseMatrix& rhs);
    BlockSparseMatrix& operator*=(Num rhs);

    Matrix<Num,std::allocator<Num>> to_matrix() const;

    const std::vector<Mat> blocks() const & {return _blocks;}
    std::vector<Mat> blocks() &             {return _blocks;}
    Mat& block(const size_t row_block, const size_t col_block) & {            return _blocks[row_block + col_block*_nrowblocks];}
    const Mat& block(const size_t row_block, const size_t col_block) const & {return _blocks[row_block + col_block*_nrowblocks];}

    void calc_frobenius_norms();//calc L2 norms of all blocks
    void recompress(); //delete insignificant blocks
    Num frobenius_norm() const;//L2 norm accumulated from all blocks
    BlockAllocator<Num> allocator() & {return BlockAllocator<Num>(_mem_pool);}
    const BlockAllocator<Num> allocator() const & {return BlockAllocator<Num>(_mem_pool);}
    BlockMemoryPool<Num>& mem_pool() & {return _mem_pool;}
    const BlockMemoryPool<Num>& mem_pool() const & {return _mem_pool;}

  private:
    size_t _nrow = 0;
    size_t _ncol = 0;
    size_t _nrowblocks = 0;
    size_t _ncolblocks = 0;
    size_t _max_blocksize_row = 0;
    size_t _max_blocksize_col = 0;
    Num _thresh = Num();
    BlockMemoryPool<Num> _mem_pool;
    std::vector<Mat> _blocks;
};

template<typename Num>
void matmult(BlockSparseMatrix<Num>& C, 
             const BlockSparseMatrix<Num>& A, const bool transa,
             const BlockSparseMatrix<Num>& B, const bool transb,
             const Num thresh, const Num& alpha = Num(1), const Num& beta = Num(0));

template<typename Num>
Num dot(const BlockSparseMatrix<Num>& lhs, const BlockSparseMatrix<Num>& rhs, Num thresh = Num(0));
//BlockSparseMatrix transpose(const BlockSparseMatrix& A, int nthreads = -1);

#endif

