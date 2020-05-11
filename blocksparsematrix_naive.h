#ifndef BLOCKSPARSEMATRIX_NAIVE_H
#define BLOCKSPARSEMATRIX_NAIVE_H

#include <vector>
#include "matrix.h"

//BlockSparseMatrix with std::allocator
template<typename Num>
class BlockSparseMatrix_naive final{
  using Mat = Matrix<Num,std::allocator<Num>>;
  public:
    explicit BlockSparseMatrix_naive() = default;//plain matrix
    //empty Matrix (but parameters set)
    explicit BlockSparseMatrix_naive(size_t nr, size_t nc, size_t target_blocksize_row, size_t target_blocksize_col, Num thresh_in);
    //initialize from input array
    explicit BlockSparseMatrix_naive(const Num* input_vals, size_t nr, size_t nc, size_t target_blocksize_row, size_t target_blocksize_col, Num thresh_in);
    //initialize from dense Matrix
    explicit BlockSparseMatrix_naive(const Mat& in, //todo allocator
        size_t target_blocksize_row, size_t target_blocksize_col, Num thresh_in);

    void scale(Num scale);
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

    BlockSparseMatrix_naive& operator-=(const BlockSparseMatrix_naive& rhs);
    BlockSparseMatrix_naive& operator+=(const BlockSparseMatrix_naive& rhs);
    BlockSparseMatrix_naive& operator*=(Num rhs);

    Mat to_matrix() const;

    const std::vector<Mat> blocks() const & {return _blocks;}
    std::vector<Mat> blocks() &             {return _blocks;}
    Mat& block(const size_t row_block, const size_t col_block) & {            return _blocks[row_block + col_block*_nrowblocks];}
    const Mat& block(const size_t row_block, const size_t col_block) const & {return _blocks[row_block + col_block*_nrowblocks];}

    void calc_frobenius_norms();//calc L2 norms of all blocks
    void recompress(); //delete insignificant blocks
    Num frobenius_norm() const;//L2 norm accumulated from all blocks

  private:
    std::vector<Mat> _blocks;
    size_t _nrow = 0;
    size_t _ncol = 0;
    size_t _nrowblocks = 0;
    size_t _ncolblocks = 0;
    size_t _max_blocksize_row = 0;
    size_t _max_blocksize_col = 0;
    Num _thresh = Num();
};

template<typename Num>
void matmult(BlockSparseMatrix_naive<Num>& C, 
             const BlockSparseMatrix_naive<Num>& A, const bool transA,
             const BlockSparseMatrix_naive<Num>& B, const bool transB,
             const Num thresh, const Num& alpha = Num(1), const Num& beta = Num(0));
#endif

