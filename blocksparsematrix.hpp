#ifndef BLOCKSPARSEMATRIX_HPP
#define BLOCKSPARSEMATRIX_HPP
#include "blocksparsematrix.h"
#include "utils.hpp"

template<typename Num>
BlockSparseMatrix<Num>::BlockSparseMatrix(
    const size_t nr, const size_t nc, const size_t target_blocksize_row, const size_t target_blocksize_col, const Num thresh_in)
  : _nrow(nr),
    _ncol(nc),
    _nrowblocks(integer_division_round_up(nr,target_blocksize_row)),
    _ncolblocks(integer_division_round_up(nc,target_blocksize_col)),
    _max_blocksize_row(target_blocksize_row),
    _max_blocksize_col(target_blocksize_col),
    _thresh(thresh_in),
    _mem_pool_ptr(new BlockMemoryPool<Num>(target_blocksize_row*target_blocksize_col)),
    _blocks(_nrowblocks*_ncolblocks,Mat(this->allocator())) {}

//initialize from input array
template<typename Num>
BlockSparseMatrix<Num>::BlockSparseMatrix(const Num* __restrict__ input_vals, size_t nr, size_t nc,
    size_t target_blocksize_row, size_t target_blocksize_col, Num thresh_in)
  : BlockSparseMatrix(nr,nc,target_blocksize_row,target_blocksize_col,thresh_in)
{
  this->copy_from_input_array(input_vals);
}

//generate values from input array
template<typename Num>
void BlockSparseMatrix<Num>::copy_from_input_array(const Num* __restrict__ input_vals){
  //generate every block matrix
  #pragma omp parallel for schedule(dynamic)
  for(size_t rcb=0;rcb<_nrowblocks*_ncolblocks;++rcb){
    const size_t col_block = rcb/_nrowblocks;
    const auto col_start = col_block*_max_blocksize_col;
    const auto col_end = std::min((col_block+1)*_max_blocksize_col,_ncol);
    const auto col_size = col_end - col_start;

    const size_t row_block = rcb%_nrowblocks;
    const auto row_start = row_block*_max_blocksize_row;
    const auto row_end = std::min((row_block+1)*_max_blocksize_row,_nrow);
    const auto row_size = row_end - row_start;

    Mat& temp = this->block(row_block,col_block);
    temp = Mat(row_size,col_size,this->allocator());
    for(size_t subcol=0;subcol<temp.ncol();++subcol){
      const auto col = col_start + subcol;
      std::copy_n(&input_vals[row_start + col*_nrow],row_size,&temp.elem(0,subcol));
    }
    const Num est = temp.calc_frobenius_norm();
    if (est < _thresh) temp = Mat(this->allocator());
  }
}

//generate from input matrix
template<typename Num>
template <class Allocator>
void BlockSparseMatrix<Num>::copy_from_input_matrix(const Matrix<Num,Allocator>& in){
  assert(this->nrow() == in.nrow());
  assert(this->ncol() == in.ncol());
  this->copy_from_input_array(in.data_ptr());
}

template<typename Num>
template <class Allocator>
BlockSparseMatrix<Num>::BlockSparseMatrix(const Matrix<Num,Allocator>& in, 
        size_t target_blocksize_row, size_t target_blocksize_col, Num thresh_in)
  : BlockSparseMatrix(in.data_ptr(),in.nrow(),in.ncol(),target_blocksize_row,target_blocksize_col,thresh_in) {}

template<typename Num>
void BlockSparseMatrix<Num>::scale(const Num scale){
  #pragma omp parallel for schedule(dynamic)
  for(size_t b=0;b<_blocks.size();++b) _blocks[b] *= scale;
  //also rescale norms
  for(size_t b=0;b<_blocks.size();++b) _blocks[b].frobenius_norm() *= scale*scale;
}

template<typename Num>
void BlockSparseMatrix<Num>::zero(){
  for (auto& block : _blocks) block = Mat(this->allocator());
}

template<typename Num>
void BlockSparseMatrix<Num>::to_pointer(Num* __restrict__ output_ptr) const{
  //generate every block matrix
  #pragma omp parallel for schedule(dynamic)
  for(size_t rcb=0;rcb<_nrowblocks*_ncolblocks;++rcb){
    const size_t col_block = rcb/_nrowblocks;
    const auto col_start = col_block*_max_blocksize_col;

    const size_t row_block = rcb%_nrowblocks;
    const auto row_start = row_block*_max_blocksize_row;

    const auto& block_act = this->block(row_block,col_block);
    
    if (block_act.size() != 0){//significant block --> copy
      //copy block col-wise
      for(size_t subcol=0;subcol<block_act.ncol();++subcol){
        const auto col = col_start + subcol;
        std::copy_n(&block_act.elem(0,subcol),block_act.nrow(),&output_ptr[row_start + _nrow*col]);
      }
    }
  }
}

template<typename Num>
Matrix<Num,std::allocator<Num>> BlockSparseMatrix<Num>::to_matrix() const{
  Matrix<Num,std::allocator<Num>> retval(_nrow,_ncol);
  this->to_pointer(retval.data_ptr());
  return retval;
}

template<typename Num>
void BlockSparseMatrix<Num>::print(const char* name, const char* format, const size_t n_per_row) const {
  const auto mat = this->to_matrix();
  mat.print(name,format,n_per_row);
}

template<typename Num>
void BlockSparseMatrix<Num>::calc_frobenius_norms(){
  #pragma omp parallel for schedule(dynamic)
  for(size_t rcb=0;rcb<_nrowblocks*_ncolblocks;++rcb){
    const size_t col_block = rcb/_nrowblocks;
    const size_t row_block = rcb%_nrowblocks;
    this->block(row_block,col_block).calc_frobenius_norm();
  }
}

template<typename Num>
void BlockSparseMatrix<Num>::recompress(){
  for(auto& block : _blocks){
    if(block.frobenius_norm() < _thresh) block = Mat(this->allocator());
  }
}

template<typename Num>
size_t BlockSparseMatrix<Num>::no_of_alloc_blocks() const{
  const auto is_block_not_empty = [](const auto& in){return in.size() > 0 ? true : false;};
  return std::count_if(_blocks.begin(),_blocks.end(),is_block_not_empty);
}

template<typename Num>
size_t BlockSparseMatrix<Num>::calc_alloc_dim() const{
  const auto sum_block_size = [](const size_t& lhs,const auto& rhs){return lhs + rhs.size();};
  return std::accumulate(_blocks.begin(),_blocks.end(),0,sum_block_size);
}

template<typename Num>
BlockSparseMatrix<Num>& BlockSparseMatrix<Num>::operator+=(const BlockSparseMatrix<Num>& rhs){
  assert(this->nblocks()    == rhs.nblocks());
  assert(this->nrowblocks() == rhs.nrowblocks());
  assert(this->ncolblocks() == rhs.ncolblocks());
  assert(this->nrow() == rhs.nrow());
  assert(this->ncol() == rhs.ncol());
  const size_t nijb = this->nblocks();
  const size_t njb = this->ncolblocks();
  //parallel loop over blocks
  #pragma omp parallel for schedule(dynamic)
  for(size_t ijb=0;ijb<nijb;++ijb){
    const size_t ib = ijb/njb;
    const size_t jb = ijb%njb;
    const auto& rhs_block = rhs.block(ib,jb);
    if(rhs_block.size() != 0){//nothing to add otherwise
      auto& this_block = this->block(ib,jb);
      if(this_block.size() == 0) this_block = rhs_block;
      else                       this_block += rhs_block;
      //recompute norms
      this_block.calc_frobenius_norm();
    }
  }
  return *this;
}

template<typename Num>
BlockSparseMatrix<Num>& BlockSparseMatrix<Num>::operator-=(const BlockSparseMatrix<Num>& rhs){
  assert(this->nblocks()    == rhs.nblocks());
  assert(this->nrowblocks() == rhs.nrowblocks());
  assert(this->ncolblocks() == rhs.ncolblocks());
  assert(this->nrow() == rhs.nrow());
  assert(this->ncol() == rhs.ncol());
  const size_t nijb = this->nblocks();
  const size_t njb = this->ncolblocks();

  //parallel loop over blocks
  #pragma omp parallel for schedule(dynamic)
  for(size_t ijb=0;ijb<nijb;++ijb){
    const size_t ib = ijb/njb;
    const size_t jb = ijb%njb;
    const auto& rhs_block = rhs.block(ib,jb);
    if(rhs_block.size() != 0){//nothing to add otherwise
      auto& this_block = this->block(ib,jb);
      if(this_block.size() == 0) this_block  = rhs_block;
      else                       this_block -= rhs_block;
      //recompute norms
      this_block.calc_frobenius_norm();
    }
  }
  return *this;
}

template<typename Num>
BlockSparseMatrix<Num>& BlockSparseMatrix<Num>::operator*=(const Num scale){
  this->scale(scale);
  return *this;
}

template<typename Num>
Num BlockSparseMatrix<Num>::frobenius_norm() const {
  //just add the L2 norms of all blocks
  return std::sqrt(std::accumulate(_blocks.begin(),_blocks.end(),Num(0),
    [](const Num norm, const auto& block){return block.frobenius_norm()*block.frobenius_norm() + norm;}));
}

template<typename Num>
Num dot(const BlockSparseMatrix<Num>& lhs, const BlockSparseMatrix<Num>& rhs, Num thresh){
  assert(lhs.nblocks()    == rhs.nblocks());
  assert(lhs.nrowblocks() == rhs.nrowblocks());
  assert(lhs.ncolblocks() == rhs.ncolblocks());
  assert(lhs.nrow() == rhs.nrow());
  assert(lhs.ncol() == rhs.ncol());
  const size_t nijb = lhs.nblocks();
  const size_t njb = lhs.ncolblocks();

  Num retval = Num(0);
  //parallel loop over blocks
  #pragma omp parallel for schedule(dynamic) reduction(+: retval)
  for(size_t ijb=0;ijb<nijb;++ijb){
    const size_t ib = ijb/njb;
    const size_t jb = ijb%njb;
    const auto& lhs_block = lhs.block(ib,jb);
    const auto& rhs_block = rhs.block(ib,jb);
    if(lhs_block.size() != 0 && rhs_block.size() != 0){
      if(lhs_block.frobenius_norm()*rhs_block.frobenius_norm() >= thresh){
        retval += dot(lhs_block,rhs_block);
      }
    }
  }
  return retval;
}

template<typename Num>
void matmult(BlockSparseMatrix<Num>& C, 
             const BlockSparseMatrix<Num>& A, const bool transA,
             const BlockSparseMatrix<Num>& B, const bool transB,
             const Num thresh, const Num& alpha, const Num& beta)
{
  using Mat = Matrix<Num,BlockAllocator<Num>>;
  if (beta == Num(1)){
    //nothing to do
  }else if (beta == Num(0)){
    C.zero();
  }else{
    C *= beta;
  }

  //check dimensions
  const size_t ni = transA? A.ncol() : A.nrow();
  const size_t nj = transB? B.nrow() : B.ncol();
  const size_t nk1 = transA? A.nrow() : A.ncol();
  const size_t nk2 = transB? B.ncol() : B.nrow();

  assert(nk1 == nk2);
  const size_t nk = nk1;

  const size_t i_block_size = transA? A.max_blocksize_col() : A.max_blocksize_row();
  const size_t j_block_size = transB? B.max_blocksize_row() : B.max_blocksize_col();

  assert(i_block_size == C.max_blocksize_row());
  assert(j_block_size == C.max_blocksize_col());
  assert(ni == C.nrow());
  assert(nj == C.ncol());

  const size_t k_block_size1 = transA? A.max_blocksize_row() : A.max_blocksize_col();
  const size_t k_block_size2 = transB? B.max_blocksize_col() : B.max_blocksize_row();
  assert(k_block_size1 == k_block_size2);
  const size_t k_block_size = k_block_size1;

  const size_t nib = integer_division_round_up(ni,i_block_size);
  assert(nib == C.nrowblocks());
  assert(nib == transA? A.ncolblocks() : A.nrowblocks());
  const size_t njb = integer_division_round_up(nj,j_block_size);
  assert(njb == C.ncolblocks());
  assert(njb == transB? B.nrowblocks() : B.ncolblocks());
  const size_t nkb = integer_division_round_up(nk,k_block_size);
  assert(nkb == transA? A.nrowblocks() : A.ncolblocks());
  assert(nkb == transB? B.ncolblocks() : B.nrowblocks());

  size_t nsig = 0;
  #pragma omp parallel for schedule(dynamic) reduction(+: nsig)
  for(size_t ijb=0;ijb<nib*njb;++ijb){
    const size_t jb = ijb/nib;
    const size_t ib = ijb%nib;
    for(size_t kb=0;kb<nkb;++kb){
      auto& c_block = C.block(ib,jb);
      const auto& a_block = transA? A.block(kb,ib) : A.block(ib,kb);
      const auto& b_block = transB? B.block(jb,kb) : B.block(kb,jb);
      if (a_block.size() != 0 && b_block.size() != 0) {//only for existing block combi
        const auto ni_act = transA? a_block.ncol() : a_block.nrow();
        const auto nj_act = transB? b_block.nrow() : b_block.ncol();
        const Num norm_a = a_block.frobenius_norm();
        const Num norm_b = b_block.frobenius_norm();
        const Num est = norm_a*norm_b;
        if (est >= thresh) {
          nsig++;
          if(c_block.size() == 0){// if not yet existing
            c_block = Mat(ni_act,nj_act,C.allocator());//allocate C block
          }
          matmult(c_block,a_block,transA,b_block,transB,alpha,Num(1));
        }
      }
      
    }
  }
  printf("  %lu/%lu (%2.2f %%)\n",nsig,nib*njb*nkb,1.e2*(double)nsig/((double)(nib*njb*nkb)));
}

template<typename Num>
void BlockSparseMatrix<Num>::create_block_pixmap(const char* file_name) const{

  FILE* output_file = fopen(file_name,"w");
  fprintf(output_file,"/* XPM */\nstatic char * matrix_xpm[] = {\n\"%i %i 12 1\",\n",(int)_ncolblocks,(int)_nrowblocks);
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

  for(size_t row_block=0;row_block<_nrowblocks;++row_block){
    fprintf(output_file,"\"");
    for(size_t col_block=0;col_block<_ncolblocks;++col_block){
      const auto& block_act = this->block(row_block,col_block);
      if (block_act.size() > 0){//block is significant
        const auto absval = block_act.frobenius_norm();
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
      }else{
        fprintf(output_file,".");
      }
    }
    fprintf(output_file,"\",\n");
  }

  fprintf(output_file,"};");
  fclose(output_file);

}
#endif

