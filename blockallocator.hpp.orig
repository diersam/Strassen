#ifndef BLOCKALLOCATOR_HPP
#define BLOCKALLOCATOR_HPP
#include "blockallocator.h"
#include <cassert>

template<typename T>
T* BlockMemoryPool<T>::allocate(size_t nelements, size_t){
  assert(nelements <= _blocksize);
  T* retval = nullptr;
  #pragma omp critical (blockallocator)
  {
    if(_free_blocks.empty()) this->_allocate_new_block_memory();//need more memory 
    retval = (T*)_free_blocks.back();
    _free_blocks.pop_back();
  }
  return retval;
}

template<typename T>
void BlockMemoryPool<T>::deallocate(void* p, size_t, size_t){
  #pragma omp critical (blockallocator)
  {
    _free_blocks.push_back((byte*)p);
  }
}

template<typename T>
void BlockMemoryPool<T>::_allocate_new_block_memory(){
  constexpr size_t min_alloc_bytes = 16lu*1024lu*1024lu;//allocate at least 16MB to be efficient
  constexpr size_t max_geo_growth = 6lu;//geometric growth up to 6 doublings
  constexpr size_t max_alloc_bytes = min_alloc_bytes<<max_geo_growth;
  static_assert(max_alloc_bytes == 1024lu*1024lu*1024lu);//allocate maximal 1GB so we do not waste too much memory

  //determine new allocation size via geometric growth, but bound by max_alloc_bytes
  const size_t n_new_bytes = min_alloc_bytes<<std::min(max_geo_growth,_mem_pool.size());
  const size_t n_new_blocks = std::max(1lu,n_new_bytes/(_blocksize*sizeof(T)));

  //get more memory
  _mem_pool.emplace_back(std::make_unique_for_overwrite<byte[]>(n_new_blocks*_blocksize*sizeof(T)));
  //_mem_pool.emplace_back(std::unique_ptr<byte[]>(new byte[n_new_blocks*_blocksize*sizeof(T)]));

  //partition new memory
  byte* ptr = _mem_pool.back().get();
  for(size_t i = 0; i< n_new_blocks;++i){
    _free_blocks.push_back(ptr);
    ptr += _blocksize*sizeof(T);
  }
  
}

template<typename T>
void BlockMemoryPool<T>::adjust_blocksize(const size_t max_blocksize){
  assert(_mem_pool.empty());//only works if no blocks are allocated yet
  assert(_free_blocks.empty());
  _blocksize = max_blocksize;
}

template<typename T>
void BlockMemoryPool<T>::release_all_memory(){
  _mem_pool.clear();
  _free_blocks.clear();
}
#endif

