#ifndef BLOCKALLOCATOR_H
#define BLOCKALLOCATOR_H
#include <vector>
#include <memory>
using byte = uint8_t;//=std::byte;

//memory pool handelling blocks of memory efficiently
template <typename T>
class BlockMemoryPool{
  public:
  constexpr BlockMemoryPool() noexcept = default;
  constexpr BlockMemoryPool(size_t blocksize) noexcept : _blocksize(blocksize){}

  void release_all_memory();
  void adjust_blocksize(size_t max_blocksize);
  T* allocate(size_t bytes, size_t alignement = 0);
  void deallocate(void* p, size_t bytes, size_t alignement = 0);

  private:
  std::vector<std::unique_ptr<byte>> _mem_pool;
  size_t _blocksize = 0;
  std::vector<byte*> _free_blocks;

  void _allocate_new_block_memory();
};

//statefull allocator for blocks of matrices with an handle to an underlying BlockMemoryPool
template <typename T>
class BlockAllocator{
  public:
  using value_type = T;
  BlockAllocator() = delete;// this should remove the possibility of nullptr 
  BlockAllocator(BlockMemoryPool<T>& mem_pool) : _mem_pool(&mem_pool){}
  BlockAllocator(BlockMemoryPool<T>* mem_pool) : _mem_pool(mem_pool){}

  //BlockAllocator(const BlockAllocator&) = default;
  //BlockAllocator& operator=(const BlockAllocator& other) = default;
  //BlockAllocator(BlockAllocator&&) = default;
  //BlockAllocator& operator=(BlockAllocator&& other) = default;
  //~BlockAllocator() = default;

  T* allocate(size_t bytes, size_t alignement = 0) & {
    assert(_mem_pool != nullptr);
    return _mem_pool->allocate(bytes);}
  void deallocate(void* p, size_t bytes, size_t alignement = 0) & {
    assert(_mem_pool != nullptr);
    _mem_pool->deallocate(p,bytes);
  }
  constexpr bool is_equal(const BlockAllocator& other) const {return _mem_pool == other._mem_pool;}

  private:
  BlockMemoryPool<T>* _mem_pool = nullptr;
};

template<typename T1,typename T2>
constexpr bool operator==(const BlockAllocator<T1>& alloc1, const BlockAllocator<T2>& alloc2){return alloc1.is_equal(alloc2);}
template<typename T1,typename T2>
constexpr bool operator!=(const BlockAllocator<T1>& alloc1, const BlockAllocator<T2>& alloc2){return !alloc1.is_equal(alloc2);}

#endif

