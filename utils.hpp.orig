#ifndef UTILS_HPP
#define UTILS_HPP
#include <unistd.h>


template<typename Integral1, typename Integral2>
constexpr Integral1 integer_division_round_up(const Integral1& lhs, const Integral2& rhs){
  if (lhs == 0) return 0;
  else{
    return 1+(lhs-1)/rhs;
  }
}

static inline size_t get_file_size(FILE* file_handle){
  const auto current = ftell(file_handle);
  fseek(file_handle,0,SEEK_SET);
  const auto start = ftell(file_handle);
  fseek(file_handle,0,SEEK_END);
  const auto end = ftell(file_handle);
  fseek(file_handle,current,SEEK_SET);
  return (size_t)(end - start);
}

//tensor indexing
//fortran indexing 3D
constexpr size_t ijk(size_t i, size_t j, size_t k, size_t dimi, size_t dimj) {
  return i + j*dimi + k*dimi*dimj;
}
//fortran indexing 2D
constexpr size_t ij(size_t i, size_t j, size_t dimi) {
  return i+j*dimi;
}

static inline void print_stack_trace(){
  char cdebug[1000];
  sprintf(cdebug, "gdb -batch -ex 'bt' -p %d", (int) getpid());
  (void)system(cdebug);
}
#endif

