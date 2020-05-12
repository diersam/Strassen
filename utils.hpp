#ifndef UTILS_HPP
#define UTILS_HPP

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

#endif

