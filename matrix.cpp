#include "matrix.h"
#include "matrix.hpp"

//generic print function
template<>
void print<double>(const double* mat_ptr, const size_t nrow, const size_t ncol, 
    const char* name, const char* format, size_t n_per_row){
  puts(name);
  for (size_t bcol=0;bcol<ncol;bcol+=n_per_row){
    const size_t start = bcol;
    const size_t end = std::min(start + n_per_row, ncol);
    for (size_t col=start;col<end;++col){
      printf("        %4lu",col+1);//column header
    }
    puts("");
    for (size_t row=0;row<nrow;++row){
      for (size_t col=start;col<end;++col){
        printf(format, mat_ptr[row + nrow*col]);
      }
      puts("");
    }
    puts("");
  }
  puts("");
}

