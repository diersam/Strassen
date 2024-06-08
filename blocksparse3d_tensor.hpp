#ifndef BLOCKSPARSE3D_TENSOR_H
#define BLOCKSPARSE3D_TENSOR_H

#include <vector>
#include "blocksparsematrix.h"

template<typename Num>
using blocksparse3d_tensor = std::vector<BlockSparseMatrix<Num>>;

#endif

