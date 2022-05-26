/*******************************************************
 * Copyright (c) 2022, SparseBase
 * All rights reserved.
 *
 * This file is distributed under MIT license.
 * The complete license agreement can be obtained at:
 * https://github.com/SU-HPC/sparsebase/blob/main/LICENSE
 ********************************************************/
#ifndef SPARSEBASE_SPARSEBASE_CONTEXT_CUDA_CONTEXT_H_
#define SPARSEBASE_SPARSEBASE_CONTEXT_CUDA_CONTEXT_H_

#include "sparsebase/format/format.h"
namespace sparsebase {
namespace context {
namespace cuda {

struct CUDAContext : ContextImplementation<CUDAContext> {
  int device_id;
  CUDAContext(int did);
  virtual bool IsEquivalent(Context *) const;
};
} // namespace cuda
} // namespace context
} // namespace sparsebase
#ifdef _HEADER_ONLY
#include "sparsebase/context/cuda/context.cu"
#endif
#endif // SPARSEBASE_SPARSEBASE_CONTEXT_CUDA_CONTEXT_H_
