/*******************************************************
 * Copyright (c) 2022 SparCity, Amro Alabsi Aljundi, Taha Atahan Akyildiz, Arda
 *Sener All rights reserved.
 *
 * This file is distributed under MIT license.
 * The complete license agreement can be obtained at:
 * https://sparcityeu.github.io/sparsebase/pages/license.html
 ********************************************************/
#include "sparsebase/context/context.h"
#ifndef SPARSEBASE_SPARSEBASE_CONTEXT_CUDA_CONTEXT_H_
#define SPARSEBASE_SPARSEBASE_CONTEXT_CUDA_CONTEXT_H_

namespace sparsebase::context {

struct CUDAContext : utils::IdentifiableImplementation<CUDAContext, Context> {
  int device_id;
  CUDAContext(int did);
  virtual bool IsEquivalent(Context *) const;
};
}  // namespace sparsebase::context
#ifdef _HEADER_ONLY
#include "sparsebase/context/cuda_context_cuda.cu"
#endif
#endif  // SPARSEBASE_SPARSEBASE_CONTEXT_CUDA_CONTEXT_H_
