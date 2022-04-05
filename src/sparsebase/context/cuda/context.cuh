//
// Created by Amro on 3/31/2022.
//

#ifndef SPARSEBASE_SPARSEBASE_CONTEXT_CUDA_CONTEXT_H_
#define SPARSEBASE_SPARSEBASE_CONTEXT_CUDA_CONTEXT_H_

#include "sparsebase/format/format.h"
namespace sparsebase{
namespace context {
namespace cuda{

struct CUDAContext : ContextImplementation<CUDAContext> {
  int device_id;
  CUDAContext(int did);
  virtual bool IsEquivalent(Context *) const;
};
}
}
}
#ifdef _HEADER_ONLY
#include "sparsebase/context/cuda/context.cu"
#endif
#endif // SPARSEBASE_SPARSEBASE_CONTEXT_CUDA_CONTEXT_H_
