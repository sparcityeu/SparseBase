//
// Created by Amro on 3/31/2022.
//

#include "sparsebase/format/format.h"
#ifndef SPARSEBASE_PROJECT_CONTEXT_CUH
#define SPARSEBASE_PROJECT_CONTEXT_CUH

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
#endif // SPARSEBASE_PROJECT_CONTEXT_CUH
