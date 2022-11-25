//
// Created by Amro on 3/31/2022.
//
#include "cuda_context_cuda.cuh"
#include "sparsebase/context/context.h"
#include "sparsebase/utils/exception.h"
namespace sparsebase::context {

CUDAContext::CUDAContext(int did) : device_id(did) {
  int device_count;
  cudaGetDeviceCount(&device_count);
  if (device_id >= device_count) {
    throw utils::CUDADeviceException(device_count, device_id);
  }
}
bool CUDAContext::IsEquivalent(Context *rhs) const {
  if (dynamic_cast<CUDAContext *>(rhs) != nullptr) {
    if (dynamic_cast<CUDAContext *>(rhs)->device_id == this->device_id)
      return true;
  }
  return false;
}
}  // namespace sparsebase::context