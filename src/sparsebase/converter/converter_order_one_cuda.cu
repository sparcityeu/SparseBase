#include "sparsebase/converter/converter.h"
#include "sparsebase/converter/converter_order_one_cuda.cuh"
#include "sparsebase/format/array.h"
#include "sparsebase/format/cuda_array_cuda.cuh"
#include "sparsebase/format/format.h"
#include "sparsebase/format/format_order_one.h"

namespace sparsebase::converter {
template <typename ValueType>
format::Format *CUDAArrayArrayConditionalFunction(format::Format *source,
                                                  context::Context *context) {
  context::CUDAContext *gpu_context =
      static_cast<context::CUDAContext *>(source->get_context());
  auto cuda_array = source->AsAbsolute<format::CUDAArray<ValueType>>();
  cudaSetDevice(gpu_context->device_id);
  ValueType *vals = nullptr;
  if (cuda_array->get_vals() != nullptr) {
    vals = new ValueType[cuda_array->get_num_nnz()];
    cudaMemcpy(vals, cuda_array->get_vals(),
               cuda_array->get_num_nnz() * sizeof(ValueType),
               cudaMemcpyDeviceToHost);
  }
  return new format::Array<ValueType>(cuda_array->get_num_nnz(), vals,
                                      format::kOwned);
}
template <typename ValueType>
format::Format *ArrayCUDAArrayConditionalFunction(format::Format *source,
                                                  context::Context *context) {
  context::CUDAContext *gpu_context =
      static_cast<context::CUDAContext *>(context);
  auto array = source->AsAbsolute<format::Array<ValueType>>();
  cudaSetDevice(gpu_context->device_id);
  ValueType *vals = nullptr;
  if (array->get_vals() != nullptr) {
    cudaMalloc(&vals, array->get_num_nnz() * sizeof(ValueType));
    cudaMemcpy(vals, array->get_vals(),
               array->get_num_nnz() * sizeof(ValueType),
               cudaMemcpyHostToDevice);
  }
  return new format::CUDAArray<ValueType>(array->get_num_nnz(), vals,
                                          *gpu_context, format::kOwned);
}
#if !defined(_HEADER_ONLY)
#include "init/cuda/converter_order_one_cuda.inc"
#endif
}  // namespace sparsebase::converter