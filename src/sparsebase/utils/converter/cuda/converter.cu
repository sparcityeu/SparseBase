#include "sparsebase/context/context.h"
#include "sparsebase/context/cuda/context.cuh"
#include "sparsebase/format/cuda/format.cuh"
#include "sparsebase/format/format.h"
#include "sparsebase/utils/converter/converter.h"

using namespace sparsebase;
using namespace sparsebase::format;
namespace sparsebase {
namespace utils {
namespace converter {
namespace cuda {

template <typename ValueType>
Format *CUDAArrayArrayConditionalFunction(Format *source,
                                          context::Context *context) {
  context::cuda::CUDAContext *gpu_context =
      static_cast<context::cuda::CUDAContext *>(source->get_context());
  auto cuda_array = source->As<format::cuda::CUDAArray<ValueType>>();
  cudaSetDevice(gpu_context->device_id);
  ValueType *vals = nullptr;
  if (cuda_array->get_vals() != nullptr) {
    vals = new ValueType[cuda_array->get_num_nnz()];
    cudaMemcpy(vals, cuda_array->get_vals(),
               cuda_array->get_num_nnz() * sizeof(ValueType),
               cudaMemcpyDeviceToHost);
  }
  return new Array<ValueType>(cuda_array->get_num_nnz(), vals);
}
template <typename ValueType>
Format *ArrayCUDAArrayConditionalFunction(Format *source,
                                          context::Context *context) {
  context::cuda::CUDAContext *gpu_context =
      static_cast<context::cuda::CUDAContext *>(context);
  auto array = source->As<Array<ValueType>>();
  cudaSetDevice(gpu_context->device_id);
  ValueType *vals = nullptr;
  if (array->get_vals() != nullptr) {
    cudaMalloc(&vals, array->get_num_nnz() * sizeof(ValueType));
    cudaMemcpy(vals, array->get_vals(),
               array->get_num_nnz() * sizeof(ValueType),
               cudaMemcpyHostToDevice);
  }
  return new format::cuda::CUDAArray<ValueType>(array->get_num_nnz(), vals,
                                                *gpu_context);
}
template <typename IDType, typename NNZType, typename ValueType>
Format *CsrCUDACsrConditionalFunction(Format *source,
                                      context::Context *context) {
  context::cuda::CUDAContext *gpu_context =
      static_cast<context::cuda::CUDAContext *>(context);
  auto csr = source->As<CSR<IDType, NNZType, ValueType>>();
  cudaSetDevice(gpu_context->device_id);
  NNZType *row_ptr;
  IDType *col;
  ValueType *vals = nullptr;
  cudaMalloc(&row_ptr, (1 + csr->get_dimensions()[0]) * sizeof(NNZType));
  cudaMemcpy(row_ptr, csr->get_row_ptr(),
             (1 + csr->get_dimensions()[0]) * sizeof(NNZType),
             cudaMemcpyHostToDevice);
  cudaMalloc(&col, csr->get_num_nnz() * sizeof(IDType));
  cudaMemcpy(col, csr->get_col(), csr->get_num_nnz() * sizeof(IDType),
             cudaMemcpyHostToDevice);
  if (csr->get_vals() != nullptr) {
    cudaMalloc(&vals, csr->get_num_nnz() * sizeof(ValueType));
    cudaMemcpy(vals, csr->get_vals(), csr->get_num_nnz() * sizeof(ValueType),
               cudaMemcpyHostToDevice);
  }
  return new format::cuda::CUDACSR<IDType, NNZType, ValueType>(
      csr->get_dimensions()[0], csr->get_dimensions()[0], csr->get_num_nnz(),
      row_ptr, col, vals, *gpu_context);
}
template <typename IDType, typename NNZType, typename ValueType>
Format *CUDACsrCUDACsrConditionalFunction(Format *source,
                                          context::Context *context) {
  context::cuda::CUDAContext *dest_gpu_context =
      static_cast<context::cuda::CUDAContext *>(context);
  auto cuda_csr =
      source->As<format::cuda::CUDACSR<IDType, NNZType, ValueType>>();
  context::cuda::CUDAContext *source_gpu_context =
      static_cast<context::cuda::CUDAContext *>(cuda_csr->get_context());
  cudaSetDevice(dest_gpu_context->device_id);
  cudaDeviceEnablePeerAccess(source_gpu_context->device_id, 0);
  NNZType *row_ptr;
  IDType *col;
  ValueType *vals = nullptr;
  cudaMalloc(&row_ptr, cuda_csr->get_dimensions()[0] * sizeof(NNZType));
  cudaMemcpy(row_ptr, cuda_csr->get_row_ptr(),
             (1 + cuda_csr->get_dimensions()[0]) * sizeof(NNZType),
             cudaMemcpyDeviceToDevice);
  cudaMalloc(&col, cuda_csr->get_num_nnz() * sizeof(IDType));
  cudaMemcpy(col, cuda_csr->get_col(), cuda_csr->get_num_nnz() * sizeof(IDType),
             cudaMemcpyDeviceToDevice);
  if (cuda_csr->get_vals() != nullptr) {
    cudaMalloc(&vals, cuda_csr->get_num_nnz() * sizeof(ValueType));
    cudaMemcpy(vals, cuda_csr->get_vals(),
               cuda_csr->get_num_nnz() * sizeof(ValueType),
               cudaMemcpyDeviceToDevice);
  }
  return new format::cuda::CUDACSR<IDType, NNZType, ValueType>(
      cuda_csr->get_dimensions()[0], cuda_csr->get_dimensions()[0],
      cuda_csr->get_num_nnz(), row_ptr, col, vals, *dest_gpu_context);
}
template <typename IDType, typename NNZType, typename ValueType>
Format *CUDACsrCsrConditionalFunction(Format *source,
                                      context::Context *context) {
  context::cuda::CUDAContext *gpu_context =
      static_cast<context::cuda::CUDAContext *>(source->get_context());
  auto cuda_csr =
      source->As<format::cuda::CUDACSR<IDType, NNZType, ValueType>>();
  cudaSetDevice(gpu_context->device_id);
  int n = cuda_csr->get_dimensions()[0];
  int nnz = cuda_csr->get_num_nnz();
  NNZType *row_ptr = new NNZType[n + 1];
  IDType *col = new IDType[nnz];
  ValueType *vals = nullptr;
  cudaMemcpy(row_ptr, cuda_csr->get_row_ptr(), (n + 1) * sizeof(NNZType),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(col, cuda_csr->get_col(), nnz * sizeof(IDType),
             cudaMemcpyDeviceToHost);
  if (cuda_csr->get_vals() != nullptr) {
    vals = new ValueType[nnz];
    cudaMemcpy(vals, cuda_csr->get_vals(), nnz * sizeof(ValueType),
               cudaMemcpyDeviceToHost);
  }
  return new CSR<IDType, NNZType, ValueType>(n, n, row_ptr, col, vals);
}

bool CUDAPeerToPeer(context::Context *from, context::Context *to) {
  auto from_gpu = static_cast<context::cuda::CUDAContext *>(from);
  auto to_gpu = static_cast<context::cuda::CUDAContext *>(to);
  int can_access;
  cudaDeviceCanAccessPeer(&can_access, from_gpu->device_id, to_gpu->device_id);
  return can_access;
}
#if !defined(_HEADER_ONLY)
#include "init/cuda/converter.inc"
#endif
} // namespace cuda
} // namespace converter
} // namespace utils
} // namespace sparsebase
