#include "sparsebase/converter/converter.h"
#include "sparsebase/converter/converter_order_two.h"
#include "sparsebase/converter/converter_order_two_cuda.cuh"
#include "sparsebase/format/cuda_csr_cuda.cuh"
#include "sparsebase/format/format.h"
#include "sparsebase/format/format_order_one.h"
#include "sparsebase/format/format_order_two.h"

namespace sparsebase::converter {
template <typename IDType, typename NNZType, typename ValueType>
format::Format *CsrCUDACsrConditionalFunction(format::Format *source,
                                      context::Context *context) {
  context::CUDAContext *gpu_context =
      static_cast<context::CUDAContext *>(context);
  auto csr = source->AsAbsolute<format::CSR<IDType, NNZType, ValueType>>();
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
    if constexpr (std::is_same_v<ValueType, void>) {
      throw utils::TypeException("Cannot create values array for type void");
    } else {
      cudaMalloc(&vals, csr->get_num_nnz() * sizeof(ValueType));
      cudaMemcpy(vals, csr->get_vals(), csr->get_num_nnz() * sizeof(ValueType),
                 cudaMemcpyHostToDevice);
    }
  }
  return new format::CUDACSR<IDType, NNZType, ValueType>(
      csr->get_dimensions()[0], csr->get_dimensions()[0], csr->get_num_nnz(),
      row_ptr, col, vals, *gpu_context);
}
template <typename IDType, typename NNZType, typename ValueType>
format::Format *CUDACsrCUDACsrConditionalFunction(format::Format *source,
                                          context::Context *context) {
  context::CUDAContext *dest_gpu_context =
      static_cast<context::CUDAContext *>(context);
  auto cuda_csr =
      source->AsAbsolute<format::CUDACSR<IDType, NNZType, ValueType>>();
  context::CUDAContext *source_gpu_context =
      static_cast<context::CUDAContext *>(cuda_csr->get_context());
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
    if constexpr (std::is_same_v<ValueType, void>) {
      throw utils::TypeException("Cannot create values array for type void");
    } else {
      cudaMalloc(&vals, cuda_csr->get_num_nnz() * sizeof(ValueType));
      cudaMemcpy(vals, cuda_csr->get_vals(),
                 cuda_csr->get_num_nnz() * sizeof(ValueType),
                 cudaMemcpyDeviceToDevice);
    }
  }
  return new format::CUDACSR<IDType, NNZType, ValueType>(
      cuda_csr->get_dimensions()[0], cuda_csr->get_dimensions()[0],
      cuda_csr->get_num_nnz(), row_ptr, col, vals, *dest_gpu_context);
}
template <typename IDType, typename NNZType, typename ValueType>
format::Format *CUDACsrCsrConditionalFunction(format::Format *source,
                                      context::Context *context) {
  context::CUDAContext *gpu_context =
      static_cast<context::CUDAContext *>(source->get_context());
  auto cuda_csr =
      source->AsAbsolute<format::CUDACSR<IDType, NNZType, ValueType>>();
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
    if constexpr (std::is_same_v<ValueType, void>) {
      throw utils::TypeException("Cannot create values array for type void");
    } else {
      vals = new ValueType[nnz];
      cudaMemcpy(vals, cuda_csr->get_vals(), nnz * sizeof(ValueType),
                 cudaMemcpyDeviceToHost);
    }
  }
  return new format::CSR<IDType, NNZType, ValueType>(n, n, row_ptr, col, vals);
}
#if !defined(_HEADER_ONLY)
#include "init/cuda/converter_order_two_cuda.inc"
#endif
}