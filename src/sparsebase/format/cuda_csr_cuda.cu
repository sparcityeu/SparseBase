#include <iostream>

#include "sparsebase/format/cuda_csr_cuda.cuh"
#include "sparsebase/format/format.h"
#include "sparsebase/format/format_order_one.h"
#include "sparsebase/format/format_order_two.h"
#include "sparsebase/utils/exception.h"
#include "sparsebase/utils/utils_cuda.cuh"

namespace sparsebase::format {

template <typename IDType, typename NNZType, typename ValueType>
CUDACSR<IDType, NNZType, ValueType>::CUDACSR(
    CUDACSR<IDType, NNZType, ValueType> &&rhs)
    : col_(std::move(rhs.col_)),
      row_ptr_(std::move(rhs.row_ptr_)),
      vals_(std::move(rhs.vals_)) {
  this->nnz_ = rhs.get_num_nnz();
  this->order_ = 2;
  this->dimension_ = rhs.dimension_;
  rhs.col_ = std::unique_ptr<IDType, std::function<void(IDType *)>>(
      nullptr, BlankDeleter<IDType>());
  rhs.row_ptr_ = std::unique_ptr<NNZType, std::function<void(NNZType *)>>(
      nullptr, BlankDeleter<NNZType>());
  rhs.vals_ = std::unique_ptr<ValueType, std::function<void(ValueType *)>>(
      nullptr, BlankDeleter<ValueType>());
  this->context_ = std::unique_ptr<sparsebase::context::Context>(
      new sparsebase::context::CUDAContext(
          rhs.get_cuda_context()->device_id));
}
template <typename IDType, typename NNZType, typename ValueType>
CUDACSR<IDType, NNZType, ValueType>
    &CUDACSR<IDType, NNZType, ValueType>::operator=(
        const CUDACSR<IDType, NNZType, ValueType> &rhs) {
  this->nnz_ = rhs.nnz_;
  this->order_ = 2;
  this->dimension_ = rhs.dimension_;
  IDType *col;
  NNZType *row_ptr;
  context::CUDAContext *gpu_context =
      static_cast<context::CUDAContext *>(this->get_cuda_context());
  cudaSetDevice(gpu_context->device_id);
  cudaMalloc(&col, rhs.get_num_nnz() * sizeof(IDType));
  cudaMemcpy(col, rhs.get_col(), rhs.get_num_nnz() * sizeof(IDType),
             cudaMemcpyDeviceToDevice);
  cudaMalloc(&row_ptr, (rhs.get_dimensions()[0] + 1) * sizeof(NNZType));
  cudaMemcpy(row_ptr, rhs.get_row_ptr(),
             (rhs.get_dimensions()[0] + 1) * sizeof(NNZType),
             cudaMemcpyDeviceToDevice);
  ValueType *vals = nullptr;
  if (rhs.get_vals() != nullptr) {
    if constexpr (std::is_same_v<ValueType, void>) {
      throw utils::TypeException("Cannot create values array for type void");
    } else {
      cudaMalloc(&vals, rhs.get_num_nnz() * sizeof(ValueType));
      cudaMemcpy(vals, rhs.get_vals(), rhs.get_num_nnz() * sizeof(ValueType),
                 cudaMemcpyDeviceToDevice);
    }
  }
  this->col_ = std::unique_ptr<IDType, std::function<void(IDType *)>>(
      col, utils::CUDADeleter<IDType>());
  this->row_ptr_ = std::unique_ptr<NNZType, std::function<void(NNZType *)>>(
      row_ptr, utils::CUDADeleter<NNZType>());
  this->vals_ = std::unique_ptr<ValueType, std::function<void(ValueType *)>>(
      vals, utils::CUDADeleter<ValueType>());
  return *this;
}
template <typename IDType, typename NNZType, typename ValueType>
CUDACSR<IDType, NNZType, ValueType>::CUDACSR(
    const CUDACSR<IDType, NNZType, ValueType> &rhs)
    : col_(nullptr, BlankDeleter<IDType>()),
      row_ptr_(nullptr, BlankDeleter<NNZType>()),
      vals_(nullptr, BlankDeleter<ValueType>()) {
  this->nnz_ = rhs.nnz_;
  this->order_ = 2;
  this->dimension_ = rhs.dimension_;
  IDType *col;
  NNZType *row_ptr;
  context::CUDAContext *gpu_context =
      static_cast<context::CUDAContext *>(this->get_context());
  cudaSetDevice(gpu_context->device_id);
  cudaMalloc(&col, rhs.get_num_nnz() * sizeof(IDType));
  cudaMemcpy(col, rhs.get_col(), rhs.get_num_nnz() * sizeof(IDType),
             cudaMemcpyDeviceToDevice);
  cudaMalloc(&row_ptr, rhs.get_dimensions()[0] * sizeof(NNZType));
  cudaMemcpy(row_ptr, rhs.get_row_ptr(),
             (rhs.get_dimensions()[0] + 1) * sizeof(NNZType),
             cudaMemcpyDeviceToDevice);
  ValueType *vals = nullptr;
  if (rhs.get_vals() != nullptr) {
    if constexpr (std::is_same_v<ValueType, void>) {
      throw utils::TypeException("Cannot create values array for type void");
    } else {
      cudaMalloc(&vals, rhs.get_num_nnz() * sizeof(ValueType));
      cudaMemcpy(vals, rhs.get_vals(), rhs.get_num_nnz() * sizeof(ValueType),
                 cudaMemcpyDeviceToDevice);
    }
  }
  this->col_ = std::unique_ptr<IDType, std::function<void(IDType *)>>(
      col, utils::CUDADeleter<IDType>());
  this->row_ptr_ = std::unique_ptr<NNZType, std::function<void(NNZType *)>>(
      row_ptr, utils::CUDADeleter<NNZType>());
  this->vals_ = std::unique_ptr<ValueType, std::function<void(ValueType *)>>(
      vals, utils::CUDADeleter<ValueType>());
}
template <typename IDType, typename NNZType, typename ValueType>
CUDACSR<IDType, NNZType, ValueType>::CUDACSR(IDType n, IDType m, NNZType nnz,
                                             NNZType *row_ptr, IDType *col,
                                             ValueType *vals,
                                             context::CUDAContext context,
                                             Ownership own)
    : row_ptr_(row_ptr, BlankDeleter<NNZType>()),
      col_(col, BlankDeleter<IDType>()),
      vals_(vals, BlankDeleter<ValueType>()) {
  this->order_ = 2;
  this->dimension_ = {(DimensionType)n, (DimensionType)m};
  this->nnz_ = nnz;
  if (own == kOwned) {
    this->row_ptr_ = std::unique_ptr<NNZType, std::function<void(NNZType *)>>(
        row_ptr, utils::CUDADeleter<NNZType>());
    this->col_ = std::unique_ptr<IDType, std::function<void(IDType *)>>(
        col, utils::CUDADeleter<IDType>());
    this->vals_ = std::unique_ptr<ValueType, std::function<void(ValueType *)>>(
        vals, utils::CUDADeleter<ValueType>());
  }
  this->context_ = std::unique_ptr<sparsebase::context::Context>(
      new sparsebase::context::CUDAContext(context));
}

template <typename IDType, typename NNZType, typename ValueType>
context::CUDAContext *
CUDACSR<IDType, NNZType, ValueType>::get_cuda_context() const {
  return static_cast<context::CUDAContext *>(this->get_context());
}
template <typename IDType, typename NNZType, typename ValueType>
Format *CUDACSR<IDType, NNZType, ValueType>::Clone() const {
  return new CUDACSR(*this);
}
template <typename IDType, typename NNZType, typename ValueType>
IDType *CUDACSR<IDType, NNZType, ValueType>::get_col() const {
  return col_.get();
}
template <typename IDType, typename NNZType, typename ValueType>
NNZType *CUDACSR<IDType, NNZType, ValueType>::get_row_ptr() const {
  return row_ptr_.get();
}
template <typename IDType, typename NNZType, typename ValueType>
ValueType *CUDACSR<IDType, NNZType, ValueType>::get_vals() const {
  return vals_.get();
}
template <typename IDType, typename NNZType, typename ValueType>
IDType *CUDACSR<IDType, NNZType, ValueType>::release_col() {
  auto col = col_.release();
  this->col_ = std::unique_ptr<IDType, std::function<void(IDType *)>>(
      col, BlankDeleter<IDType>());
  return col;
}
template <typename IDType, typename NNZType, typename ValueType>
NNZType *CUDACSR<IDType, NNZType, ValueType>::release_row_ptr() {
  auto row_ptr = row_ptr_.release();
  this->row_ptr_ = std::unique_ptr<NNZType, std::function<void(NNZType *)>>(
      row_ptr, BlankDeleter<NNZType>());
  return row_ptr;
}
template <typename IDType, typename NNZType, typename ValueType>
ValueType *CUDACSR<IDType, NNZType, ValueType>::release_vals() {
  auto vals = vals_.release();
  this->vals_ = std::unique_ptr<ValueType, std::function<void(ValueType *)>>(
      vals, BlankDeleter<ValueType>());
  return vals;
}

template <typename IDType, typename NNZType, typename ValueType>
void CUDACSR<IDType, NNZType, ValueType>::set_col(
    IDType *col, context::CUDAContext context, Ownership own) {
  if (own == kOwned) {
    this->col_ = std::unique_ptr<IDType, std::function<void(IDType *)>>(
        col, utils::CUDADeleter<IDType>());
  } else {
    this->col_ = std::unique_ptr<IDType, std::function<void(IDType *)>>(
        col, BlankDeleter<IDType>());
  }
}

template <typename IDType, typename NNZType, typename ValueType>
void CUDACSR<IDType, NNZType, ValueType>::set_row_ptr(
    NNZType *row_ptr, context::CUDAContext context, Ownership own) {
  if (own == kOwned) {
    this->row_ptr_ = std::unique_ptr<NNZType, std::function<void(NNZType *)>>(
        row_ptr, utils::CUDADeleter<NNZType>());
  } else {
    this->row_ptr_ = std::unique_ptr<NNZType, std::function<void(NNZType *)>>(
        row_ptr, BlankDeleter<NNZType>());
  }
}

template <typename IDType, typename NNZType, typename ValueType>
void CUDACSR<IDType, NNZType, ValueType>::set_vals(
    ValueType *vals, context::CUDAContext context, Ownership own) {
  if (own == kOwned) {
    this->vals_ = std::unique_ptr<ValueType, std::function<void(ValueType *)>>(
        vals, utils::CUDADeleter<ValueType>());
  } else {
    this->vals_ = std::unique_ptr<ValueType, std::function<void(ValueType *)>>(
        vals, BlankDeleter<ValueType>());
  }
}

template <typename IDType, typename NNZType, typename ValueType>
bool CUDACSR<IDType, NNZType, ValueType>::RowPtrIsOwned() {
  return (this->row_ptr_.get_deleter().target_type() !=
          typeid(BlankDeleter<NNZType>));
}

template <typename IDType, typename NNZType, typename ValueType>
bool CUDACSR<IDType, NNZType, ValueType>::ColIsOwned() {
  return (this->col_.get_deleter().target_type() !=
          typeid(BlankDeleter<IDType>));
}

template <typename IDType, typename NNZType, typename ValueType>
bool CUDACSR<IDType, NNZType, ValueType>::ValsIsOwned() {
  return (this->vals_.get_deleter().target_type() !=
          typeid(BlankDeleter<ValueType>));
}
template <typename IDType, typename NNZType, typename ValueType>
CUDACSR<IDType, NNZType, ValueType>::~CUDACSR() {}

#ifndef _HEADER_ONLY
#include "init/cuda/cuda_csr_cuda.inc"
#endif
}  // namespace cuda