#include "sparsebase/format/cuda_array_cuda.cuh"
#include "sparsebase/utils/utils_cuda.cuh"

namespace sparsebase::format {

template <typename ValueType>
CUDAArray<ValueType>::CUDAArray(CUDAArray<ValueType> &&rhs)
    : vals_(std::move(rhs.vals_)) {
  static_assert(!std::is_same_v<ValueType, void>,
                "Cannot create CUDAArray with void ValueType");
  this->nnz_ = rhs.get_num_nnz();
  this->order_ = 1;
  this->dimension_ = rhs.dimension_;
  rhs.vals_ = std::unique_ptr<ValueType, std::function<void(ValueType *)>>(
      nullptr, BlankDeleter<ValueType>());
  this->context_ = std::unique_ptr<sparsebase::context::Context>(
      new sparsebase::context::CPUContext);
}
template <typename ValueType>
CUDAArray<ValueType> &CUDAArray<ValueType>::operator=(
    const CUDAArray<ValueType> &rhs) {
  static_assert(!std::is_same_v<ValueType, void>,
                "Cannot create CUDAArray with void ValueType");
  this->nnz_ = rhs.nnz_;
  this->order_ = 1;
  this->dimension_ = rhs.dimension_;
  ValueType *vals = nullptr;
  context::CUDAContext *gpu_context =
      static_cast<context::CUDAContext *>(this->get_context());
  if (rhs.get_vals() != nullptr) {
    cudaSetDevice(gpu_context->device_id);
    cudaMalloc(&vals, rhs.get_num_nnz() * sizeof(ValueType));
    cudaMemcpy(vals, rhs.get_vals(), rhs.get_num_nnz() * sizeof(ValueType),
               cudaMemcpyDeviceToDevice);
    vals = new ValueType[rhs.get_num_nnz()];
    std::copy(rhs.get_vals(), rhs.get_vals() + rhs.get_num_nnz(), vals);
  }
  this->vals_ = std::unique_ptr<ValueType, std::function<void(ValueType *)>>(
      vals, utils::CUDADeleter<ValueType>());
  return *this;
}
template <typename ValueType>
CUDAArray<ValueType>::CUDAArray(const CUDAArray<ValueType> &rhs)
    : vals_(nullptr, BlankDeleter<ValueType>()) {
  static_assert(!std::is_same_v<ValueType, void>,
                "Cannot create CUDAArray with void ValueType");
  this->nnz_ = rhs.nnz_;
  this->order_ = 1;
  this->dimension_ = rhs.dimension_;
  ValueType *vals = nullptr;
  if (rhs.get_vals() != nullptr) {
    vals = new ValueType[rhs.get_num_nnz()];
    std::copy(rhs.get_vals(), rhs.get_vals() + rhs.get_num_nnz(), vals);
  }
  this->vals_ = std::unique_ptr<ValueType, std::function<void(ValueType *)>>(
      vals, Deleter<ValueType>());
  this->context_ = std::unique_ptr<sparsebase::context::Context>(
      new sparsebase::context::CPUContext);
}
template <typename ValueType>
CUDAArray<ValueType>::CUDAArray(DimensionType nnz, ValueType *vals,
                                context::CUDAContext context, Ownership own)
    : vals_(vals, BlankDeleter<ValueType>()) {
  static_assert(!std::is_same_v<ValueType, void>,
                "Cannot create CUDAArray with void ValueType");
  this->order_ = 1;
  this->dimension_ = {(DimensionType)nnz};
  this->nnz_ = nnz;
  if (own == kOwned) {
    this->vals_ = std::unique_ptr<ValueType, std::function<void(ValueType *)>>(
        vals, utils::CUDADeleter<ValueType>());
  }
  this->context_ = std::unique_ptr<sparsebase::context::Context>(
      new sparsebase::context::CUDAContext(context));
}

template <typename ValueType>
Format *CUDAArray<ValueType>::Clone() const {
  static_assert(!std::is_same_v<ValueType, void>,
                "Cannot create CUDAArray with void ValueType");
  return new CUDAArray(*this);
}
template <typename ValueType>
ValueType *CUDAArray<ValueType>::get_vals() const {
  static_assert(!std::is_same_v<ValueType, void>,
                "Cannot create CUDAArray with void ValueType");
  return vals_.get();
}
template <typename ValueType>
ValueType *CUDAArray<ValueType>::release_vals() {
  static_assert(!std::is_same_v<ValueType, void>,
                "Cannot create CUDAArray with void ValueType");
  auto vals = vals_.release();
  this->vals_ = std::unique_ptr<ValueType, std::function<void(ValueType *)>>(
      vals, BlankDeleter<ValueType>());
  return vals;
}

template <typename ValueType>
void CUDAArray<ValueType>::set_vals(ValueType *vals, Ownership own) {
  static_assert(!std::is_same_v<ValueType, void>,
                "Cannot create CUDAArray with void ValueType");
  if (own == kOwned) {
    this->vals_ = std::unique_ptr<ValueType, std::function<void(ValueType *)>>(
        vals, Deleter<ValueType>());
  } else {
    this->vals_ = std::unique_ptr<ValueType, std::function<void(ValueType *)>>(
        vals, BlankDeleter<ValueType>());
  }
}

template <typename ValueType>
bool CUDAArray<ValueType>::ValsIsOwned() {
  static_assert(!std::is_same_v<ValueType, void>,
                "Cannot create CUDAArray with void ValueType");
  return (this->vals_.get_deleter().target_type() !=
          typeid(BlankDeleter<ValueType>));
}
template <typename ValueType>
CUDAArray<ValueType>::~CUDAArray() {}
// format.inc
#ifndef _HEADER_ONLY
#include "init/cuda/cuda_array_cuda.inc"
#endif
}  // namespace sparsebase::format