#include "sparsebase/format/array.h"
#include "sparsebase/utils/logger.h"
namespace sparsebase::format {

template <typename ValueType>
Array<ValueType>::Array(Array<ValueType> &&rhs) : vals_(std::move(rhs.vals_)) {
  static_assert(!std::is_same_v<ValueType, void>,
                "A format::Array cannot contain have ValueType as void");
  this->nnz_ = rhs.get_num_nnz();
  this->order_ = 1;
  this->dimension_ = rhs.dimension_;
  rhs.vals_ = std::unique_ptr<ValueType, std::function<void(ValueType *)>>(
      nullptr, BlankDeleter<ValueType>());
  this->context_ = std::unique_ptr<sparsebase::context::Context>(
      new sparsebase::context::CPUContext);
}
template <typename ValueType>
Array<ValueType> &Array<ValueType>::operator=(const Array<ValueType> &rhs) {
  static_assert(!std::is_same_v<ValueType, void>,
                "A format::Array cannot contain have ValueType as void");
  this->nnz_ = rhs.nnz_;
  this->order_ = 1;
  this->dimension_ = rhs.dimension_;
  ValueType *vals = nullptr;
  if constexpr (!std::is_same_v<ValueType, void>) {
    if (rhs.get_vals() != nullptr) {
      vals = new ValueType[rhs.get_num_nnz()];
      std::copy(rhs.get_vals(), rhs.get_vals() + rhs.get_num_nnz(), vals);
    }
    this->vals_ = std::unique_ptr<ValueType, std::function<void(ValueType *)>>(
        vals, Deleter<ValueType>());
    return *this;
  } else {
    throw utils::TypeException("Cannot create an array with ValueType == void");
  }
}
template <typename ValueType>
Array<ValueType>::Array(const Array<ValueType> &rhs)
    : vals_(nullptr, BlankDeleter<ValueType>()) {
  static_assert(!std::is_same_v<ValueType, void>,
                "A format::Array cannot contain have ValueType as void");
  this->nnz_ = rhs.nnz_;
  this->order_ = 1;
  this->dimension_ = rhs.dimension_;
  ValueType *vals = nullptr;
  if constexpr (!std::is_same_v<ValueType, void>) {
    if (rhs.get_vals() != nullptr) {
      vals = new ValueType[rhs.get_num_nnz()];
      std::copy(rhs.get_vals(), rhs.get_vals() + rhs.get_num_nnz(), vals);
    }
    this->vals_ = std::unique_ptr<ValueType, std::function<void(ValueType *)>>(
        vals, Deleter<ValueType>());
    this->context_ = std::unique_ptr<sparsebase::context::Context>(
        new sparsebase::context::CPUContext);
  } else {
    throw utils::TypeException("Cannot create an array with ValueType == void");
  }
}
template <typename ValueType>
Array<ValueType>::Array(DimensionType nnz, ValueType *vals, Ownership own)
    : vals_(vals, BlankDeleter<ValueType>()) {
  static_assert(!std::is_same_v<ValueType, void>,
                "A format::Array cannot contain have ValueType as void");
  this->order_ = 1;
  this->dimension_ = {(DimensionType)nnz};
  this->nnz_ = nnz;
  if (own == kOwned) {
    this->vals_ = std::unique_ptr<ValueType, std::function<void(ValueType *)>>(
        vals, Deleter<ValueType>());
  }
  this->context_ = std::unique_ptr<sparsebase::context::Context>(
      new sparsebase::context::CPUContext);
}

template <typename ValueType>
Format *Array<ValueType>::Clone() const {
  static_assert(!std::is_same_v<ValueType, void>,
                "A format::Array cannot contain have ValueType as void");
  return new Array(*this);
}
template <typename ValueType>
ValueType *Array<ValueType>::get_vals() const {
  static_assert(!std::is_same_v<ValueType, void>,
                "A format::Array cannot contain have ValueType as void");
  return vals_.get();
}
template <typename ValueType>
ValueType *Array<ValueType>::release_vals() {
  static_assert(!std::is_same_v<ValueType, void>,
                "A format::Array cannot contain have ValueType as void");
  auto vals = vals_.release();
  this->vals_ = std::unique_ptr<ValueType, std::function<void(ValueType *)>>(
      vals, BlankDeleter<ValueType>());
  return vals;
}

template <typename ValueType>
void Array<ValueType>::set_vals(ValueType *vals, Ownership own) {
  static_assert(!std::is_same_v<ValueType, void>,
                "A format::Array cannot contain have ValueType as void");
  if (own == kOwned) {
    this->vals_ = std::unique_ptr<ValueType, std::function<void(ValueType *)>>(
        vals, Deleter<ValueType>());
  } else {
    this->vals_ = std::unique_ptr<ValueType, std::function<void(ValueType *)>>(
        vals, BlankDeleter<ValueType>());
  }
}

template <typename ValueType>
bool Array<ValueType>::ValsIsOwned() {
  static_assert(!std::is_same_v<ValueType, void>,
                "A format::Array cannot contain have ValueType as void");
  return (this->vals_.get_deleter().target_type() !=
          typeid(BlankDeleter<ValueType>));
}
template <typename ValueType>
Array<ValueType>::~Array() {
  static_assert(!std::is_same_v<ValueType, void>,
                "A format::Array cannot contain have ValueType as void");
}
#ifndef _HEADER_ONLY
#include "init/array.inc"
#endif
}
