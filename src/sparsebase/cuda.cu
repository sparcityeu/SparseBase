#include <iostream>
#include "sparse_format.h"

namespace sparsebase {

namespace format {
  template <typename T> struct CUDADeleter {
    void operator()(T *obj) {
      cudaFree(obj);
    }
  };

  template <typename IDType, typename NNZType, typename ValueType>
  class CUDACSR : public FormatImplementation<CUDACSR<IDType, NNZType, ValueType>> {
    CUDACSR(IDType n, IDType m, NNZType *row_ptr, IDType *col, ValueType *vals, context::CUDAContext context,
        Ownership own = kNotOwned);
    CUDACSR(const CUDACSR<IDType, NNZType, ValueType> &);
    CUDACSR(CUDACSR<IDType, NNZType, ValueType> &&);
    CUDACSR<IDType, NNZType, ValueType> &
    operator=(const CUDACSR<IDType, NNZType, ValueType> &);
    Format *clone() const override;
    virtual ~CUDACSR();
    NNZType *get_row_ptr() const;
    IDType *get_col() const;
    ValueType *get_vals() const;

    NNZType *release_row_ptr();
    IDType *release_col();
    ValueType *release_vals();

    void set_row_ptr(NNZType *, context::CUDAContext context, Ownership own = kNotOwned);
    void set_col(IDType *, context::CUDAContext context, Ownership own = kNotOwned);
    void set_vals(ValueType *, context::CUDAContext context, Ownership own = kNotOwned);

    virtual bool ColIsOwned();
    virtual bool RowPtrIsOwned();
    virtual bool ValsIsOwned();

  protected:
    std::unique_ptr<NNZType, std::function<void(NNZType *)>> row_ptr_;
    std::unique_ptr<IDType, std::function<void(IDType *)>> col_;
    std::unique_ptr<ValueType, std::function<void(ValueType *)>> vals_;

  };


template <typename IDType, typename NNZType, typename ValueType>
CUDACSR<IDType, NNZType, ValueType>::CUDACSR(CUDACSR<IDType, NNZType, ValueType> &&rhs)
    : col_(std::move(rhs.col_)), row_ptr_(std::move(rhs.row_ptr_)),
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
  this->context_ = std::unique_ptr<sparsebase::context::Context>(new sparsebase::context::CUDAContext);
}
template <typename IDType, typename NNZType, typename ValueType>
CUDACSR<IDType, NNZType, ValueType> &CUDACSR<IDType, NNZType, ValueType>::operator=(
    const CUDACSR<IDType, NNZType, ValueType> &rhs) {
  this->nnz_ = rhs.nnz_;
  this->order_ = 2;
  this->dimension_ = rhs.dimension_;
  auto col = new IDType[rhs.get_num_nnz()];
  std::copy(rhs.get_col(), rhs.get_col() + rhs.get_num_nnz(), col);
  auto row_ptr = new NNZType[rhs.get_num_nnz()];
  std::copy(rhs.get_row_ptr(), rhs.get_row_ptr() + rhs.get_num_nnz(), row_ptr);
  ValueType *vals = nullptr;
  if (rhs.get_vals() != nullptr) {
    vals = new ValueType[rhs.get_num_nnz()];
    std::copy(rhs.get_vals(), rhs.get_vals() + rhs.get_num_nnz(), vals);
  }
  this->col_ = std::unique_ptr<IDType, std::function<void(IDType *)>>(
      col, CUDADeleter<IDType>());
  this->row_ptr_ = std::unique_ptr<NNZType, std::function<void(NNZType *)>>(
      row_ptr, CUDADeleter<NNZType>());
  this->vals_ = std::unique_ptr<ValueType, std::function<void(ValueType *)>>(
      vals, CUDADeleter<ValueType>());
  return *this;
}
template <typename IDType, typename NNZType, typename ValueType>
CUDACSR<IDType, NNZType, ValueType>::CUDACSR(const CUDACSR<IDType, NNZType, ValueType> &rhs)
    : col_(nullptr, BlankDeleter<IDType>()),
      row_ptr_(nullptr, BlankDeleter<NNZType>()),
      vals_(nullptr, BlankDeleter<ValueType>()) {
  this->nnz_ = rhs.nnz_;
  this->order_ = 2;
  this->dimension_ = rhs.dimension_;
  auto col = new IDType[rhs.get_num_nnz()];
  std::copy(rhs.get_col(), rhs.get_col() + rhs.get_num_nnz(), col);
  auto row_ptr = new NNZType[rhs.get_num_nnz()];
  std::copy(rhs.get_row_ptr(), rhs.get_row_ptr() + rhs.get_num_nnz(), row_ptr);
  ValueType *vals = nullptr;
  if (rhs.get_vals() != nullptr) {
    vals = new ValueType[rhs.get_num_nnz()];
    std::copy(rhs.get_vals(), rhs.get_vals() + rhs.get_num_nnz(), vals);
  }
  this->col_ = std::unique_ptr<IDType, std::function<void(IDType *)>>(
      col, CUDADeleter<IDType>());
  this->row_ptr_ = std::unique_ptr<NNZType, std::function<void(NNZType *)>>(
      row_ptr, CUDADeleter<NNZType>());
  this->vals_ = std::unique_ptr<ValueType, std::function<void(ValueType *)>>(
      vals, CUDADeleter<ValueType>());
  this->context_ = std::unique_ptr<sparsebase::context::Context>(new sparsebase::context::CUDAContext);
}
template <typename IDType, typename NNZType, typename ValueType>
CUDACSR<IDType, NNZType, ValueType>::CUDACSR(IDType n, IDType m, NNZType *row_ptr,
                                     IDType *col, ValueType *vals, context::CUDAContext context,
                                     Ownership own)
    : row_ptr_(row_ptr, BlankDeleter<NNZType>()),
      col_(col, BlankDeleter<IDType>()),
      vals_(vals, BlankDeleter<ValueType>()) {
  this->order_ = 2;
  this->dimension_ = {(DimensionType)n, (DimensionType)m};
  this->nnz_ = this->row_ptr_[this->dimension_[0]];
  if (own == kOwned) {
    this->row_ptr_ = std::unique_ptr<NNZType, std::function<void(NNZType *)>>(
        row_ptr, CUDADeleter<NNZType>());
    this->col_ = std::unique_ptr<IDType, std::function<void(IDType *)>>(
        col, CUDADeleter<IDType>());
    this->vals_ =
        std::unique_ptr<ValueType, std::function<void(ValueType *)>>(
            vals, CUDADeleter<ValueType>());
  }
  this->context_ = std::unique_ptr<sparsebase::context::Context>(new sparsebase::context::CUDAContext);
}

template <typename IDType, typename NNZType, typename ValueType>
Format *CUDACSR<IDType, NNZType, ValueType>::clone() const {
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
void CUDACSR<IDType, NNZType, ValueType>::set_col(IDType *col, context::CUDAContext context, Ownership own) {
  if (own == kOwned) {
    this->col_ = std::unique_ptr<IDType, std::function<void(IDType *)>>(
        col, CUDADeleter<IDType>());
  } else {
    this->col_ = std::unique_ptr<IDType, std::function<void(IDType *)>>(
        col, BlankDeleter<IDType>());
  }
}

template <typename IDType, typename NNZType, typename ValueType>
void CUDACSR<IDType, NNZType, ValueType>::set_row_ptr(NNZType *row_ptr, context::CUDAContext context,
                                                  Ownership own) {
  if (own == kOwned) {
    this->row_ptr_ = std::unique_ptr<NNZType, std::function<void(NNZType *)>>(
        row_ptr, CUDADeleter<NNZType>());
  } else {
    this->row_ptr_ = std::unique_ptr<NNZType, std::function<void(NNZType *)>>(
        row_ptr, BlankDeleter<NNZType>());
  }
}

template <typename IDType, typename NNZType, typename ValueType>
void CUDACSR<IDType, NNZType, ValueType>::set_vals(ValueType *vals, context::CUDAContext context, Ownership own) {
  if (own == kOwned) {
    this->vals_ =
        std::unique_ptr<ValueType, std::function<void(ValueType *)>>(
            vals, CUDADeleter<ValueType>());
  } else {
    this->vals_ =
        std::unique_ptr<ValueType, std::function<void(ValueType *)>>(
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

};
namespace utils {

};
};
