#include "sparsebase/format/csc.h"
#include "sparsebase/utils/logger.h"
namespace sparsebase::format {

template <typename IDType, typename NNZType, typename ValueType>
CSC<IDType, NNZType, ValueType>::CSC(CSC<IDType, NNZType, ValueType> &&rhs)
    : row_(std::move(rhs.row_)),
      col_ptr_(std::move(rhs.col_ptr_)),
      vals_(std::move(rhs.vals_)) {
  this->nnz_ = rhs.get_num_nnz();
  this->order_ = 2;
  this->dimension_ = rhs.dimension_;
  rhs.row_ = std::unique_ptr<IDType, std::function<void(IDType *)>>(
      nullptr, BlankDeleter<IDType>());
  rhs.col_ptr_ = std::unique_ptr<NNZType, std::function<void(NNZType *)>>(
      nullptr, BlankDeleter<NNZType>());
  rhs.vals_ = std::unique_ptr<ValueType, std::function<void(ValueType *)>>(
      nullptr, BlankDeleter<ValueType>());
  this->context_ = std::unique_ptr<sparsebase::context::Context>(
      new sparsebase::context::CPUContext);
}
template <typename IDType, typename NNZType, typename ValueType>
CSC<IDType, NNZType, ValueType> &CSC<IDType, NNZType, ValueType>::operator=(
    const CSC<IDType, NNZType, ValueType> &rhs) {
  this->nnz_ = rhs.nnz_;
  this->order_ = 2;
  this->dimension_ = rhs.dimension_;
  auto row = new IDType[rhs.get_num_nnz()];
  std::copy(rhs.get_row(), rhs.get_row() + rhs.get_num_nnz(), row);
  auto col_ptr = new NNZType[(rhs.get_dimensions()[0] + 1)];
  std::copy(rhs.get_col_ptr(),
            rhs.get_col_ptr() + (rhs.get_dimensions()[0] + 1), col_ptr);
  ValueType *vals = nullptr;
  if constexpr (!std::is_same_v<ValueType, void>) {
    if (rhs.get_vals() != nullptr) {
      vals = new ValueType[rhs.get_num_nnz()];
      std::copy(rhs.get_vals(), rhs.get_vals() + rhs.get_num_nnz(), vals);
    }
  }
  this->row_ = std::unique_ptr<IDType, std::function<void(IDType *)>>(
      row, Deleter<IDType>());
  this->col_ptr_ = std::unique_ptr<NNZType, std::function<void(NNZType *)>>(
      col_ptr, Deleter<NNZType>());
  this->vals_ = std::unique_ptr<ValueType, std::function<void(ValueType *)>>(
      vals, Deleter<ValueType>());
  return *this;
}
template <typename IDType, typename NNZType, typename ValueType>
CSC<IDType, NNZType, ValueType>::CSC(const CSC<IDType, NNZType, ValueType> &rhs)
    : row_(nullptr, BlankDeleter<IDType>()),
      col_ptr_(nullptr, BlankDeleter<NNZType>()),
      vals_(nullptr, BlankDeleter<ValueType>()) {
  this->nnz_ = rhs.nnz_;
  this->order_ = 2;
  this->dimension_ = rhs.dimension_;
  auto row = new IDType[rhs.get_num_nnz()];
  std::copy(rhs.get_row(), rhs.get_row() + rhs.get_num_nnz(), row);
  auto col_ptr = new NNZType[(rhs.get_dimensions()[0] + 1)];
  std::copy(rhs.get_col_ptr(),
            rhs.get_col_ptr() + (rhs.get_dimensions()[0] + 1), col_ptr);
  ValueType *vals = nullptr;
  if constexpr (!std::is_same_v<ValueType, void>) {
    if (rhs.get_vals() != nullptr) {
      vals = new ValueType[rhs.get_num_nnz()];
      std::copy(rhs.get_vals(), rhs.get_vals() + rhs.get_num_nnz(), vals);
    }
  }
  this->row_ = std::unique_ptr<IDType, std::function<void(IDType *)>>(
      row, Deleter<IDType>());
  this->col_ptr_ = std::unique_ptr<NNZType, std::function<void(NNZType *)>>(
      col_ptr, Deleter<NNZType>());
  this->vals_ = std::unique_ptr<ValueType, std::function<void(ValueType *)>>(
      vals, Deleter<ValueType>());
  this->context_ = std::unique_ptr<sparsebase::context::Context>(
      new sparsebase::context::CPUContext);
}
template <typename IDType, typename NNZType, typename ValueType>
CSC<IDType, NNZType, ValueType>::CSC(IDType n, IDType m, NNZType *col_ptr,
                                     IDType *row, ValueType *vals,
                                     Ownership own, bool ignore_sort)
    : col_ptr_(col_ptr, BlankDeleter<NNZType>()),
      row_(row, BlankDeleter<IDType>()),
      vals_(vals, BlankDeleter<ValueType>()) {
  this->order_ = 2;
  this->dimension_ = {(DimensionType)n, (DimensionType)m};
  this->nnz_ = col_ptr[this->dimension_[0]];
  if (own == kOwned) {
    this->col_ptr_ = std::unique_ptr<NNZType, std::function<void(NNZType *)>>(
        col_ptr, Deleter<NNZType>());
    this->row_ = std::unique_ptr<IDType, std::function<void(IDType *)>>(
        row, Deleter<IDType>());
    this->vals_ = std::unique_ptr<ValueType, std::function<void(ValueType *)>>(
        vals, Deleter<ValueType>());
  }
  this->context_ = std::unique_ptr<sparsebase::context::Context>(
      new sparsebase::context::CPUContext);

  if (!ignore_sort) {
    bool not_sorted = false;

#pragma omp parallel for default(none) reduction(||            \
                                                 : not_sorted) \
    shared(row, col_ptr, n)
    for (IDType i = 0; i < n; i++) {
      NNZType start = col_ptr[i];
      NNZType end = col_ptr[i + 1];
      IDType prev_value = 0;
      for (NNZType j = start; j < end; j++) {
        if (row[j] < prev_value) {
          not_sorted = true;
          break;
        }
        prev_value = row[j];
      }
    }

    if (not_sorted) {
      utils::Logger logger(typeid(this));
      logger.Log("CSC column array must be sorted. Sorting...",
                 utils::LOG_LVL_WARNING);

#pragma omp parallel for default(none) shared(col_ptr, row, vals, n)
      for (IDType i = 0; i < n; i++) {
        NNZType start = col_ptr[i];
        NNZType end = col_ptr[i + 1];

        if (end - start <= 1) {
          continue;
        }

        if constexpr (std::is_same_v<ValueType, void>) {
          std::vector<IDType> sort_vec;
          for (NNZType j = start; j < end; j++) {
            sort_vec.emplace_back(row[j]);
          }
          std::sort(sort_vec.begin(), sort_vec.end(), std::less<ValueType>());
          for (NNZType j = start; j < end; j++) {
            row[j] = sort_vec[j - start];
          }
        } else {
          std::vector<std::pair<IDType, ValueType>> sort_vec;
          for (NNZType j = start; j < end; j++) {
            ValueType val = (vals != nullptr) ? vals[j] : 0;
            sort_vec.emplace_back(row[j], val);
          }
          std::sort(sort_vec.begin(), sort_vec.end(),
                    std::less<std::pair<IDType, ValueType>>());
          for (NNZType j = start; j < end; j++) {
            if (vals != nullptr) {
              vals[j] = sort_vec[j - start].second;
            }
            row[j] = sort_vec[j - start].first;
          }
        }
      }
    }
  }
}

template <typename IDType, typename NNZType, typename ValueType>
Format *CSC<IDType, NNZType, ValueType>::Clone() const {
  return new CSC(*this);
}
template <typename IDType, typename NNZType, typename ValueType>
IDType *CSC<IDType, NNZType, ValueType>::get_row() const {
  return row_.get();
}
template <typename IDType, typename NNZType, typename ValueType>
NNZType *CSC<IDType, NNZType, ValueType>::get_col_ptr() const {
  return col_ptr_.get();
}
template <typename IDType, typename NNZType, typename ValueType>
ValueType *CSC<IDType, NNZType, ValueType>::get_vals() const {
  return vals_.get();
}
template <typename IDType, typename NNZType, typename ValueType>
IDType *CSC<IDType, NNZType, ValueType>::release_row() {
  auto row = row_.release();
  this->row_ = std::unique_ptr<IDType, std::function<void(IDType *)>>(
      row, BlankDeleter<IDType>());
  return row;
}
template <typename IDType, typename NNZType, typename ValueType>
NNZType *CSC<IDType, NNZType, ValueType>::release_col_ptr() {
  auto col_ptr = col_ptr_.release();
  this->col_ptr_ = std::unique_ptr<NNZType, std::function<void(NNZType *)>>(
      col_ptr, BlankDeleter<NNZType>());
  return col_ptr;
}
template <typename IDType, typename NNZType, typename ValueType>
ValueType *CSC<IDType, NNZType, ValueType>::release_vals() {
  auto vals = vals_.release();
  this->vals_ = std::unique_ptr<ValueType, std::function<void(ValueType *)>>(
      vals, BlankDeleter<ValueType>());
  return vals;
}

template <typename IDType, typename NNZType, typename ValueType>
void CSC<IDType, NNZType, ValueType>::set_row(IDType *row, Ownership own) {
  if (own == kOwned) {
    this->row_ = std::unique_ptr<IDType, std::function<void(IDType *)>>(
        row, Deleter<IDType>());
  } else {
    this->row_ = std::unique_ptr<IDType, std::function<void(IDType *)>>(
        row, BlankDeleter<IDType>());
  }
}

template <typename IDType, typename NNZType, typename ValueType>
void CSC<IDType, NNZType, ValueType>::set_col_ptr(NNZType *col_ptr,
                                                  Ownership own) {
  if (own == kOwned) {
    this->col_ptr_ = std::unique_ptr<NNZType, std::function<void(NNZType *)>>(
        col_ptr, Deleter<NNZType>());
  } else {
    this->col_ptr_ = std::unique_ptr<NNZType, std::function<void(NNZType *)>>(
        col_ptr, BlankDeleter<NNZType>());
  }
}

template <typename IDType, typename NNZType, typename ValueType>
void CSC<IDType, NNZType, ValueType>::set_vals(ValueType *vals, Ownership own) {
  if (own == kOwned) {
    this->vals_ = std::unique_ptr<ValueType, std::function<void(ValueType *)>>(
        vals, Deleter<ValueType>());
  } else {
    this->vals_ = std::unique_ptr<ValueType, std::function<void(ValueType *)>>(
        vals, BlankDeleter<ValueType>());
  }
}

template <typename IDType, typename NNZType, typename ValueType>
bool CSC<IDType, NNZType, ValueType>::ColPtrIsOwned() {
  return (this->col_ptr_.get_deleter().target_type() !=
          typeid(BlankDeleter<NNZType>));
}

template <typename IDType, typename NNZType, typename ValueType>
bool CSC<IDType, NNZType, ValueType>::RowIsOwned() {
  return (this->row_.get_deleter().target_type() !=
          typeid(BlankDeleter<IDType>));
}

template <typename IDType, typename NNZType, typename ValueType>
bool CSC<IDType, NNZType, ValueType>::ValsIsOwned() {
  return (this->vals_.get_deleter().target_type() !=
          typeid(BlankDeleter<ValueType>));
}
template <typename IDType, typename NNZType, typename ValueType>
CSC<IDType, NNZType, ValueType>::~CSC() {}

#ifndef _HEADER_ONLY
#include "init/csc.inc"
#endif
}
