#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "sparsebase/format/format.h"
#include "sparsebase/utils/converter/converter.h"
#include "sparsebase/utils/exception.h"

using namespace sparsebase::utils;

namespace sparsebase::format {

template <typename IDType, typename NNZType, typename ValueType>
COO<IDType, NNZType, ValueType>::COO(COO<IDType, NNZType, ValueType> &&rhs)
    : col_(std::move(rhs.col_)), row_(std::move(rhs.row_)),
      vals_(std::move(rhs.vals_)) {
  this->nnz_ = rhs.get_num_nnz();
  this->order_ = 2;
  this->dimension_ = rhs.dimension_;
  rhs.col_ = std::unique_ptr<IDType[], std::function<void(IDType *)>>(
      nullptr, BlankDeleter<IDType>());
  rhs.row_ = std::unique_ptr<IDType[], std::function<void(IDType *)>>(
      nullptr, BlankDeleter<IDType>());
  rhs.vals_ = std::unique_ptr<ValueType[], std::function<void(ValueType *)>>(
      nullptr, BlankDeleter<ValueType>());
  this->context_ = std::unique_ptr<sparsebase::context::Context>(
      new sparsebase::context::CPUContext);
}
template <typename IDType, typename NNZType, typename ValueType>
COO<IDType, NNZType, ValueType> &COO<IDType, NNZType, ValueType>::operator=(
    const COO<IDType, NNZType, ValueType> &rhs) {
  this->nnz_ = rhs.nnz_;
  this->order_ = 2;
  this->dimension_ = rhs.dimension_;
  auto col = new IDType[rhs.get_num_nnz()];
  std::copy(rhs.get_col(), rhs.get_col() + rhs.get_num_nnz(), col);
  auto row = new IDType[rhs.get_num_nnz()];
  std::copy(rhs.get_row(), rhs.get_row() + rhs.get_num_nnz(), row);
  ValueType *vals = nullptr;
  if (rhs.get_vals() != nullptr) {
    vals = new ValueType[rhs.get_num_nnz()];
    std::copy(rhs.get_vals(), rhs.get_vals() + rhs.get_num_nnz(), vals);
  }
  this->col_ = std::unique_ptr<IDType[], std::function<void(IDType *)>>(
      col, Deleter<IDType>());
  this->row_ = std::unique_ptr<IDType[], std::function<void(IDType *)>>(
      row, Deleter<IDType>());
  this->vals_ = std::unique_ptr<ValueType[], std::function<void(ValueType *)>>(
      vals, Deleter<ValueType>());
  return *this;
}
template <typename IDType, typename NNZType, typename ValueType>
COO<IDType, NNZType, ValueType>::COO(const COO<IDType, NNZType, ValueType> &rhs)
    : col_(nullptr, BlankDeleter<IDType>()),
      row_(nullptr, BlankDeleter<IDType>()),
      vals_(nullptr, BlankDeleter<ValueType>()) {
  this->nnz_ = rhs.nnz_;
  this->order_ = 2;
  this->dimension_ = rhs.dimension_;
  auto col = new IDType[rhs.get_num_nnz()];
  std::copy(rhs.get_col(), rhs.get_col() + rhs.get_num_nnz(), col);
  auto row = new IDType[rhs.get_num_nnz()];
  std::copy(rhs.get_row(), rhs.get_row() + rhs.get_num_nnz(), row);
  ValueType *vals = nullptr;
  if (rhs.get_vals() != nullptr) {
    vals = new ValueType[rhs.get_num_nnz()];
    std::copy(rhs.get_vals(), rhs.get_vals() + rhs.get_num_nnz(), vals);
  }
  this->col_ = std::unique_ptr<IDType[], std::function<void(IDType *)>>(
      col, Deleter<IDType>());
  this->row_ = std::unique_ptr<IDType[], std::function<void(IDType *)>>(
      row, Deleter<IDType>());
  this->vals_ = std::unique_ptr<ValueType[], std::function<void(ValueType *)>>(
      vals, Deleter<ValueType>());
  this->context_ = std::unique_ptr<sparsebase::context::Context>(
      new sparsebase::context::CPUContext);
}
template <typename IDType, typename NNZType, typename ValueType>
COO<IDType, NNZType, ValueType>::COO(IDType n, IDType m, NNZType nnz,
                                     IDType *row, IDType *col, ValueType *vals,
                                     Ownership own, bool ignore_sort)
    : col_(col, BlankDeleter<IDType>()), row_(row, BlankDeleter<IDType>()),
      vals_(vals, BlankDeleter<ValueType>()) {
  this->nnz_ = nnz;
  this->order_ = 2;
  this->dimension_ = {(DimensionType)n, (DimensionType)m};
  if (own == kOwned) {
    this->col_ = std::unique_ptr<IDType[], std::function<void(IDType *)>>(
        col, Deleter<IDType>());
    this->row_ = std::unique_ptr<IDType[], std::function<void(IDType *)>>(
        row, Deleter<IDType>());
    this->vals_ =
        std::unique_ptr<ValueType[], std::function<void(ValueType *)>>(
            vals, Deleter<ValueType>());
  }
  this->context_ = std::unique_ptr<sparsebase::context::Context>(
      new sparsebase::context::CPUContext);

  bool not_sorted = false;
  if(!ignore_sort){
    IDType prev_row = 0;
    IDType prev_col = 0;
    for(DimensionType i=0; i<nnz; i++){
      if(prev_row > row[i] || (prev_row == row[i] && prev_col > col[i])){
        not_sorted = true;
        break;
      }
      prev_row = row[i];
      prev_col = col[i];
    }
  }

  if(not_sorted){
    std::cerr << "COO arrays must be sorted. Sorting..." << std::endl;
    std::vector<std::tuple<IDType, IDType, ValueType>> sort_vec;
    for(DimensionType i=0; i<nnz; i++){
      ValueType value = (vals != nullptr) ? vals[i] : 0;
      sort_vec.emplace_back(row[i], col[i], value);
    }
    std::sort(sort_vec.begin(), sort_vec.end(), [](std::tuple<IDType, IDType, ValueType> t1,
                                                   std::tuple<IDType, IDType, ValueType> t2) {
      if(std::get<0>(t1) == std::get<0>(t2)){
        return std::get<1>(t1) < std::get<1>(t2);
      }
      return std::get<0>(t1) < std::get<0>(t2);
    });

    for(DimensionType i=0; i<nnz; i++){
      auto& t = sort_vec[i];
      row[i] = std::get<0>(t);
      col[i] = std::get<1>(t);

      if(vals != nullptr){
        vals[i] = std::get<2>(t);
      }
    }
  }
}
template <typename IDType, typename NNZType, typename ValueType>
Format *COO<IDType, NNZType, ValueType>::Clone() const {
  return new COO(*this);
}
template <typename IDType, typename NNZType, typename ValueType>
IDType *COO<IDType, NNZType, ValueType>::get_col() const {
  return col_.get();
}
template <typename IDType, typename NNZType, typename ValueType>
IDType *COO<IDType, NNZType, ValueType>::get_row() const {
  return row_.get();
}
template <typename IDType, typename NNZType, typename ValueType>
ValueType *COO<IDType, NNZType, ValueType>::get_vals() const {
  return vals_.get();
}
template <typename IDType, typename NNZType, typename ValueType>
IDType *COO<IDType, NNZType, ValueType>::release_col() {
  auto col = col_.release();
  this->col_ = std::unique_ptr<IDType[], std::function<void(IDType *)>>(
      col, BlankDeleter<IDType>());
  return col;
}
template <typename IDType, typename NNZType, typename ValueType>
IDType *COO<IDType, NNZType, ValueType>::release_row() {
  auto row = row_.release();
  this->row_ = std::unique_ptr<IDType[], std::function<void(IDType *)>>(
      row, BlankDeleter<IDType>());
  return row;
}
template <typename IDType, typename NNZType, typename ValueType>
ValueType *COO<IDType, NNZType, ValueType>::release_vals() {
  auto vals = vals_.release();
  this->vals_ = std::unique_ptr<ValueType[], std::function<void(ValueType *)>>(
      vals, BlankDeleter<ValueType>());
  return vals;
}

template <typename IDType, typename NNZType, typename ValueType>
void COO<IDType, NNZType, ValueType>::set_col(IDType *col, Ownership own) {
  if (own == kOwned) {
    this->col_ = std::unique_ptr<IDType[], std::function<void(IDType *)>>(
        col, Deleter<IDType>());
  } else {
    this->col_ = std::unique_ptr<IDType[], std::function<void(IDType *)>>(
        col, BlankDeleter<IDType>());
  }
}

template <typename IDType, typename NNZType, typename ValueType>
void COO<IDType, NNZType, ValueType>::set_row(IDType *row, Ownership own) {
  if (own == kOwned) {
    this->row_ = std::unique_ptr<IDType[], std::function<void(IDType *)>>(
        row, Deleter<IDType>());
  } else {
    this->row_ = std::unique_ptr<IDType[], std::function<void(IDType *)>>(
        row, BlankDeleter<IDType>());
  }
}

template <typename IDType, typename NNZType, typename ValueType>
void COO<IDType, NNZType, ValueType>::set_vals(ValueType *vals, Ownership own) {
  if (own == kOwned) {
    this->vals_ =
        std::unique_ptr<ValueType[], std::function<void(ValueType *)>>(
            vals, Deleter<ValueType>());
  } else {
    this->vals_ =
        std::unique_ptr<ValueType[], std::function<void(ValueType *)>>(
            vals, BlankDeleter<ValueType>());
  }
}

template <typename IDType, typename NNZType, typename ValueType>
bool COO<IDType, NNZType, ValueType>::RowIsOwned() {
  return (this->row_.get_deleter().target_type() !=
          typeid(BlankDeleter<IDType>));
}

template <typename IDType, typename NNZType, typename ValueType>
bool COO<IDType, NNZType, ValueType>::ColIsOwned() {
  return (this->col_.get_deleter().target_type() !=
          typeid(BlankDeleter<IDType>));
}

template <typename IDType, typename NNZType, typename ValueType>
bool COO<IDType, NNZType, ValueType>::ValsIsOwned() {
  return (this->vals_.get_deleter().target_type() !=
          typeid(BlankDeleter<ValueType>));
}

template <typename IDType, typename NNZType, typename ValueType>
COO<IDType, NNZType, ValueType>::~COO(){};

// template <typename IDType, typename NNZType, typename ValueType>
// CSR<IDType, NNZType, ValueType>::CSR() {
//   this->order_ = 2;
//   this->format_ = Format::kCSRFormat;
//   this->dimension_ = std::vector<IDType>(2, 0);
//   this->nnz_ = 0;
//   this->row_ptr_ = std::unique_ptr<NNZType, void(*)(NNZType*)>(nullptr,
//   BlankDeleter<NNZType>()); this->col_  = std::unique_ptr<IDType,
//   void(*)(NNZType*)>(nullptr, BlankDeleter<IDType>()); this->vals_ =
//   std::unique_ptr<ValueType, void(*)(NNZType*)>(nullptr,
//   BlankDeleter<ValueType>());
// }
template <typename IDType, typename NNZType, typename ValueType>
CSR<IDType, NNZType, ValueType>::CSR(CSR<IDType, NNZType, ValueType> &&rhs)
    : col_(std::move(rhs.col_)), row_ptr_(std::move(rhs.row_ptr_)),
      vals_(std::move(rhs.vals_)) {
  this->nnz_ = rhs.get_num_nnz();
  this->order_ = 2;
  this->dimension_ = rhs.dimension_;
  rhs.col_ = std::unique_ptr<IDType[], std::function<void(IDType *)>>(
      nullptr, BlankDeleter<IDType>());
  rhs.row_ptr_ = std::unique_ptr<NNZType[], std::function<void(NNZType *)>>(
      nullptr, BlankDeleter<NNZType>());
  rhs.vals_ = std::unique_ptr<ValueType[], std::function<void(ValueType *)>>(
      nullptr, BlankDeleter<ValueType>());
  this->context_ = std::unique_ptr<sparsebase::context::Context>(
      new sparsebase::context::CPUContext);
}
template <typename IDType, typename NNZType, typename ValueType>
CSR<IDType, NNZType, ValueType> &CSR<IDType, NNZType, ValueType>::operator=(
    const CSR<IDType, NNZType, ValueType> &rhs) {
  this->nnz_ = rhs.nnz_;
  this->order_ = 2;
  this->dimension_ = rhs.dimension_;
  auto col = new IDType[rhs.get_num_nnz()];
  std::copy(rhs.get_col(), rhs.get_col() + rhs.get_num_nnz(), col);
  auto row_ptr = new NNZType[(rhs.get_dimensions()[0] + 1)];
  std::copy(rhs.get_row_ptr(),
            rhs.get_row_ptr() + (rhs.get_dimensions()[0] + 1), row_ptr);
  ValueType *vals = nullptr;
  if (rhs.get_vals() != nullptr) {
    vals = new ValueType[rhs.get_num_nnz()];
    std::copy(rhs.get_vals(), rhs.get_vals() + rhs.get_num_nnz(), vals);
  }
  this->col_ = std::unique_ptr<IDType[], std::function<void(IDType *)>>(
      col, Deleter<IDType>());
  this->row_ptr_ = std::unique_ptr<NNZType[], std::function<void(NNZType *)>>(
      row_ptr, Deleter<NNZType>());
  this->vals_ = std::unique_ptr<ValueType[], std::function<void(ValueType *)>>(
      vals, Deleter<ValueType>());
  return *this;
}
template <typename IDType, typename NNZType, typename ValueType>
CSR<IDType, NNZType, ValueType>::CSR(const CSR<IDType, NNZType, ValueType> &rhs)
    : col_(nullptr, BlankDeleter<IDType>()),
      row_ptr_(nullptr, BlankDeleter<NNZType>()),
      vals_(nullptr, BlankDeleter<ValueType>()) {
  this->nnz_ = rhs.nnz_;
  this->order_ = 2;
  this->dimension_ = rhs.dimension_;
  auto col = new IDType[rhs.get_num_nnz()];
  std::copy(rhs.get_col(), rhs.get_col() + rhs.get_num_nnz(), col);
  auto row_ptr = new NNZType[(rhs.get_dimensions()[0] + 1)];
  std::copy(rhs.get_row_ptr(),
            rhs.get_row_ptr() + (rhs.get_dimensions()[0] + 1), row_ptr);
  ValueType *vals = nullptr;
  if (rhs.get_vals() != nullptr) {
    vals = new ValueType[rhs.get_num_nnz()];
    std::copy(rhs.get_vals(), rhs.get_vals() + rhs.get_num_nnz(), vals);
  }
  this->col_ = std::unique_ptr<IDType[], std::function<void(IDType *)>>(
      col, Deleter<IDType>());
  this->row_ptr_ = std::unique_ptr<NNZType[], std::function<void(NNZType *)>>(
      row_ptr, Deleter<NNZType>());
  this->vals_ = std::unique_ptr<ValueType[], std::function<void(ValueType *)>>(
      vals, Deleter<ValueType>());
  this->context_ = std::unique_ptr<sparsebase::context::Context>(
      new sparsebase::context::CPUContext);
}
template <typename IDType, typename NNZType, typename ValueType>
CSR<IDType, NNZType, ValueType>::CSR(IDType n, IDType m, NNZType *row_ptr,
                                     IDType *col, ValueType *vals,
                                     Ownership own, bool ignore_sort)
    : row_ptr_(row_ptr, BlankDeleter<NNZType>()),
      col_(col, BlankDeleter<IDType>()),
      vals_(vals, BlankDeleter<ValueType>()) {
  this->order_ = 2;
  this->dimension_ = {(DimensionType)n, (DimensionType)m};
  this->nnz_ = this->row_ptr_[this->dimension_[0]];
  if (own == kOwned) {
    this->row_ptr_ = std::unique_ptr<NNZType[], std::function<void(NNZType *)>>(
        row_ptr, Deleter<NNZType>());
    this->col_ = std::unique_ptr<IDType[], std::function<void(IDType *)>>(
        col, Deleter<IDType>());
    this->vals_ =
        std::unique_ptr<ValueType[], std::function<void(ValueType *)>>(
            vals, Deleter<ValueType>());
  }
  this->context_ = std::unique_ptr<sparsebase::context::Context>(
      new sparsebase::context::CPUContext);

  if(!ignore_sort){
    bool not_sorted = false;

#pragma omp parallel for default(none) reduction(||: not_sorted) shared(col, row_ptr, n)
    for(IDType i=0; i<n; i++){
        NNZType start = row_ptr[i];
        NNZType end = row_ptr[i+1];
        IDType prev_value = 0;
        for(NNZType j=start; j<end; j++){
          if(col[j] < prev_value){
            not_sorted = true;
            break;
          }
          prev_value = col[j];
        }
    }

    if(not_sorted){
      std::cerr << "CSR column array must be sorted. Sorting..." << std::endl;

#pragma omp parallel for default(none) shared(row_ptr, col, vals, n)
      for(IDType i=0; i<n; i++){
        NNZType start = row_ptr[i];
        NNZType end = row_ptr[i+1];

        if(end-start <= 1){
          continue;
        }

        std::vector<std::pair<IDType, ValueType>> sort_vec;
        for(NNZType j=start; j<end; j++){
          ValueType val = (vals != nullptr) ? vals[j] : 0;
          sort_vec.emplace_back(col[j], val);
        }
        std::sort(sort_vec.begin(), sort_vec.end(), std::less<std::pair<IDType, ValueType>>());
        for(NNZType j=start; j<end; j++){
          if(vals != nullptr){
            vals[j] = sort_vec[j-start].second;
          }
          col[j] = sort_vec[j-start].first;
        }
      }

    }
  }

}

template <typename IDType, typename NNZType, typename ValueType>
Format *CSR<IDType, NNZType, ValueType>::Clone() const {
  return new CSR(*this);
}
template <typename IDType, typename NNZType, typename ValueType>
IDType *CSR<IDType, NNZType, ValueType>::get_col() const {
  return col_.get();
}
template <typename IDType, typename NNZType, typename ValueType>
NNZType *CSR<IDType, NNZType, ValueType>::get_row_ptr() const {
  return row_ptr_.get();
}
template <typename IDType, typename NNZType, typename ValueType>
ValueType *CSR<IDType, NNZType, ValueType>::get_vals() const {
  return vals_.get();
}
template <typename IDType, typename NNZType, typename ValueType>
IDType *CSR<IDType, NNZType, ValueType>::release_col() {
  auto col = col_.release();
  this->col_ = std::unique_ptr<IDType[], std::function<void(IDType *)>>(
      col, BlankDeleter<IDType>());
  return col;
}
template <typename IDType, typename NNZType, typename ValueType>
NNZType *CSR<IDType, NNZType, ValueType>::release_row_ptr() {
  auto row_ptr = row_ptr_.release();
  this->row_ptr_ = std::unique_ptr<NNZType[], std::function<void(NNZType *)>>(
      row_ptr, BlankDeleter<NNZType>());
  return row_ptr;
}
template <typename IDType, typename NNZType, typename ValueType>
ValueType *CSR<IDType, NNZType, ValueType>::release_vals() {
  auto vals = vals_.release();
  this->vals_ = std::unique_ptr<ValueType[], std::function<void(ValueType *)>>(
      vals, BlankDeleter<ValueType>());
  return vals;
}

template <typename IDType, typename NNZType, typename ValueType>
void CSR<IDType, NNZType, ValueType>::set_col(IDType *col, Ownership own) {
  if (own == kOwned) {
    this->col_ = std::unique_ptr<IDType[], std::function<void(IDType *)>>(
        col, Deleter<IDType>());
  } else {
    this->col_ = std::unique_ptr<IDType[], std::function<void(IDType *)>>(
        col, BlankDeleter<IDType>());
  }
}

template <typename IDType, typename NNZType, typename ValueType>
void CSR<IDType, NNZType, ValueType>::set_row_ptr(NNZType *row_ptr,
                                                  Ownership own) {
  if (own == kOwned) {
    this->row_ptr_ = std::unique_ptr<NNZType[], std::function<void(NNZType *)>>(
        row_ptr, Deleter<NNZType>());
  } else {
    this->row_ptr_ = std::unique_ptr<NNZType[], std::function<void(NNZType *)>>(
        row_ptr, BlankDeleter<NNZType>());
  }
}

template <typename IDType, typename NNZType, typename ValueType>
void CSR<IDType, NNZType, ValueType>::set_vals(ValueType *vals, Ownership own) {
  if (own == kOwned) {
    this->vals_ =
        std::unique_ptr<ValueType[], std::function<void(ValueType *)>>(
            vals, Deleter<ValueType>());
  } else {
    this->vals_ =
        std::unique_ptr<ValueType[], std::function<void(ValueType *)>>(
            vals, BlankDeleter<ValueType>());
  }
}

template <typename IDType, typename NNZType, typename ValueType>
bool CSR<IDType, NNZType, ValueType>::RowPtrIsOwned() {
  return (this->row_ptr_.get_deleter().target_type() !=
          typeid(BlankDeleter<NNZType>));
}

template <typename IDType, typename NNZType, typename ValueType>
bool CSR<IDType, NNZType, ValueType>::ColIsOwned() {
  return (this->col_.get_deleter().target_type() !=
          typeid(BlankDeleter<IDType>));
}

template <typename IDType, typename NNZType, typename ValueType>
bool CSR<IDType, NNZType, ValueType>::ValsIsOwned() {
  return (this->vals_.get_deleter().target_type() !=
          typeid(BlankDeleter<ValueType>));
}
template <typename IDType, typename NNZType, typename ValueType>
CSR<IDType, NNZType, ValueType>::~CSR() {}

template <typename ValueType>
Array<ValueType>::Array(Array<ValueType> &&rhs) : vals_(std::move(rhs.vals_)) {
  this->nnz_ = rhs.get_num_nnz();
  this->order_ = 1;
  this->dimension_ = rhs.dimension_;
  rhs.vals_ = std::unique_ptr<ValueType[], std::function<void(ValueType *)>>(
      nullptr, BlankDeleter<ValueType>());
  this->context_ = std::unique_ptr<sparsebase::context::Context>(
      new sparsebase::context::CPUContext);
}
template <typename ValueType>
Array<ValueType> &Array<ValueType>::operator=(const Array<ValueType> &rhs) {
  this->nnz_ = rhs.nnz_;
  this->order_ = 1;
  this->dimension_ = rhs.dimension_;
  ValueType *vals = nullptr;
  if (rhs.get_vals() != nullptr) {
    vals = new ValueType[rhs.get_num_nnz()];
    std::copy(rhs.get_vals(), rhs.get_vals() + rhs.get_num_nnz(), vals);
  }
  this->vals_ = std::unique_ptr<ValueType[], std::function<void(ValueType *)>>(
      vals, Deleter<ValueType>());
  return *this;
}
template <typename ValueType>
Array<ValueType>::Array(const Array<ValueType> &rhs)
    : vals_(nullptr, BlankDeleter<ValueType>()) {
  this->nnz_ = rhs.nnz_;
  this->order_ = 1;
  this->dimension_ = rhs.dimension_;
  ValueType *vals = nullptr;
  if (rhs.get_vals() != nullptr) {
    vals = new ValueType[rhs.get_num_nnz()];
    std::copy(rhs.get_vals(), rhs.get_vals() + rhs.get_num_nnz(), vals);
  }
  this->vals_ = std::unique_ptr<ValueType[], std::function<void(ValueType *)>>(
      vals, Deleter<ValueType>());
  this->context_ = std::unique_ptr<sparsebase::context::Context>(
      new sparsebase::context::CPUContext);
}
template <typename ValueType>
Array<ValueType>::Array(DimensionType nnz, ValueType *vals, Ownership own)
    : vals_(vals, BlankDeleter<ValueType>()) {
  this->order_ = 1;
  this->dimension_ = {(DimensionType)nnz};
  this->nnz_ = nnz;
  if (own == kOwned) {
    this->vals_ =
        std::unique_ptr<ValueType[], std::function<void(ValueType *)>>(
            vals, Deleter<ValueType>());
  }
  this->context_ = std::unique_ptr<sparsebase::context::Context>(
      new sparsebase::context::CPUContext);
}

template <typename ValueType> Format *Array<ValueType>::Clone() const {
  return new Array(*this);
}
template <typename ValueType> ValueType *Array<ValueType>::get_vals() const {
  return vals_.get();
}
template <typename ValueType> ValueType *Array<ValueType>::release_vals() {
  auto vals = vals_.release();
  this->vals_ = std::unique_ptr<ValueType[], std::function<void(ValueType *)>>(
      vals, BlankDeleter<ValueType>());
  return vals;
}

template <typename ValueType>
void Array<ValueType>::set_vals(ValueType *vals, Ownership own) {
  if (own == kOwned) {
    this->vals_ =
        std::unique_ptr<ValueType[], std::function<void(ValueType *)>>(
            vals, Deleter<ValueType>());
  } else {
    this->vals_ =
        std::unique_ptr<ValueType[], std::function<void(ValueType *)>>(
            vals, BlankDeleter<ValueType>());
  }
}

template <typename ValueType> bool Array<ValueType>::ValsIsOwned() {
  return (this->vals_.get_deleter().target_type() !=
          typeid(BlankDeleter<ValueType>));
}
template <typename ValueType> Array<ValueType>::~Array() {}
#if !defined(_HEADER_ONLY)
#include "init/format.inc"
#endif

}; // namespace sparsebase::format
