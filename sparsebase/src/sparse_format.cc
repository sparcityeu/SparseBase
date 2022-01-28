#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "sparsebase/sparse_exception.h"
#include "sparsebase/sparse_format.h"

using namespace sparsebase::utils;

namespace sparsebase {

namespace format {

template <typename IDType, typename NNZType, typename ValueType>
AbstractSparseFormat<IDType, NNZType, ValueType>::~AbstractSparseFormat(){};
template <typename IDType, typename NNZType, typename ValueType>
AbstractSparseFormat<IDType, NNZType, ValueType>::AbstractSparseFormat() {}

template <typename IDType, typename NNZType, typename ValueType>
unsigned int AbstractSparseFormat<IDType, NNZType, ValueType>::get_order() const {
  return order_;
}

template <typename IDType, typename NNZType, typename ValueType>
Format AbstractSparseFormat<IDType, NNZType, ValueType>::get_format() const {
  return format_;
}

template <typename IDType, typename NNZType, typename ValueType>
std::vector<IDType> AbstractSparseFormat<IDType, NNZType, ValueType>::get_dimensions() const {
  return dimension_;
}

template <typename IDType, typename NNZType, typename ValueType>
NNZType AbstractSparseFormat<IDType, NNZType, ValueType>::get_num_nnz() const {
  return nnz_;
}

template <typename IDType, typename NNZType, typename ValueType>
NNZType* AbstractSparseFormat<IDType, NNZType, ValueType>::get_row_ptr() const {
  throw InvalidDataMember(std::to_string(get_format()), std::string("row_ptr"));
}

template <typename IDType, typename NNZType, typename ValueType>
IDType* AbstractSparseFormat<IDType, NNZType, ValueType>::get_col() const {
  throw InvalidDataMember(std::to_string(get_format()), std::string("col"));
}

template <typename IDType, typename NNZType, typename ValueType>
IDType* AbstractSparseFormat<IDType, NNZType, ValueType>::get_row() const {
  throw InvalidDataMember(std::to_string(get_format()), std::string("is"));
}

template <typename IDType, typename NNZType, typename ValueType>
ValueType* AbstractSparseFormat<IDType, NNZType, ValueType>::get_vals() const {
  throw InvalidDataMember(std::to_string(get_format()), std::string("vals"));
}

template <typename IDType, typename NNZType, typename ValueType>
ValueType** AbstractSparseFormat<IDType, NNZType, ValueType>::get_ind() const {
  throw InvalidDataMember(std::to_string(get_format()), std::string("ind"));
}

template <typename IDType, typename NNZType, typename ValueType>
NNZType*AbstractSparseFormat<IDType, NNZType, ValueType>::release_row_ptr() {
  throw InvalidDataMember(std::to_string(get_format()), std::string("row_ptr"));
}

template <typename IDType, typename NNZType, typename ValueType>
IDType* AbstractSparseFormat<IDType, NNZType, ValueType>::release_col() {
  throw InvalidDataMember(std::to_string(get_format()), std::string("col"));
}

template <typename IDType, typename NNZType, typename ValueType>
IDType* AbstractSparseFormat<IDType, NNZType, ValueType>::release_row() {
  throw InvalidDataMember(std::to_string(get_format()), std::string("is"));
}

template <typename IDType, typename NNZType, typename ValueType>
ValueType* AbstractSparseFormat<IDType, NNZType, ValueType>::release_vals() {
  throw InvalidDataMember(std::to_string(get_format()), std::string("vals"));
}

template <typename IDType, typename NNZType, typename ValueType>
ValueType **AbstractSparseFormat<IDType, NNZType, ValueType>::release_ind() {
  throw InvalidDataMember(std::to_string(get_format()), std::string("ind"));
}

template <typename IDType, typename NNZType, typename ValueType>
void AbstractSparseFormat<IDType, NNZType, ValueType>::set_row_ptr(NNZType*, Ownership) {
  throw InvalidDataMember(std::to_string(get_format()), std::string("row_ptr"));
}

template <typename IDType, typename NNZType, typename ValueType>
void AbstractSparseFormat<IDType, NNZType, ValueType>::set_row(IDType*, Ownership) {
  throw InvalidDataMember(std::to_string(get_format()), std::string("row"));
}

template <typename IDType, typename NNZType, typename ValueType>
void AbstractSparseFormat<IDType, NNZType, ValueType>::set_col(IDType*, Ownership) {
  throw InvalidDataMember(std::to_string(get_format()), std::string("col"));
}


template <typename IDType, typename NNZType, typename ValueType>
void AbstractSparseFormat<IDType, NNZType, ValueType>::set_vals(ValueType*, Ownership) {
  throw InvalidDataMember(std::to_string(get_format()), std::string("vals"));
}

template <typename IDType, typename NNZType, typename ValueType>
void AbstractSparseFormat<IDType, NNZType, ValueType>::set_ind(ValueType**, Ownership) {
  throw InvalidDataMember(std::to_string(get_format()), std::string("ind"));
}

template <typename IDType, typename NNZType, typename ValueType>
bool AbstractSparseFormat<IDType, NNZType, ValueType>::RowIsOwned() {
  throw InvalidDataMember(std::to_string(get_format()), std::string("row"));
}

template <typename IDType, typename NNZType, typename ValueType>
bool AbstractSparseFormat<IDType, NNZType, ValueType>::ColIsOwned() {
  throw InvalidDataMember(std::to_string(get_format()), std::string("col"));
}

template <typename IDType, typename NNZType, typename ValueType>
bool AbstractSparseFormat<IDType, NNZType, ValueType>::RowPtrIsOwned() {
  throw InvalidDataMember(std::to_string(get_format()), std::string("row_ptr"));
}

template <typename IDType, typename NNZType, typename ValueType>
bool AbstractSparseFormat<IDType, NNZType, ValueType>::ValsIsOwned() {
  throw InvalidDataMember(std::to_string(get_format()), std::string("vals"));
}

template <typename IDType, typename NNZType, typename ValueType>
bool AbstractSparseFormat<IDType, NNZType, ValueType>::IndIsOwned() {
  throw InvalidDataMember(std::to_string(get_format()), std::string("ind"));
}
//template <typename IDType, typename NNZType, typename ValueType>
//COO<IDType, NNZType, ValueType>::COO() {
//  this->order_ = 2;
//  this->format_ = Format::kCOOFormat;
//  this->dimension_ = std::vector<IDType>(2, 0);
//  this->nnz_ = 0;
//  this->col_  = std::unique_ptr<IDType, void(*)(IDType*)>(nullptr, BlankDeleter<IDType>());
//  this->row_ = std::unique_ptr<IDType, void(*)(IDType*)>(nullptr, BlankDeleter<IDType>());
//  this->vals_ = std::unique_ptr<ValueType, void(*)(ValueType*)>(nullptr, BlankDeleter<ValueType>());
//}
template <typename IDType, typename NNZType, typename ValueType>
COO<IDType, NNZType, ValueType>::COO(COO<IDType, NNZType, ValueType> && rhs): col_(std::move(rhs.col_)), row_(std::move(rhs.row_)), vals_(std::move(rhs.vals_)) {
  this->nnz_ = rhs.get_num_nnz();
  this->format_ = Format::kCSRFormat;
  this->order_ = 2;
  this->dimension_ = rhs.dimension_;
  rhs.col_ = std::unique_ptr<IDType[], std::function<void (IDType*)>>(nullptr, BlankDeleter<IDType>());
  rhs.row_ = std::unique_ptr<IDType[], std::function<void (IDType*)>>(nullptr, BlankDeleter<IDType>());
  rhs.vals_ = std::unique_ptr<ValueType[], std::function<void (ValueType*)>>(nullptr, BlankDeleter<ValueType>());
}
template <typename IDType, typename NNZType, typename ValueType>
COO<IDType, NNZType, ValueType>& COO<IDType, NNZType, ValueType>::operator=(const COO<IDType, NNZType, ValueType> & rhs){
  this->nnz_ = rhs.nnz_;
  this->format_ = Format::kCOOFormat;
  this->order_ = 2;
  this->dimension_ = rhs.dimension_;
  auto col = new IDType[rhs.get_num_nnz()];
  std::copy(rhs.get_col(), rhs.get_col()+rhs.get_num_nnz(), col);
  auto row = new IDType[rhs.get_num_nnz()];
  std::copy(rhs.get_row(), rhs.get_row()+rhs.get_num_nnz(), row);
  ValueType* vals = nullptr;
  if (rhs.get_vals() != nullptr){
    vals = new ValueType[rhs.get_num_nnz()];
    std::copy(rhs.get_vals(), rhs.get_vals()+rhs.get_num_nnz(), vals);
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
COO<IDType, NNZType, ValueType>::COO(const COO<IDType, NNZType, ValueType> & rhs): col_(nullptr, BlankDeleter<IDType>()), row_(nullptr, BlankDeleter<IDType>()), vals_(nullptr, BlankDeleter<ValueType>()) {
  this->nnz_ = rhs.nnz_;
  this->format_ = Format::kCOOFormat;
  this->order_ = 2;
  this->dimension_ = rhs.dimension_;
  auto col = new IDType[rhs.get_num_nnz()];
  std::copy(rhs.get_col(), rhs.get_col()+rhs.get_num_nnz(), col);
  auto row = new IDType[rhs.get_num_nnz()];
  std::copy(rhs.get_row(), rhs.get_row()+rhs.get_num_nnz(), row);
  ValueType* vals = nullptr;
  if (rhs.get_vals() != nullptr){
    vals = new ValueType[rhs.get_num_nnz()];
    std::copy(rhs.get_vals(), rhs.get_vals()+rhs.get_num_nnz(), vals);
  }
  this->col_ = std::unique_ptr<IDType[], std::function<void(IDType *)>>(
      col, Deleter<IDType>());
  this->row_ = std::unique_ptr<IDType[], std::function<void(IDType *)>>(
      row, Deleter<IDType>());
  this->vals_ = std::unique_ptr<ValueType[], std::function<void(ValueType *)>>(
      vals, Deleter<ValueType>());
}
template <typename IDType, typename NNZType, typename ValueType>
COO<IDType, NNZType, ValueType>::COO(IDType n, IDType m, NNZType nnz, IDType *row,
                             IDType *col, ValueType *vals, Ownership own): col_(col, BlankDeleter<IDType>()), row_(row, BlankDeleter<IDType>()), vals_(vals, BlankDeleter<ValueType>()) {
  this->nnz_ = nnz;
  this->format_ = Format::kCSRFormat;
  this->order_ = 2;
  this->dimension_ = {n, m};
  if (own == kOwned) {
    this->col_  = std::unique_ptr<IDType[], std::function<void (IDType*)>>(col, Deleter<IDType>());
    this->row_ = std::unique_ptr<IDType[], std::function<void (IDType*)>>(row, Deleter<IDType>());
    this->vals_ = std::unique_ptr<ValueType[], std::function<void (ValueType*)>>(vals, Deleter<ValueType>());
  }
}
template <typename IDType, typename NNZType, typename ValueType>
Format COO<IDType, NNZType, ValueType>::get_format() const {
  return kCOOFormat;
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
ValueType * COO<IDType, NNZType, ValueType>::get_vals() const {
  return vals_.get();
}
template <typename IDType, typename NNZType, typename ValueType>
IDType*  COO<IDType, NNZType, ValueType>::release_col() {
  auto col = col_.release();
  this->col_  = std::unique_ptr<IDType[], std::function<void (IDType*)>>(col, BlankDeleter<IDType>());
  return col;
}
template <typename IDType, typename NNZType, typename ValueType>
IDType*  COO<IDType, NNZType, ValueType>::release_row() {
  auto row = row_.release();
  this->row_  = std::unique_ptr<IDType[], std::function<void (IDType*)>>(row, BlankDeleter<IDType>());
  return row;
}
template <typename IDType, typename NNZType, typename ValueType>
ValueType* COO<IDType, NNZType, ValueType>::release_vals() {
  auto vals = vals_.release();
  this->vals_  = std::unique_ptr<ValueType[], std::function<void (ValueType*)>>(vals, BlankDeleter<ValueType>());
  return vals;
}

template <typename IDType, typename NNZType, typename ValueType>
void COO<IDType, NNZType, ValueType>::set_col(IDType* col, Ownership own) {
  if (own == kOwned){
    this->col_  = std::unique_ptr<IDType[], std::function<void (IDType*)>>(col, Deleter<IDType>());
  } else {
    this->col_  = std::unique_ptr<IDType[], std::function<void (IDType*)>>(col, BlankDeleter<IDType>());
  }
}

template <typename IDType, typename NNZType, typename ValueType>
void COO<IDType, NNZType, ValueType>::set_row(IDType* row, Ownership own) {
  if (own == kOwned){
    this->row_  = std::unique_ptr<IDType[], std::function<void (IDType*)>>(row, Deleter<IDType>());
  } else {
    this->row_  = std::unique_ptr<IDType[], std::function<void (IDType*)>>(row, BlankDeleter<IDType>());
  }
}

template <typename IDType, typename NNZType, typename ValueType>
void COO<IDType, NNZType, ValueType>::set_vals(ValueType* vals, Ownership own) {
  if (own == kOwned){
    this->vals_  = std::unique_ptr<ValueType[], std::function<void (ValueType*)>>(vals, Deleter<ValueType>());
  } else {
    this->vals_  = std::unique_ptr<ValueType[], std::function<void (ValueType*)>>(vals, BlankDeleter<ValueType>());
  }
}

template <typename IDType, typename NNZType, typename ValueType>
bool COO<IDType, NNZType, ValueType>::RowIsOwned() {
  return (this->row_.get_deleter().target_type() != typeid(BlankDeleter<IDType>));
}

template <typename IDType, typename NNZType, typename ValueType>
bool COO<IDType, NNZType, ValueType>::ColIsOwned() {
  return (this->col_.get_deleter().target_type() != typeid(BlankDeleter<IDType>));
}

template <typename IDType, typename NNZType, typename ValueType>
bool COO<IDType, NNZType, ValueType>::ValsIsOwned() {
  return (this->vals_.get_deleter().target_type() != typeid(BlankDeleter<ValueType>));
}

template <typename IDType, typename NNZType, typename ValueType>
COO<IDType, NNZType, ValueType>::~COO(){};

//template <typename IDType, typename NNZType, typename ValueType>
//CSR<IDType, NNZType, ValueType>::CSR() {
//  this->order_ = 2;
//  this->format_ = Format::kCSRFormat;
//  this->dimension_ = std::vector<IDType>(2, 0);
//  this->nnz_ = 0;
//  this->row_ptr_ = std::unique_ptr<NNZType, void(*)(NNZType*)>(nullptr, BlankDeleter<NNZType>());
//  this->col_  = std::unique_ptr<IDType, void(*)(NNZType*)>(nullptr, BlankDeleter<IDType>());
//  this->vals_ = std::unique_ptr<ValueType, void(*)(NNZType*)>(nullptr, BlankDeleter<ValueType>());
//}
template <typename IDType, typename NNZType, typename ValueType>
CSR<IDType, NNZType, ValueType>::CSR(CSR<IDType, NNZType, ValueType> && rhs): col_(std::move(rhs.col_)), row_ptr_(std::move(rhs.row_ptr_)), vals_(std::move(rhs.vals_)) {
  this->nnz_ = rhs.get_num_nnz();
  this->format_ = Format::kCSRFormat;
  this->order_ = 2;
  this->dimension_ = rhs.dimension_;
  rhs.col_ = std::unique_ptr<IDType[], std::function<void (IDType*)>>(nullptr, BlankDeleter<IDType>());
  rhs.row_ptr_ = std::unique_ptr<NNZType[], std::function<void (NNZType*)>>(nullptr, BlankDeleter<NNZType>());
  rhs.vals_ = std::unique_ptr<ValueType[], std::function<void (ValueType*)>>(nullptr, BlankDeleter<ValueType>());
}
template <typename IDType, typename NNZType, typename ValueType>
CSR<IDType, NNZType, ValueType>& CSR<IDType, NNZType, ValueType>::operator=(const CSR<IDType, NNZType, ValueType> & rhs){
  this->nnz_ = rhs.nnz_;
  this->format_ = Format::kCSRFormat;
  this->order_ = 2;
  this->dimension_ = rhs.dimension_;
  auto col = new IDType[rhs.get_num_nnz()];
  std::copy(rhs.get_col(), rhs.get_col()+rhs.get_num_nnz(), col);
  auto row_ptr = new NNZType[rhs.get_num_nnz()];
  std::copy(rhs.get_row_ptr(), rhs.get_row_ptr()+rhs.get_num_nnz(), row_ptr);
  ValueType* vals = nullptr;
  if (rhs.get_vals() != nullptr){
    vals = new ValueType[rhs.get_num_nnz()];
    std::copy(rhs.get_vals(), rhs.get_vals()+rhs.get_num_nnz(), vals);
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
CSR<IDType, NNZType, ValueType>::CSR(const CSR<IDType, NNZType, ValueType> & rhs): col_(nullptr, BlankDeleter<IDType>()), row_ptr_(nullptr, BlankDeleter<NNZType>()), vals_(nullptr, BlankDeleter<ValueType>()) {
  this->nnz_ = rhs.nnz_;
  this->format_ = Format::kCSRFormat;
  this->order_ = 2;
  this->dimension_ = rhs.dimension_;
  auto col = new IDType[rhs.get_num_nnz()];
  std::copy(rhs.get_col(), rhs.get_col()+rhs.get_num_nnz(), col);
  auto row_ptr = new NNZType[rhs.get_num_nnz()];
  std::copy(rhs.get_row_ptr(), rhs.get_row_ptr()+rhs.get_num_nnz(), row_ptr);
  ValueType* vals = nullptr;
  if (rhs.get_vals() != nullptr){
    vals = new ValueType[rhs.get_num_nnz()];
    std::copy(rhs.get_vals(), rhs.get_vals()+rhs.get_num_nnz(), vals);
  }
  this->col_ = std::unique_ptr<IDType[], std::function<void(IDType *)>>(
      col, Deleter<IDType>());
  this->row_ptr_ = std::unique_ptr<NNZType[], std::function<void(NNZType *)>>(
      row_ptr, Deleter<NNZType>());
  this->vals_ = std::unique_ptr<ValueType[], std::function<void(ValueType *)>>(
      vals, Deleter<ValueType>());
}
template <typename IDType, typename NNZType, typename ValueType>
CSR<IDType, NNZType, ValueType>::CSR(IDType n, IDType m, NNZType *row_ptr, IDType *col,
                             ValueType *vals, Ownership own): row_ptr_(row_ptr, BlankDeleter<NNZType>()), col_(col, BlankDeleter<IDType>()), vals_(vals, BlankDeleter<ValueType>()) {
  this->format_ = Format::kCSRFormat;
  this->order_ = 2;
  this->dimension_ = {n, m};
  this->nnz_ = this->row_ptr_[this->dimension_[0]];
  if (own == kOwned){
    this->row_ptr_ = std::unique_ptr<NNZType[], std::function<void(NNZType*)>>(row_ptr, Deleter<NNZType>());
    this->col_  = std::unique_ptr<IDType[], std::function<void (IDType*)>>(col, Deleter<IDType>());
    this->vals_ = std::unique_ptr<ValueType[], std::function<void (ValueType*)>>(vals, Deleter<ValueType>());
  }
}
template <typename IDType, typename NNZType, typename ValueType>
Format CSR<IDType, NNZType, ValueType>::get_format() const {
  return kCSRFormat;
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
IDType* CSR<IDType, NNZType, ValueType>::release_col() {
  auto col = col_.release();
  this->col_  = std::unique_ptr<IDType[], std::function<void (IDType*)>>(col, BlankDeleter<IDType>());
  return col;
}
template <typename IDType, typename NNZType, typename ValueType>
NNZType* CSR<IDType, NNZType, ValueType>::release_row_ptr() {
  auto row_ptr = row_ptr_.release();
  this->row_ptr_ = std::unique_ptr<NNZType[], std::function<void(NNZType*)>>(row_ptr, BlankDeleter<NNZType>());
  return row_ptr;
}
template <typename IDType, typename NNZType, typename ValueType>
ValueType* CSR<IDType, NNZType, ValueType>::release_vals() {
  auto vals = vals_.release();
  this->vals_ = std::unique_ptr<ValueType[], std::function<void (ValueType*)>>(vals, BlankDeleter<ValueType>());
  return vals;
}

template <typename IDType, typename NNZType, typename ValueType>
void CSR<IDType, NNZType, ValueType>::set_col(IDType* col, Ownership own) {
  if (own == kOwned){
    this->col_  = std::unique_ptr<IDType[], std::function<void (IDType*)>>(col, Deleter<IDType>());
  } else {
    this->col_  = std::unique_ptr<IDType[], std::function<void (IDType*)>>(col, BlankDeleter<IDType>());
  }
}

template <typename IDType, typename NNZType, typename ValueType>
void CSR<IDType, NNZType, ValueType>::set_row_ptr(NNZType* row_ptr, Ownership own) {
  if (own == kOwned){
    this->row_ptr_  = std::unique_ptr<NNZType[], std::function<void (NNZType*)>>(row_ptr, Deleter<NNZType>());
  } else {
    this->row_ptr_  = std::unique_ptr<NNZType[], std::function<void (NNZType*)>>(row_ptr, BlankDeleter<NNZType>());
  }
}

template <typename IDType, typename NNZType, typename ValueType>
void CSR<IDType, NNZType, ValueType>::set_vals(ValueType* vals, Ownership own) {
  if (own == kOwned){
    this->vals_  = std::unique_ptr<ValueType[], std::function<void (ValueType*)>>(vals, Deleter<ValueType>());
  } else {
    this->vals_  = std::unique_ptr<ValueType[], std::function<void (ValueType*)>>(vals, BlankDeleter<ValueType>());
  }
}

template <typename IDType, typename NNZType, typename ValueType>
bool CSR<IDType, NNZType, ValueType>::RowPtrIsOwned() {
  return (this->row_ptr_.get_deleter().target_type() != typeid(BlankDeleter<NNZType>));
}

template <typename IDType, typename NNZType, typename ValueType>
bool CSR<IDType, NNZType, ValueType>::ColIsOwned() {
  return (this->col_.get_deleter().target_type() != typeid(BlankDeleter<IDType>));
}

template <typename IDType, typename NNZType, typename ValueType>
bool CSR<IDType, NNZType, ValueType>::ValsIsOwned() {
  return (this->vals_.get_deleter().target_type() != typeid(BlankDeleter<ValueType>));
}
template <typename IDType, typename NNZType, typename ValueType>
CSR<IDType, NNZType, ValueType>::~CSR() {}

//template <typename IDType, typename NNZType, typename ValueType>
//CSF<IDType, NNZType, ValueType>::CSF(unsigned int order) {
//  // init CSF
//}
//template <typename IDType, typename NNZType, typename ValueType>
//IDType **CSF<IDType, NNZType, ValueType>::get_ind() {
//  return ind_;
//}
//template <typename IDType, typename NNZType, typename ValueType>
//ValueType *CSF<IDType, NNZType, ValueType>::get_vals() {
//  return vals_;
//}
//template <typename IDType, typename NNZType, typename ValueType>
//CSF<IDType, NNZType, ValueType>::~CSF(){};

#ifdef NDEBUG
#include "init/sparse_format.inc"
#else
template class COO<int, int, int>;
template class COO<unsigned int, unsigned int, unsigned int>;
template class CSR<unsigned int, unsigned int, unsigned int>;
template class CSR<int, int, int>;
#endif

} //namespace format

}; // namespace sparsebase