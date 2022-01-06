#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "sparsebase/sparse_exception.h"
#include "sparsebase/sparse_format.h"

namespace sparsebase {

template <typename IDType, typename NNZType, typename ValueType>
AbstractSparseFormat<IDType, NNZType, ValueType>::~AbstractSparseFormat(){};
template <typename IDType, typename NNZType, typename ValueType>
AbstractSparseFormat<IDType, NNZType, ValueType>::AbstractSparseFormat() {}

template <typename IDType, typename NNZType, typename ValueType>
unsigned int AbstractSparseFormat<IDType, NNZType, ValueType>::get_order() {
  return order_;
}

template <typename IDType, typename NNZType, typename ValueType>
Format AbstractSparseFormat<IDType, NNZType, ValueType>::get_format() {
  return format_;
}

template <typename IDType, typename NNZType, typename ValueType>
std::vector<IDType> AbstractSparseFormat<IDType, NNZType, ValueType>::get_dimensions() {
  return dimension_;
}

template <typename IDType, typename NNZType, typename ValueType>
NNZType AbstractSparseFormat<IDType, NNZType, ValueType>::get_num_nnz() {
  return nnz_;
}

template <typename IDType, typename NNZType, typename ValueType>
NNZType *AbstractSparseFormat<IDType, NNZType, ValueType>::get_row_ptr() {
  throw InvalidDataMember(std::to_string(get_format()), std::string("row_ptr"));
}

template <typename IDType, typename NNZType, typename ValueType>
IDType *AbstractSparseFormat<IDType, NNZType, ValueType>::get_col() {
  throw InvalidDataMember(std::to_string(get_format()), std::string("col"));
}

template <typename IDType, typename NNZType, typename ValueType>
IDType *AbstractSparseFormat<IDType, NNZType, ValueType>::get_row() {
  throw InvalidDataMember(std::to_string(get_format()), std::string("is"));
}

template <typename IDType, typename NNZType, typename ValueType>
ValueType *AbstractSparseFormat<IDType, NNZType, ValueType>::get_vals() {
  throw InvalidDataMember(std::to_string(get_format()), std::string("vals"));
}

template <typename IDType, typename NNZType, typename ValueType>
IDType **AbstractSparseFormat<IDType, NNZType, ValueType>::get_ind() {
  throw InvalidDataMember(std::to_string(get_format()), std::string("ind"));
}

template <typename IDType, typename NNZType, typename ValueType>
COO<IDType, NNZType, ValueType>::COO() {
  this->order_ = 2;
  this->format_ = Format::kCOOFormat;
  this->dimension_ = std::vector<IDType>(2, 0);
  this->nnz_ = 0;
  col_ = nullptr;
  vals_ = nullptr;
}
template <typename IDType, typename NNZType, typename ValueType>
COO<IDType, NNZType, ValueType>::COO(IDType n, IDType m, NNZType nnz, IDType *row,
                             IDType *col, ValueType *vals) {
  col_ = col;
  row_ = row;
  vals_ = vals;
  this->nnz_ = nnz;
  this->format_ = Format::kCSRFormat;
  this->order_ = 2;
  this->dimension_ = {n, m};
}
template <typename IDType, typename NNZType, typename ValueType>
Format COO<IDType, NNZType, ValueType>::get_format() {
  return kCOOFormat;
}
template <typename IDType, typename NNZType, typename ValueType>
IDType *COO<IDType, NNZType, ValueType>::get_col() {
  return col_;
}
template <typename IDType, typename NNZType, typename ValueType>
IDType *COO<IDType, NNZType, ValueType>::get_row() {
  return row_;
}
template <typename IDType, typename NNZType, typename ValueType>
ValueType *COO<IDType, NNZType, ValueType>::get_vals() {
  return vals_;
}
template <typename IDType, typename NNZType, typename ValueType>
COO<IDType, NNZType, ValueType>::~COO(){};

template <typename IDType, typename NNZType, typename ValueType>
CSR<IDType, NNZType, ValueType>::CSR() {
  this->order_ = 2;
  this->format_ = Format::kCSRFormat;
  this->dimension_ = std::vector<IDType>(2, 0);
  this->nnz_ = 0;
  this->col_ = nullptr;
  this->row_ptr_ = nullptr;
  this->vals_ = nullptr;
}
template <typename IDType, typename NNZType, typename ValueType>
CSR<IDType, NNZType, ValueType>::CSR(IDType n, IDType m, NNZType *row_ptr, IDType *col,
                             ValueType *vals) {
  this->row_ptr_ = row_ptr;
  this->col_ = col;
  this->vals_ = vals;
  this->format_ = Format::kCSRFormat;
  this->order_ = 2;
  this->dimension_ = {n, m};
  this->nnz_ = this->row_ptr_[this->dimension_[0]];
}
template <typename IDType, typename NNZType, typename ValueType>
Format CSR<IDType, NNZType, ValueType>::get_format() {
  return kCSRFormat;
}
template <typename IDType, typename NNZType, typename ValueType>
IDType *CSR<IDType, NNZType, ValueType>::get_col() {
  return col_;
}
template <typename IDType, typename NNZType, typename ValueType>
NNZType *CSR<IDType, NNZType, ValueType>::get_row_ptr() {
  return row_ptr_;
}
template <typename IDType, typename NNZType, typename ValueType>
ValueType *CSR<IDType, NNZType, ValueType>::get_vals() {
  return vals_;
}
template <typename IDType, typename NNZType, typename ValueType>
CSR<IDType, NNZType, ValueType>::~CSR() {}

template <typename IDType, typename NNZType, typename ValueType>
CSF<IDType, NNZType, ValueType>::CSF(unsigned int order) {
  // init CSF
}
template <typename IDType, typename NNZType, typename ValueType>
IDType **CSF<IDType, NNZType, ValueType>::get_ind() {
  return ind_;
}
template <typename IDType, typename NNZType, typename ValueType>
ValueType *CSF<IDType, NNZType, ValueType>::get_vals() {
  return vals_;
}
template <typename IDType, typename NNZType, typename ValueType>
CSF<IDType, NNZType, ValueType>::~CSF(){};

#ifdef NDEBUG
#include "init/sparse_format.inc"
#else
template class COO<int, int, int>;
template class COO<unsigned int, unsigned int, unsigned int>;
template class COO<unsigned int, unsigned int, void>;
template class CSR<unsigned int, unsigned int, unsigned int>;
template class CSR<unsigned int, unsigned int, void>;
template class CSR<int, int, int>;
#endif
}; // namespace sparsebase