#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "sparsebase/sparse_exception.hpp"
#include "sparsebase/sparse_format.hpp"

namespace sparsebase {

template <typename ID, typename NumNonZeros, typename Value>
AbstractSparseFormat<ID, NumNonZeros, Value>::~AbstractSparseFormat(){};
template <typename ID, typename NumNonZeros, typename Value>
AbstractSparseFormat<ID, NumNonZeros, Value>::AbstractSparseFormat() {}

template <typename ID, typename NumNonZeros, typename Value>
unsigned int AbstractSparseFormat<ID, NumNonZeros, Value>::get_order() {
  return order_;
}

template <typename ID, typename NumNonZeros, typename Value>
Format AbstractSparseFormat<ID, NumNonZeros, Value>::get_format() {
  return format_;
}

template <typename ID, typename NumNonZeros, typename Value>
std::vector<ID> AbstractSparseFormat<ID, NumNonZeros, Value>::get_dimensions() {
  return dimension_;
}

template <typename ID, typename NumNonZeros, typename Value>
NumNonZeros AbstractSparseFormat<ID, NumNonZeros, Value>::get_num_nnz() {
  return nnz_;
}

template <typename ID, typename NumNonZeros, typename Value>
NumNonZeros *AbstractSparseFormat<ID, NumNonZeros, Value>::get_row_ptr() {
  throw InvalidDataMember(std::to_string(get_format()), std::string("row_ptr"));
}

template <typename ID, typename NumNonZeros, typename Value>
ID *AbstractSparseFormat<ID, NumNonZeros, Value>::get_col() {
  throw InvalidDataMember(std::to_string(get_format()), std::string("col"));
}

template <typename ID, typename NumNonZeros, typename Value>
ID *AbstractSparseFormat<ID, NumNonZeros, Value>::get_row() {
  throw InvalidDataMember(std::to_string(get_format()), std::string("is"));
}

template <typename ID, typename NumNonZeros, typename Value>
Value *AbstractSparseFormat<ID, NumNonZeros, Value>::get_vals() {
  throw InvalidDataMember(std::to_string(get_format()), std::string("vals"));
}

template <typename ID, typename NumNonZeros, typename Value>
ID **AbstractSparseFormat<ID, NumNonZeros, Value>::get_ind() {
  throw InvalidDataMember(std::to_string(get_format()), std::string("ind"));
}

template <typename ID, typename NumNonZeros, typename Value>
COO<ID, NumNonZeros, Value>::COO() {
  this->order_ = 2;
  this->format_ = Format::COO_f;
  this->dimension_ = std::vector<ID>(2, 0);
  this->nnz_ = 0;
  col_ = nullptr;
  vals_ = nullptr;
}
template <typename ID, typename NumNonZeros, typename Value>
COO<ID, NumNonZeros, Value>::COO(ID n, ID m, NumNonZeros nnz, ID *row,
                             ID *col, Value *vals) {
  col_ = col;
  row_ = row;
  vals_ = vals;
  this->nnz_ = nnz;
  this->format_ = Format::CSR_f;
  this->order_ = 2;
  this->dimension_ = {n, m};
}
template <typename ID, typename NumNonZeros, typename Value>
Format COO<ID, NumNonZeros, Value>::get_format() {
  return COO_f;
}
template <typename ID, typename NumNonZeros, typename Value>
ID *COO<ID, NumNonZeros, Value>::get_col() {
  return col_;
}
template <typename ID, typename NumNonZeros, typename Value>
ID *COO<ID, NumNonZeros, Value>::get_row() {
  return row_;
}
template <typename ID, typename NumNonZeros, typename Value>
Value *COO<ID, NumNonZeros, Value>::get_vals() {
  return vals_;
}
template <typename ID, typename NumNonZeros, typename Value>
COO<ID, NumNonZeros, Value>::~COO(){};

template <typename ID, typename NumNonZeros, typename Value>
CSR<ID, NumNonZeros, Value>::CSR() {
  this->order_ = 2;
  this->format_ = Format::CSR_f;
  this->dimension_ = std::vector<ID>(2, 0);
  this->nnz_ = 0;
  this->col_ = nullptr;
  this->row_ptr_ = nullptr;
  this->vals_ = nullptr;
}
template <typename ID, typename NumNonZeros, typename Value>
CSR<ID, NumNonZeros, Value>::CSR(ID n, ID m, NumNonZeros *row_ptr, ID *col,
                             Value *vals) {
  this->row_ptr_ = row_ptr;
  this->col_ = col;
  this->vals_ = vals;
  this->format_ = Format::CSR_f;
  this->order_ = 2;
  this->dimension_ = {n, m};
  this->nnz_ = this->row_ptr_[this->dimension_[0]];
}
template <typename ID, typename NumNonZeros, typename Value>
Format CSR<ID, NumNonZeros, Value>::get_format() {
  return CSR_f;
}
template <typename ID, typename NumNonZeros, typename Value>
ID *CSR<ID, NumNonZeros, Value>::get_col() {
  return col_;
}
template <typename ID, typename NumNonZeros, typename Value>
ID *CSR<ID, NumNonZeros, Value>::get_row_ptr() {
  return row_ptr_;
}
template <typename ID, typename NumNonZeros, typename Value>
Value *CSR<ID, NumNonZeros, Value>::get_vals() {
  return vals_;
}
template <typename ID, typename NumNonZeros, typename Value>
CSR<ID, NumNonZeros, Value>::~CSR() {}

template <typename ID, typename NumNonZeros, typename Value>
CSF<ID, NumNonZeros, Value>::CSF(unsigned int order) {
  // init CSF
}
template <typename ID, typename NumNonZeros, typename Value>
ID **CSF<ID, NumNonZeros, Value>::get_ind() {
  return ind_;
}
template <typename ID, typename NumNonZeros, typename Value>
Value *CSF<ID, NumNonZeros, Value>::get_vals() {
  return vals_;
}
template <typename ID, typename NumNonZeros, typename Value>
CSF<ID, NumNonZeros, Value>::~CSF(){};

template class COO<int, int, int>;
template class COO<unsigned int, unsigned int, unsigned int>;
template class COO<unsigned int, unsigned int, void>;
template class CSR<unsigned int, unsigned int, unsigned int>;
template class CSR<unsigned int, unsigned int, void>;
template class CSR<int, int, int>;
}; // namespace sparsebase