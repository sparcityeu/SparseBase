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
  return order;
}

template <typename ID, typename NumNonZeros, typename Value>
Format AbstractSparseFormat<ID, NumNonZeros, Value>::get_format() {
  return format;
}

template <typename ID, typename NumNonZeros, typename Value>
std::vector<ID> AbstractSparseFormat<ID, NumNonZeros, Value>::get_dimensions() {
  return dimension;
}

template <typename ID, typename NumNonZeros, typename Value>
NumNonZeros AbstractSparseFormat<ID, NumNonZeros, Value>::get_num_nnz() {
  return nnz;
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
  this->order = 2;
  this->format = Format::COO_f;
  this->dimension = std::vector<ID>(2, 0);
  this->nnz = 0;
  col = nullptr;
  vals = nullptr;
}
template <typename ID, typename NumNonZeros, typename Value>
COO<ID, NumNonZeros, Value>::COO(ID _n, ID _m, NumNonZeros _nnz, ID *_row,
                             ID *_col, Value *_vals) {
  col = _col;
  row = _row;
  vals = _vals;
  this->nnz = _nnz;
  this->format = Format::CSR_f;
  this->order = 2;
  this->dimension = {_n, _m};
}
template <typename ID, typename NumNonZeros, typename Value>
Format COO<ID, NumNonZeros, Value>::get_format() {
  return COO_f;
}
template <typename ID, typename NumNonZeros, typename Value>
ID *COO<ID, NumNonZeros, Value>::get_col() {
  return col;
}
template <typename ID, typename NumNonZeros, typename Value>
ID *COO<ID, NumNonZeros, Value>::get_row() {
  return row;
}
template <typename ID, typename NumNonZeros, typename Value>
Value *COO<ID, NumNonZeros, Value>::get_vals() {
  return vals;
}
template <typename ID, typename NumNonZeros, typename Value>
COO<ID, NumNonZeros, Value>::~COO(){};

template <typename ID, typename NumNonZeros, typename Value>
CSR<ID, NumNonZeros, Value>::CSR() {
  this->order = 2;
  this->format = Format::CSR_f;
  this->dimension = std::vector<ID>(2, 0);
  this->nnz = 0;
  this->col = nullptr;
  this->row_ptr = nullptr;
  this->vals = nullptr;
}
template <typename ID, typename NumNonZeros, typename Value>
CSR<ID, NumNonZeros, Value>::CSR(ID _n, ID _m, NumNonZeros *_row_ptr, ID *_col,
                             Value *_vals) {
  this->row_ptr = _row_ptr;
  this->col = _col;
  this->vals = _vals;
  this->format = Format::CSR_f;
  this->order = 2;
  this->dimension = {_n, _m};
  this->nnz = this->row_ptr[this->dimension[0]];
}
template <typename ID, typename NumNonZeros, typename Value>
Format CSR<ID, NumNonZeros, Value>::get_format() {
  return CSR_f;
}
template <typename ID, typename NumNonZeros, typename Value>
ID *CSR<ID, NumNonZeros, Value>::get_col() {
  return col;
}
template <typename ID, typename NumNonZeros, typename Value>
ID *CSR<ID, NumNonZeros, Value>::get_row_ptr() {
  return row_ptr;
}
template <typename ID, typename NumNonZeros, typename Value>
Value *CSR<ID, NumNonZeros, Value>::get_vals() {
  return vals;
}
template <typename ID, typename NumNonZeros, typename Value>
CSR<ID, NumNonZeros, Value>::~CSR() {}

template <typename ID, typename NumNonZeros, typename Value>
CSF<ID, NumNonZeros, Value>::CSF(unsigned int order) {
  // init CSF
}
template <typename ID, typename NumNonZeros, typename Value>
ID **CSF<ID, NumNonZeros, Value>::get_ind() {
  return ind;
}
template <typename ID, typename NumNonZeros, typename Value>
Value *CSF<ID, NumNonZeros, Value>::get_vals() {
  return vals;
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