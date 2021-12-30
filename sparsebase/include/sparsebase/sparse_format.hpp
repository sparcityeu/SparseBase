#ifndef _TENSOR_HPP
#define _TENSOR_HPP

#include <algorithm>
#include <cstring>
#include <fstream>
#include <vector>


namespace sparsebase {

//! Enum keeping formats
enum Format {
  //! CSR Format
  CSR_f = 0,
  //! COO Format
  COO_f = 1
};
// TENSORS

template <typename ID, typename NumNonZeros, typename Value> class SparseFormat {
public:
  Format format;
  virtual ~SparseFormat(){};
  virtual unsigned int get_order() = 0;
  virtual Format get_format() = 0;
  virtual std::vector<ID> get_dimensions() = 0;
  virtual NumNonZeros get_num_nnz() = 0;
  virtual NumNonZeros *get_row_ptr() = 0;
  virtual ID *get_col() = 0;
  virtual ID *get_row() = 0;
  virtual Value *get_vals() = 0;
  virtual ID **get_ind() = 0;
};

// abstract class
template <typename ID, typename NumNonZeros, typename Value>
class AbstractSparseFormat : public SparseFormat<ID, NumNonZeros, Value> {
public:
  // initialize order in the constructor
  AbstractSparseFormat();
  virtual ~AbstractSparseFormat();
  unsigned int get_order() override;
  virtual Format get_format() override;
  std::vector<ID> get_dimensions() override;
  NumNonZeros get_num_nnz() override;
  NumNonZeros *get_row_ptr() override;
  ID *get_col() override;
  ID *get_row() override;
  Value *get_vals() override;
  ID **get_ind() override;

  unsigned int order;
  std::vector<ID> dimension;
  Format format;
  NumNonZeros nnz;
};

template <typename ID, typename NumNonZeros, typename Value>
class COO : public AbstractSparseFormat<ID, NumNonZeros, Value> {
public:
  COO();
  COO(ID _n, ID _m, NumNonZeros _nnz, ID *_row, ID *_col, Value *_vals);
  virtual ~COO();
  Format get_format() override;
  ID *col;
  ID *row;
  Value *vals;
  ID *get_col() override;
  ID *get_row() override;
  Value *get_vals() override;
};
template <typename ID, typename NumNonZeros, typename Value>
class CSR : public AbstractSparseFormat<ID, NumNonZeros, Value> {
public:
  CSR();
  CSR(ID _n, ID _m, NumNonZeros *_row_ptr, ID *_col, Value *_vals);
  Format get_format() override;
  virtual ~CSR();
  NumNonZeros *row_ptr;
  ID *col;
  Value *vals;
  ID *get_row_ptr() override;
  ID *get_col() override;
  Value *get_vals() override;
};

template <typename ID, typename NumNonZeros, typename Value>
class CSF : public AbstractSparseFormat<ID, NumNonZeros, Value> {
public:
  CSF(unsigned int order);
  Format get_format() override;
  virtual ~CSF();
  NumNonZeros **ind;
  Value *vals;
  ID **get_ind() override;
  Value *get_vals() override;
};

} // namespace sparsebase
#endif
