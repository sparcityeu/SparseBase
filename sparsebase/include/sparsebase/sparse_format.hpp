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

  Format format_;
  unsigned int order_;
  std::vector<ID> dimension_;
  NumNonZeros nnz_;
};

template <typename ID, typename NumNonZeros, typename Value>
class COO : public AbstractSparseFormat<ID, NumNonZeros, Value> {
public:
  COO();
  COO(ID n, ID m, NumNonZeros nnz, ID *row, ID *col, Value *vals);
  virtual ~COO();
  Format get_format() override;
  ID *get_col() override;
  ID *get_row() override;
  Value *get_vals() override;

  ID *col_;
  ID *row_;
  Value *vals_;
};
template <typename ID, typename NumNonZeros, typename Value>
class CSR : public AbstractSparseFormat<ID, NumNonZeros, Value> {
public:
  CSR();
  CSR(ID n, ID m, NumNonZeros *row_ptr, ID *col, Value *vals);
  Format get_format() override;
  virtual ~CSR();
  ID *get_row_ptr() override;
  ID *get_col() override;
  Value *get_vals() override;

  NumNonZeros *row_ptr_;
  ID *col_;
  Value *vals_;
};

template <typename ID, typename NumNonZeros, typename Value>
class CSF : public AbstractSparseFormat<ID, NumNonZeros, Value> {
public:
  CSF(unsigned int order);
  Format get_format() override;
  virtual ~CSF();
  ID **get_ind() override;
  Value *get_vals() override;

  NumNonZeros **ind_;
  Value *vals_;
};

} // namespace sparsebase
#endif
