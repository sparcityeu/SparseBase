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
  kCSRFormat = 0,
  //! COO Format
  kCOOFormat = 1
};
// TENSORS

template <typename IDType, typename NNZType, typename ValueType> class SparseFormat {
public:
  virtual ~SparseFormat(){};
  virtual unsigned int get_order() = 0;
  virtual Format get_format() = 0;
  virtual std::vector<IDType> get_dimensions() = 0;
  virtual NNZType get_num_nnz() = 0;
  virtual NNZType *get_row_ptr() = 0;
  virtual IDType *get_col() = 0;
  virtual IDType *get_row() = 0;
  virtual ValueType *get_vals() = 0;
  virtual IDType **get_ind() = 0;
};

// abstract class
template <typename IDType, typename NNZType, typename ValueType>
class AbstractSparseFormat : public SparseFormat<IDType, NNZType, ValueType> {
public:
  // initialize order in the constructor
  AbstractSparseFormat();
  virtual ~AbstractSparseFormat();
  unsigned int get_order() override;
  virtual Format get_format() override;
  std::vector<IDType> get_dimensions() override;
  NNZType get_num_nnz() override;
  NNZType *get_row_ptr() override;
  IDType *get_col() override;
  IDType *get_row() override;
  ValueType *get_vals() override;
  IDType **get_ind() override;

  Format format_;
  unsigned int order_;
  std::vector<IDType> dimension_;
  NNZType nnz_;
};

template <typename IDType, typename NNZType, typename ValueType>
class COO : public AbstractSparseFormat<IDType, NNZType, ValueType> {
public:
  COO();
  COO(IDType n, IDType m, NNZType nnz, IDType *row, IDType *col, ValueType *vals);
  virtual ~COO();
  Format get_format() override;
  IDType *get_col() override;
  IDType *get_row() override;
  ValueType *get_vals() override;

  IDType *col_;
  IDType *row_;
  ValueType *vals_;
};
template <typename IDType, typename NNZType, typename ValueType>
class CSR : public AbstractSparseFormat<IDType, NNZType, ValueType> {
public:
  CSR();
  CSR(IDType n, IDType m, NNZType *row_ptr, IDType *col, ValueType *vals);
  Format get_format() override;
  virtual ~CSR();
  NNZType *get_row_ptr() override;
  IDType *get_col() override;
  ValueType *get_vals() override;

  NNZType *row_ptr_;
  IDType *col_;
  ValueType *vals_;
};

template <typename IDType, typename NNZType, typename ValueType>
class CSF : public AbstractSparseFormat<IDType, NNZType, ValueType> {
public:
  CSF(unsigned int order);
  Format get_format() override;
  virtual ~CSF();
  IDType **get_ind() override;
  ValueType *get_vals() override;

  NNZType **ind_;
  ValueType *vals_;
};

} // namespace sparsebase
#endif
