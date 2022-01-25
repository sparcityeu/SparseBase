#ifndef _TENSOR_HPP
#define _TENSOR_HPP

#include <algorithm>
#include <cstring>
#include <fstream>
#include <vector>
#include <memory>


namespace sparsebase {

//! Enum keeping formats
enum Format {
  //! CSR Format
  kCSRFormat = 0,
  //! COO Format
  kCOOFormat = 1
};
// TENSORS

template <typename T>
struct Deleter{
  void operator()(T*obj){
    if constexpr (!std::is_same_v<void, T>) {
      if (obj != nullptr)
        delete obj;
    }
  }
};

template <long long dim, typename T>
struct Deleter2D{
  void operator()(T*obj){
    for (long long i = 0 ; i < dim; i++){
      delete [] obj[i];
    }
    delete [] obj;
  }
};

template <class T>
struct BlankDeleter{
  void operator()(T*obj){
  }
};

template <typename IDType, typename NNZType, typename ValueType> class SparseFormat {
public:
  virtual ~SparseFormat(){};
  virtual unsigned int get_order() const = 0;
  virtual Format get_format() const = 0;
  virtual std::vector<IDType> get_dimensions() const = 0;
  virtual NNZType get_num_nnz() const = 0;
  virtual NNZType * get_row_ptr() const = 0;
  virtual IDType *get_col() const = 0;
  virtual IDType * get_row() const = 0;
  virtual ValueType * get_vals() const = 0;
  virtual ValueType **get_ind() const = 0;
  virtual NNZType* release_row_ptr() = 0;
  virtual IDType* release_col() = 0;
  virtual IDType* release_row() = 0;
  virtual ValueType* release_vals() = 0;
  virtual ValueType ** release_ind() = 0;
};

// abstract class
template <typename IDType, typename NNZType, typename ValueType>
class AbstractSparseFormat : public SparseFormat<IDType, NNZType, ValueType> {
public:
  // initialize order in the constructor
  AbstractSparseFormat();
  virtual ~AbstractSparseFormat();
  unsigned int get_order() const override;
  virtual Format get_format() const override;
  std::vector<IDType> get_dimensions() const override;
  NNZType get_num_nnz() const override;
  NNZType * get_row_ptr() const override;
  IDType *get_col() const override;
  IDType * get_row() const override;
  ValueType * get_vals() const override;
  ValueType **get_ind() const override;

  NNZType* release_row_ptr() override;
  IDType* release_col() override;
  IDType* release_row() override;
  ValueType* release_vals() override;
  ValueType ** release_ind() override;

protected:
  Format format_;
  unsigned int order_;
  std::vector<IDType> dimension_;
  NNZType nnz_;
};

template <typename IDType, typename NNZType, typename ValueType>
class COO : public AbstractSparseFormat<IDType, NNZType, ValueType> {
public:
  //COO();
  COO(IDType n, IDType m, NNZType nnz, IDType *row, IDType *col, ValueType *vals, bool own = false);
  COO(const COO<IDType, NNZType, ValueType>&);
  COO(COO<IDType, NNZType, ValueType>&&);
  virtual ~COO();
  Format get_format() const override;
  IDType *get_col() const override;
  IDType * get_row() const override;
  ValueType * get_vals() const override;

  IDType* release_col() override;
  IDType* release_row() override;
  ValueType* release_vals() override;

protected:
  std::unique_ptr<IDType[], std::function<void(IDType*)>> col_;
  std::unique_ptr<IDType[], std::function<void(IDType*)>> row_;
  std::unique_ptr<ValueType[], std::function<void (ValueType*)>> vals_;
};
template <typename IDType, typename NNZType, typename ValueType>
class CSR : public AbstractSparseFormat<IDType, NNZType, ValueType> {
public:
  //CSR();
  CSR(IDType n, IDType m, NNZType *row_ptr, IDType *col, ValueType *vals, bool own = false);
  CSR(const CSR<IDType, NNZType, ValueType>&);
  CSR(CSR<IDType, NNZType, ValueType>&&);
  Format get_format() const override;
  virtual ~CSR();
  NNZType * get_row_ptr() const override;
  IDType *get_col() const override;
  ValueType * get_vals() const override;

  NNZType* release_row_ptr() override;
  IDType* release_col() override;
  ValueType* release_vals() override;
  
protected:
  std::unique_ptr<NNZType[], std::function<void(NNZType*)>> row_ptr_;
  std::unique_ptr<IDType[], std::function<void(IDType*)>> col_;
  std::unique_ptr<ValueType[], std::function<void(ValueType*)>> vals_;
};

//template <typename IDType, typename NNZType, typename ValueType>
//class CSF : public AbstractSparseFormat<IDType, NNZType, ValueType> {
//public:
//  CSF(unsigned int order);
//  Format get_format() override;
//  virtual ~CSF();
//  IDType **get_ind() override;
//  ValueType *get_vals() override;
//
//  NNZType **ind_;
//  ValueType *vals_;
//};

} // namespace sparsebase
#endif
