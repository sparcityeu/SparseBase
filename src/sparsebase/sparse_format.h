#ifndef _TENSOR_HPP
#define _TENSOR_HPP

#include "config.h"
#include "sparse_exception.h"
#include <algorithm>
#include <cstring>
#include <fstream>
#include <functional>
#include <memory>
#include <typeindex>
#include <typeinfo>
#include <vector>

namespace sparsebase {

namespace context {
  struct Context{
    virtual bool IsEquivalent(Context*) const = 0;
    virtual std::type_index get_context_type_member() const = 0;
    virtual ~Context();
  };

  template <typename ContextType> 
  struct ContextImplementation : public Context {
    virtual std::type_index get_context_type_member() const {
      return typeid(ContextType);
    }
    static std::type_index get_context_type() {
      return typeid(ContextType);
    }
  };

  struct CPUContext : ContextImplementation<CPUContext>{
    virtual bool IsEquivalent(Context *) const;
  };

};

namespace format {

enum Ownership {
  kNotOwned = 0,
  kOwned = 1,
};

template <typename T> struct Deleter {
  void operator()(T *obj) {
    if (obj != nullptr)
      delete obj;
  }
};

template <long long dim, typename T> struct Deleter2D {
  void operator()(T *obj) {
    for (long long i = 0; i < dim; i++) {
      delete[] obj[i];
    }
    delete[] obj;
  }
};

template <class T> struct BlankDeleter {
  void operator()(T *obj) {}
};

typedef unsigned long long DimensionType;

class Format {
public:
  virtual std::type_index get_format_id() = 0;
  virtual ~Format() = default;

  virtual Format *Clone() const = 0;
  virtual std::vector<DimensionType> get_dimensions() const = 0;
  virtual DimensionType get_num_nnz() const = 0;
  virtual DimensionType get_order() const = 0;
  virtual context::Context* get_context() const = 0;
  virtual std::type_index get_context_type() const = 0;

  template <typename T> T *As() {
    if (this->get_format_id() == std::type_index(typeid(T))) {
      return static_cast<T *>(this);
    }
    throw utils::TypeException(get_format_id().name(), typeid(T).name());
  }
};

template <typename FormatType> class FormatImplementation : public Format {
public:
  virtual std::vector<DimensionType> get_dimensions() const{
    return dimension_;
  }
  virtual DimensionType get_num_nnz() const{
    return nnz_;
  }
  virtual DimensionType get_order() const{
    return order_;
  }
  virtual context::Context* get_context() const{
    return context_.get();
  }
  std::type_index get_format_id() final {
    return typeid(FormatType);
  }
  static std::type_index get_format_id_static() { return typeid(FormatType); }

  virtual std::type_index get_context_type() const {
    return this->context_->get_context_type_member();
  }

protected:
  DimensionType order_;
  std::vector<DimensionType> dimension_;
  DimensionType nnz_;
  std::unique_ptr<sparsebase::context::Context> context_;
};

template <typename IDType, typename NNZType, typename ValueType>
class COO : public FormatImplementation<COO<IDType, NNZType, ValueType>> {
public:
  COO(IDType n, IDType m, NNZType nnz, IDType *row, IDType *col,
      ValueType *vals, Ownership own = kNotOwned);
  COO(const COO<IDType, NNZType, ValueType> &);
  COO(COO<IDType, NNZType, ValueType> &&);
  COO<IDType, NNZType, ValueType> &
  operator=(const COO<IDType, NNZType, ValueType> &);
  Format *Clone() const override;
  virtual ~COO();
  IDType *get_col() const;
  IDType *get_row() const;
  ValueType *get_vals() const;

  IDType *release_col();
  IDType *release_row();
  ValueType *release_vals();

  void set_row(IDType *, Ownership own = kNotOwned);
  void set_col(IDType *, Ownership own = kNotOwned);
  void set_vals(ValueType *, Ownership own = kNotOwned);

  virtual bool RowIsOwned();
  virtual bool ColIsOwned();
  virtual bool ValsIsOwned();

protected:
  std::unique_ptr<IDType[], std::function<void(IDType *)>> col_;
  std::unique_ptr<IDType[], std::function<void(IDType *)>> row_;
  std::unique_ptr<ValueType[], std::function<void(ValueType *)>> vals_;
};

template <typename ValueType>
class Array : public FormatImplementation<Array<ValueType>> {
public:
  Array(DimensionType nnz, ValueType *row_ptr, Ownership own = kNotOwned);
  Array(const Array<ValueType> &);
  Array(Array<ValueType> &&);
  Array<ValueType> &
  operator=(const Array<ValueType> &);
  Format *Clone() const override;
  virtual ~Array();
  ValueType *get_vals() const;

  ValueType *release_vals();

  void set_vals(ValueType *, Ownership own = kNotOwned);

  virtual bool ValsIsOwned();

protected:
  std::unique_ptr<ValueType[], std::function<void(ValueType *)>> vals_;
};

template <typename IDType, typename NNZType, typename ValueType>
class CSR : public FormatImplementation<CSR<IDType, NNZType, ValueType>> {
public:
  CSR(IDType n, IDType m, NNZType *row_ptr, IDType *col, ValueType *vals,
      Ownership own = kNotOwned);
  CSR(const CSR<IDType, NNZType, ValueType> &);
  CSR(CSR<IDType, NNZType, ValueType> &&);
  CSR<IDType, NNZType, ValueType> &
  operator=(const CSR<IDType, NNZType, ValueType> &);
  Format *Clone() const override;
  virtual ~CSR();
  NNZType *get_row_ptr() const;
  IDType *get_col() const;
  ValueType *get_vals() const;

  NNZType *release_row_ptr();
  IDType *release_col();
  ValueType *release_vals();

  void set_row_ptr(NNZType *, Ownership own = kNotOwned);
  void set_col(IDType *, Ownership own = kNotOwned);
  void set_vals(ValueType *, Ownership own = kNotOwned);

  virtual bool ColIsOwned();
  virtual bool RowPtrIsOwned();
  virtual bool ValsIsOwned();

protected:
  std::unique_ptr<NNZType[], std::function<void(NNZType *)>> row_ptr_;
  std::unique_ptr<IDType[], std::function<void(IDType *)>> col_;
  std::unique_ptr<ValueType[], std::function<void(ValueType *)>> vals_;
};

} // namespace format

} // namespace sparsebase
#ifdef _HEADER_ONLY
#include "sparse_format.cc"
#ifdef CUDA
#include "cuda/format.cu"
#endif
#endif
#endif
