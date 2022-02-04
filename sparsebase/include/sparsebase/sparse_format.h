#ifndef _TENSOR_HPP
#define _TENSOR_HPP

#include <algorithm>
#include <cstring>
#include <fstream>
#include <vector>
#include <memory>
#include <functional>
#include <typeindex>
#include <typeinfo>
#include "sparse_exception.h"

namespace sparsebase {

namespace format {

enum Ownership {
  kNotOwned = 0,
  kOwned = 1,
};


template <typename T>
struct Deleter{
  void operator()(T*obj){
    if (obj != nullptr)
      delete obj;
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

class Format {
public:
  virtual std::type_index get_format_id() = 0;

  template<typename T>
  T* As(){
    if(this->get_format_id() == std::type_index(typeid(T))){
      return static_cast<T*>(this);
    }
    throw utils::TypeException(get_format_id().name(), typeid(T).name());
  }
};

template <typename IDType, typename NNZType, typename ValueType, template <typename,typename,typename> class FormatType>
class FormatImpl : public Format{
public:

    std::vector<IDType> get_dimensions() const{
        return dimension_;
    };
    NNZType get_num_nnz() const {
        return nnz_ ;
    };
    unsigned int get_order() const {
        return order_;
    }
    virtual Format* clone() const = 0;

    std::type_index get_format_id() final {
        return typeid(FormatType<IDType,NNZType,ValueType>);
    }

    static std::type_index get_format_id_static(){
        return typeid(FormatType<IDType, NNZType, ValueType>);
    }

    FormatType<IDType,NNZType,ValueType>* This(){
        return this->As<FormatType<IDType,NNZType,ValueType>>();
    }

protected:
    unsigned int order_;
    std::vector<IDType> dimension_;
    NNZType nnz_;

};

template <typename IDType, typename NNZType, typename ValueType>
class COO : public FormatImpl<IDType, NNZType, ValueType, COO> {
public:
  COO(IDType n, IDType m, NNZType nnz, IDType *row, IDType *col, ValueType *vals, Ownership own = kNotOwned);
  COO(const COO<IDType, NNZType, ValueType>&);
  COO(COO<IDType, NNZType, ValueType>&&);
  COO<IDType, NNZType, ValueType>& operator=(const COO<IDType, NNZType, ValueType>&);
  Format* clone() const override;
  virtual ~COO();
  IDType *get_col() const ;
  IDType * get_row() const ;
  ValueType * get_vals() const ;

  IDType* release_col() ;
  IDType* release_row() ;
  ValueType* release_vals() ;

  void set_row(IDType*, Ownership own = kNotOwned) ;
  void set_col(IDType*, Ownership own = kNotOwned) ;
  void set_vals(ValueType*, Ownership own = kNotOwned) ;

  virtual bool RowIsOwned() ;
  virtual bool ColIsOwned() ;
  virtual bool ValsIsOwned() ;
protected:
  std::unique_ptr<IDType[], std::function<void(IDType*)>> col_;
  std::unique_ptr<IDType[], std::function<void(IDType*)>> row_;
  std::unique_ptr<ValueType[], std::function<void (ValueType*)>> vals_;
};


template <typename IDType, typename NNZType, typename ValueType>
class CSR : public FormatImpl<IDType, NNZType, ValueType, CSR> {
public:
  CSR(IDType n, IDType m, NNZType *row_ptr, IDType *col, ValueType *vals, Ownership own = kNotOwned);
  CSR(const CSR<IDType, NNZType, ValueType>&);
  CSR(CSR<IDType, NNZType, ValueType>&&);
  CSR<IDType, NNZType, ValueType>& operator=(const CSR<IDType, NNZType, ValueType>&);
  Format* clone() const override;
  virtual ~CSR();
  NNZType * get_row_ptr() const ;
  IDType *get_col() const ;
  ValueType * get_vals() const ;

  NNZType* release_row_ptr() ;
  IDType* release_col() ;
  ValueType* release_vals() ;

  void set_row_ptr(NNZType*, Ownership own = kNotOwned) ;
  void set_col(IDType*, Ownership own = kNotOwned) ;
  void set_vals(ValueType*, Ownership own = kNotOwned) ;
  
  virtual bool ColIsOwned() ;
  virtual bool RowPtrIsOwned() ;
  virtual bool ValsIsOwned() ;
protected:
  std::unique_ptr<NNZType[], std::function<void(NNZType*)>> row_ptr_;
  std::unique_ptr<IDType[], std::function<void(IDType*)>> col_;
  std::unique_ptr<ValueType[], std::function<void(ValueType*)>> vals_;
};

} // namespace format

} // namespace sparsebase
#endif
