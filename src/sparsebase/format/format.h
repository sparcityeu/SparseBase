/*******************************************************
 * Copyright (c) 2022 SparCity, Amro Alabsi Aljundi, Taha Atahan Akyildiz, Arda Sener
 * All rights reserved.
 *
 * This file is distributed under MIT license.
 * The complete license agreement can be obtained at:
 * https://sparcityeu.github.io/sparsebase/pages/license.html
 ********************************************************/
#ifndef SPARSEBASE_SPARSEBASE_FORMAT_FORMAT_H_
#define SPARSEBASE_SPARSEBASE_FORMAT_FORMAT_H_

#include "sparsebase/config.h"
#include "sparsebase/context/context.h"
#include "sparsebase/utils/exception.h"
#include <algorithm>
#include <cstring>
#include <fstream>
#include <functional>
#include <memory>
#include <typeindex>
#include <typeinfo>
#include <vector>

namespace sparsebase {

namespace format {

//TODO: move to utils
// Thanks to artificial mind blog: https://artificial-mind.net/blog/2020/10/03/always-false
template <typename ... T>
constexpr bool always_false = false;

#include <limits>
#include <stdint.h>

using std::numeric_limits;

// Cross float-integral type conversion is not currently available
template <typename T, typename U>
bool CanTypeFitValue(const U value) {
  if constexpr (std::is_integral_v<T> != std::is_integral_v<U>) return false;
  if constexpr (std::is_integral_v<T> && std::is_integral_v<U> ) {
    const intmax_t botT = []() {
      if constexpr (std::is_floating_point_v<T>)
        return intmax_t(-(numeric_limits<T>::max()));
      else
        return intmax_t(numeric_limits<T>::min());
    }();
    const intmax_t botU = []() {
      if constexpr (std::is_floating_point_v<U>)
        return intmax_t(-(numeric_limits<U>::max()));
      else
        return intmax_t(numeric_limits<U>::min());
    }();
    const uintmax_t topT = uintmax_t(numeric_limits<T>::max());
    const uintmax_t topU = uintmax_t(numeric_limits<U>::max());
    return !((botT > botU && value < (U)(botT)) ||
             (topT < topU && value > (U)(topT)));
  } else if constexpr(!std::is_integral_v<T> && !std::is_integral_v<U> ) {
    const double botT = []() {
      if constexpr (std::is_floating_point_v<T>)
        return T(-(numeric_limits<T>::max()));
      else
        return T(numeric_limits<T>::min());
    }();
    const double botU = []() {
      if constexpr (std::is_floating_point_v<U>)
        return U(-(numeric_limits<U>::max()));
      else
        return U(numeric_limits<U>::min());
    }();
    const double topT = numeric_limits<T>::max();
    const double topU = numeric_limits<U>::max();
    return !((botT > botU && value < (U)(botT)) ||
             (topT < topU && value > (U)(topT)));
  }
  //} else if constexpr (std::is_integral_v<T> && !std::is_integral_v<U> ){
  //  const double topT = double(numeric_limits<T>::max());
  //  const uintmax_t topU = uintmax_t(numeric_limits<U>::max());
  //  const double botT = []() {
  //    if constexpr (std::is_floating_point_v<T>)
  //      return double(-(numeric_limits<T>::max()));
  //    else
  //      return double(numeric_limits<T>::min());
  //  }();
  //  const intmax_t botU = []() {
  //    if constexpr (std::is_floating_point_v<U>)
  //      return intmax_t(-(numeric_limits<U>::max()));
  //    else
  //      return intmax_t(numeric_limits<U>::min());
  //  }();
  //  return !(double(topU) > topT && double(value) > topT) || !(double(botU) < botT && double(value) > topT);
  //} else {
  //  const uintmax_t topT = uintmax_t(numeric_limits<T>::max());
  //  const double topU = double(numeric_limits<U>::max());
  //  const intmax_t botT = []() {
  //    if constexpr (std::is_floating_point_v<T>)
  //      return intmax_t(-(numeric_limits<T>::max()));
  //    else
  //      return intmax_t(numeric_limits<T>::min());
  //  }();
  //  const double botU = []() {
  //    if constexpr (std::is_floating_point_v<U>)
  //      return double(-(numeric_limits<U>::max()));
  //    else
  //      return double(numeric_limits<U>::min());
  //  }();
  //  return !(topU > double(topT) && value > double(topT)) || !(botU < double(botT) && value > double(topT));
  //}
}

//TODO: move to utils
template <typename FromType, typename ToType>
inline bool isTypeConversionSafe(FromType from_val, ToType to_val) {
  return from_val == to_val && CanTypeFitValue<ToType>(from_val);
}

template <typename ToType, typename FromType, typename SizeType>
ToType *ConvertArrayType(FromType *from_ptr, SizeType size) {
  if (from_ptr == nullptr) return nullptr;
  auto to_ptr = new ToType[size];
  for (SizeType i = 0; i < size; i++){
    to_ptr[i] = from_ptr[i];
    if (!isTypeConversionSafe(from_ptr[i], to_ptr[i])) {
      throw utils::TypeException(
          "Could not convert array from type " +
          std::string(std::type_index(typeid(FromType)).name()) + " to type " +
          std::string(std::type_index(typeid(ToType)).name()) + ". Overflow detected");
    }
  }
  return to_ptr;
}

//! Enum depicting the ownership status of a Format instance
enum Ownership {
  //! When used the arrays are owned by the user (Format instance is not
  //! responsible from deallocation)
  kNotOwned = 0,
  //! When used the arrays are owned and will be deallocated by the Format
  //! instance
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

//! Type used to represent sizes in the library (usually will be a 64-bit
//! unsigned integer type)
typedef unsigned long long DimensionType;

//! Base class representing a Sparse Data Format (like CSR, COO etc.)
/*!
 * Format is base class representing a Sparse Data Format.
 * All Sparse Data Format Objects (like CSR and COO) are derived from Format.
 * Most of the library uses Format pointers as input and output.
 * By itself Format has very little functionality besides giving size
 * information. Often a user will have to use the As() function to convert it to
 * a concrete format.
 */
class Format {
public:
  //! Returns a type identifier for the concrete format class
  /*!
   * \return A type_index object from the standard C++ library that represents
   * the type of the underlying concrete object. For example, a CSR and a COO
   * object will have different identifiers. But also, objects of the same class
   * with varying template types (like CSR<int,int,int> and CSR<int,int,float)
   * will also have different identifiers.
   */
  virtual std::type_index get_format_id() = 0;
  virtual ~Format() = default;

  //! Performs a deep copy of the Format object and returns the pointer to the
  //! newly created object
  virtual Format *Clone() const = 0;

  //! Returns the sizes of all dimensions as a vector
  virtual std::vector<DimensionType> get_dimensions() const = 0;

  //! Returns the number of non-zero elements stored
  virtual DimensionType get_num_nnz() const = 0;

  //! Returns the number of dimensions
  virtual DimensionType get_order() const = 0;

  //! Returns the context for the format instance
  virtual context::Context *get_context() const = 0;

  //! Returns the type_index for the context for the format instance
  virtual std::type_index get_context_type() const = 0;

  //! Templated function that can be used to cast to a concrete format class
  /*!
   * \tparam T a concrete format class (for example: CSR<int,int,int>)
   * \return A concrete format pointer to this object (for example:
   * CSR<int,int,int>*)
   */
  template <typename T> typename std::remove_pointer<T>::type *As() {
    using TBase = typename std::remove_pointer<T>::type;
    if (this->get_format_id() == std::type_index(typeid(TBase))) {
      return static_cast<TBase *>(this);
    }
    throw utils::TypeException(get_format_id().name(), typeid(TBase).name());
  }

  //! Templated function that can be used to check the concrete type of this object
  /*!
   * \tparam T a concrete format class (for example: CSR<int,int,int>)
   * \return true if the type of this object is T
   */
  template <typename T> bool Is() {
    using TBase = typename std::remove_pointer<T>::type;
    return this->get_format_id() == std::type_index(typeid(TBase));
  }

};

//! A class derived from the base Format class, mostly used for development
//! purposes
/*!
 * FormatImplementation derives from the Format class.
 * It implements some common functionality used by all formats.
 * The Curiously recurring template pattern (CRTP) is used here for retrieving
 * type information of the subclasses (like CSR and COO).
 *
 *
 * Under normal circumstances users should not interact with this class directly
 * unless they are defining their own format in which case this class should be
 * derived from by passing the newly defined format as a template parameter.
 *
 * \tparam FormatType used for CRTP, should be a concrete format class
 * (for example: CSR<int,int,int>)
 */
template <typename FormatType, typename Base = Format> class FormatImplementation : public Base {
public:
  virtual std::vector<DimensionType> get_dimensions() const {
    return dimension_;
  }
  virtual DimensionType get_num_nnz() const { return nnz_; }
  virtual DimensionType get_order() const { return order_; }
  virtual context::Context *get_context() const { return context_.get(); }

  //! Returns the std::type_index for the concrete Format class that this
  //! instance is a member of
  virtual std::type_index get_format_id() { return typeid(FormatType); }

  //! A static variant of the get_format_id() function
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

template <typename ValueType>
class FormatOrderOne : public FormatImplementation<FormatOrderOne<ValueType>, Format> {
  public:
  //! Converts `this` to a FormatOrderOne object of type ToType<ValueType>
  /*!
   * \param to_context context used to carry out the conversion.
   * \param is_move_conversion whether to carry out a move conversion.
   * \return If `this` is of type `ToType<ValueType>` then it returns the same object
   * but as a different type. If not, it will convert `this` to a new FormatOrderOne 
   * object and return a pointer to the new object.
   */
    template <template <typename> typename ToType>
    ToType<ValueType> *Convert(context::Context *to_context=nullptr,
                               bool is_move_conversion = false);

  template <template <typename> typename ToType, typename ToValueType>
  ToType<ToValueType> *Convert(bool is_move_conversion = false);

    template <template <typename> typename ToType, typename ToValueType>
    struct TypeConverter {
      ToType<ToValueType> * operator()(FormatOrderOne<ValueType>*){
        static_assert(always_false<ToValueType>, "Cannot do type conversion for the requested type. Throw a rock through one of our devs' windows");
      }
    };
};
template <typename IDType, typename NNZType, typename ValueType>
class FormatOrderTwo
    : public FormatImplementation<FormatOrderTwo<IDType, NNZType, ValueType>,
                                  Format> {
  public:
  //! Converts `this` to a FormatOrderTwo object of type ToType<IDType, NNZType, ValueType>
  /*!
   * \param to_context context used to carry out the conversion.
   * \param is_move_conversion whether to carry out a move conversion.
   * \return If `this` is of type `ToType<ValueType>` then it returns the same object
   * but as a different type. If not, it will convert `this` to a new FormatOrderOne 
   * object and return a pointer to the new object.
   */
    template <template <typename, typename, typename> class ToType>
    ToType<IDType, NNZType, ValueType> *
    Convert(context::Context *to_context=nullptr, bool is_move_conversion = false);


  template <template <typename, typename, typename> class ToType, typename ToIDType, typename ToNNZType, typename ToValueType>
  ToType<ToIDType, ToNNZType, ToValueType> *
  Convert(bool is_move_conversion = false);
  template <template <typename, typename, typename> class ToType,
      typename ToIDType, typename ToNNZType, typename ToValueType>
  struct TypeConverter{
    ToType<ToIDType, ToNNZType, ToValueType>* operator()(FormatOrderTwo<IDType, NNZType, ValueType>*){
      static_assert(always_false<ToIDType>, "Cannot do type conversion for the requested type. Throw a rock through one of our devs' windows");
    }
  };

};

//! Coordinate List Sparse Data Format
/*!
 * Coordinate List format keeps 3 arrays row, col, vals.
 * The i-th element is at the coordinate row[i], col[i] in the matrix.
 * The value for the i-th element is vals[i].
 *
 * \tparam IDType type used for the dimensions
 * \tparam NNZType type used for non-zeros and the number of non-zeros
 * \tparam ValueType type used for the stored values
 *
 * N. Sato and W. F. Tinney, "Techniques for Exploiting the Sparsity or the
 * Network Admittance Matrix," in IEEE Transactions on Power Apparatus and
 * Systems, vol. 82, no. 69, pp. 944-950, Dec. 1963,
 * doi: 10.1109/TPAS.1963.291477.
 */
template <typename IDType, typename NNZType, typename ValueType>
class COO : public FormatImplementation<COO<IDType, NNZType, ValueType>, FormatOrderTwo<IDType, NNZType, ValueType>> {
public:
  COO(IDType n, IDType m, NNZType nnz, IDType *row, IDType *col,
      ValueType *vals, Ownership own = kNotOwned, bool ignore_sort = false);
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

//! One dimensional Format class that wraps a native C++ array
/*!
 * This class basically functions as a wrapper for native C++ arrays such that
 * they can be used polymorphically with the rest of the Format classes
 * @tparam ValueType type that the array stores
 */
template <typename ValueType>
class Array : public FormatImplementation<Array<ValueType>, FormatOrderOne<ValueType>> {
public:
  Array(DimensionType nnz, ValueType *row_ptr, Ownership own = kNotOwned);
  Array(const Array<ValueType> &);
  Array(Array<ValueType> &&);
  Array<ValueType> &operator=(const Array<ValueType> &);
  Format *Clone() const override;
  virtual ~Array();
  ValueType *get_vals() const;

  ValueType *release_vals();

  void set_vals(ValueType *, Ownership own = kNotOwned);

  virtual bool ValsIsOwned();

protected:
  std::unique_ptr<ValueType[], std::function<void(ValueType *)>> vals_;
};

//! Compressed Sparse Row Sparse Data Format
/*!
 * Compressed Sparse Row format keeps 3 arrays row_ptr, col, vals.
 * The i-th element in the row_ptr array denotes the index the i-th row starts
 * in the col array.
 * The col, vals arrays are identical to the COO format.
 *
 * \tparam IDType type used for the dimensions
 * \tparam NNZType type used for non-zeros and the number of non-zeros
 * \tparam ValueType type used for the stored values
 *
 * Buluç, Aydın; Fineman, Jeremy T.; Frigo, Matteo; Gilbert, John R.; Leiserson,
 * Charles E. (2009). Parallel sparse matrix-vector and matrix-transpose-vector
 * multiplication using compressed sparse blocks. ACM Symp. on Parallelism
 * in Algorithms and Architectures. CiteSeerX 10.1.1.211.5256.
 */
template <typename IDType, typename NNZType, typename ValueType>
class CSR : public FormatImplementation<CSR<IDType, NNZType, ValueType>, FormatOrderTwo<IDType, NNZType, ValueType>> {
public:
  CSR(IDType n, IDType m, NNZType *row_ptr, IDType *col, ValueType *vals,
      Ownership own = kNotOwned, bool ignore_sort = false);
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

//! Compressed Sparse Column Sparse Data Format
/*!
 * Compressed Sparse Column format keeps 3 arrays col_ptr, row, vals.
 * The i-th element in the col_ptr array denotes the index the i-th column starts
 * in the row array.
 * The row, vals arrays are identical to the COO format.
 *
 * \tparam IDType type used for the dimensions
 * \tparam NNZType type used for non-zeros and the number of non-zeros
 * \tparam ValueType type used for the stored values
 *
 * Buluç, Aydın; Fineman, Jeremy T.; Frigo, Matteo; Gilbert, John R.; Leiserson,
 * Charles E. (2009). Parallel sparse matrix-vector and matrix-transpose-vector
 * multiplication using compressed sparse blocks. ACM Symp. on Parallelism
 * in Algorithms and Architectures. CiteSeerX 10.1.1.211.5256.
 */
template <typename IDType, typename NNZType, typename ValueType>
class CSC : public FormatImplementation<CSC<IDType, NNZType, ValueType>, FormatOrderTwo<IDType, NNZType, ValueType>> {
public:
  CSC(IDType n, IDType m, NNZType *col_ptr, IDType *col, ValueType *vals,
      Ownership own = kNotOwned, bool ignore_sort = false);
  CSC(const CSC<IDType, NNZType, ValueType> &);
  CSC(CSC<IDType, NNZType, ValueType> &&);
  CSC<IDType, NNZType, ValueType> &
  operator=(const CSC<IDType, NNZType, ValueType> &);
  Format *Clone() const override;
  virtual ~CSC();
  NNZType *get_col_ptr() const;
  IDType *get_row() const;
  ValueType *get_vals() const;

  NNZType *release_col_ptr();
  IDType *release_row();
  ValueType *release_vals();

  void set_col_ptr(NNZType *, Ownership own = kNotOwned);
  void set_row(IDType *, Ownership own = kNotOwned);
  void set_vals(ValueType *, Ownership own = kNotOwned);

  virtual bool ColPtrIsOwned();
  virtual bool RowIsOwned();
  virtual bool ValsIsOwned();

protected:
  std::unique_ptr<NNZType[], std::function<void(NNZType *)>> col_ptr_;
  std::unique_ptr<IDType[], std::function<void(IDType *)>> row_;
  std::unique_ptr<ValueType[], std::function<void(ValueType *)>> vals_;
};

template <typename IDType, typename NNZType, typename ValueType>
template <typename ToIDType, typename ToNNZType, typename ToValueType>
struct format::FormatOrderTwo<IDType, NNZType, ValueType>::TypeConverter
    <CSR, ToIDType, ToNNZType, ToValueType> {
  CSR<ToIDType, ToNNZType, ToValueType> *operator()(FormatOrderTwo<IDType, NNZType, ValueType>* source) {
    CSR<IDType, NNZType, ValueType>* csr = source->template As<CSR<IDType, NNZType, ValueType>>();
    auto dims = csr->get_dimensions();
    auto num_nnz = csr->get_num_nnz();
    ToNNZType * new_row_ptr = ConvertArrayType<ToNNZType>(csr->get_row_ptr(), dims[0] + 1);
    ToIDType * new_col = ConvertArrayType<ToIDType>(csr->get_col(), num_nnz);
    ToValueType * new_vals = ConvertArrayType<ToValueType>(csr->get_vals(), num_nnz);
    return new CSR<ToIDType, ToNNZType, ToValueType>(dims[0], dims[1], new_row_ptr, new_col, new_vals, kOwned);
  }
};
template <typename IDType, typename NNZType, typename ValueType>
template <typename ToIDType, typename ToNNZType, typename ToValueType>
struct format::FormatOrderTwo<IDType, NNZType, ValueType>::TypeConverter
    <CSC, ToIDType, ToNNZType, ToValueType> {
  CSC<ToIDType, ToNNZType, ToValueType> *operator()(FormatOrderTwo<IDType, NNZType, ValueType>* source) {
    CSC<IDType, NNZType, ValueType>* csc = source->template As<CSC<IDType, NNZType, ValueType>>();
    auto dims = csc->get_dimensions();
    auto num_nnz = csc->get_num_nnz();
    ToNNZType * new_col_ptr = ConvertArrayType<ToNNZType>(csc->get_col_ptr(), dims[0] + 1);
    ToIDType * new_row = ConvertArrayType<ToIDType>(csc->get_row(), num_nnz);
    ToValueType * new_vals = ConvertArrayType<ToValueType>(csc->get_vals(), num_nnz);
    return new CSC<ToIDType, ToNNZType, ToValueType>(dims[0], dims[1], new_col_ptr, new_row, new_vals, kOwned);
  }
};
template <typename IDType, typename NNZType, typename ValueType>
template <typename ToIDType, typename ToNNZType, typename ToValueType>
struct format::FormatOrderTwo<IDType, NNZType, ValueType>::TypeConverter
    <COO, ToIDType, ToNNZType, ToValueType> {
  COO<ToIDType, ToNNZType, ToValueType> *operator()(FormatOrderTwo<IDType, NNZType, ValueType>* source) {
    COO<IDType, NNZType, ValueType>* coo = source->template As<COO<IDType, NNZType, ValueType>>();
    auto dims = coo->get_dimensions();
    auto num_nnz = coo->get_num_nnz();
    ToIDType * new_col = ConvertArrayType<ToIDType>(coo->get_col(), num_nnz);
    ToIDType * new_row = ConvertArrayType<ToIDType>(coo->get_row(), num_nnz);
    ToValueType * new_vals = ConvertArrayType<ToValueType>(coo->get_vals(), num_nnz);
    return new COO<ToIDType, ToNNZType, ToValueType>(dims[0], dims[1], num_nnz, new_row, new_col, new_vals, kOwned);
  }
};

template <typename ValueType>
template <typename ToValueType>
struct FormatOrderOne<ValueType>::TypeConverter<format::Array, ToValueType> {
  format::Array<ToValueType> * operator()(FormatOrderOne<ValueType>* source){
    auto arr = source->template As<format::Array<ValueType>>();
    auto num_nnz = arr->get_num_nnz();
    ToValueType * new_vals = ConvertArrayType<ToValueType>(arr->get_vals(), num_nnz);
    return new Array<ToValueType>(num_nnz, new_vals, kOwned);
  }
};
} // namespace format

} // namespace sparsebase

#include "sparsebase/utils/converter/converter.h"

namespace sparsebase {
namespace format {

template <typename ValueType>
template <template <typename> class ToType>
ToType<ValueType> *sparsebase::format::FormatOrderOne<ValueType>::Convert(
    context::Context *to_context, bool is_move_conversion) {
  static_assert(std::is_base_of<format::FormatOrderOne<ValueType>,
                                ToType<ValueType>>::value,
                "T must be a format::Format");
  sparsebase::utils::converter::ConverterOrderOne<ValueType> converter;
  context::Context *actual_context =
      to_context == nullptr ?  this->get_context() : to_context;
  return converter
      .Convert(this, ToType<ValueType>::get_format_id_static(), actual_context,
               is_move_conversion)
      ->template As<ToType<ValueType>>();
}
template <typename ValueType>
template <template <typename> typename ToType, typename ToValueType>
ToType<ToValueType> *FormatOrderOne<ValueType>::Convert(bool is_move_conversion){
  static_assert(
      std::is_base_of<format::FormatOrderOne<ToValueType>,
          ToType<ToValueType>>::value,
      "T must be an order one format");

  utils::converter::ConverterOrderOne<ValueType> converter;
  if (this->get_format_id() != ToType<ValueType>::get_format_id_static()) {
    auto converted_format =
    converter.template Convert<ToType<ValueType>>(
        this, this->get_context(), is_move_conversion);
    auto type_converted_format =
        TypeConverter<ToType, ToValueType>()(
            converted_format);
    delete converted_format;
    return type_converted_format;
  } else {
    return TypeConverter<ToType, ToValueType>()(this);
  }
}

template <typename IDType, typename NNZType, typename ValueType>
template <template <typename, typename, typename> class ToType>
ToType<IDType, NNZType, ValueType> *
FormatOrderTwo<IDType, NNZType, ValueType>::Convert(
    context::Context *to_context, bool is_move_conversion) {
  static_assert(
      std::is_base_of<format::FormatOrderTwo<IDType, NNZType, ValueType>,
                      ToType<IDType, NNZType, ValueType>>::value,
      "T must be an order two format");
  sparsebase::utils::converter::ConverterOrderTwo<IDType, NNZType, ValueType>
      converter;
  context::Context* actual_context =
      to_context == nullptr ?  this->get_context() : to_context;
  return converter
      .Convert(this, ToType<IDType, NNZType, ValueType>::get_format_id_static(),
               actual_context, is_move_conversion)
      ->template As<ToType<IDType, NNZType, ValueType>>();
}

template <typename IDType, typename NNZType, typename ValueType>
template <template <typename, typename, typename> class ToType, typename ToIDType, typename ToNNZType, typename ToValueType>
ToType<ToIDType, ToNNZType, ToValueType> *
FormatOrderTwo<IDType, NNZType, ValueType>::Convert(bool is_move_conversion) {
  static_assert(
      std::is_base_of<format::FormatOrderTwo<ToIDType, ToNNZType, ToValueType>,
          ToType<ToIDType, ToNNZType, ToValueType>>::value,
      "T must be an order two format");
  utils::converter::ConverterOrderTwo<IDType, NNZType, ValueType> converter;
  if (this->get_format_id() != ToType<IDType, NNZType, ValueType>::get_format_id_static()) {
    auto converted_format =
        converter.template Convert<ToType<IDType, NNZType, ValueType>>(
            this, this->get_context(), is_move_conversion);
    auto type_converted_format =
        TypeConverter<ToType, ToIDType, ToNNZType, ToValueType>()(
            converted_format);
    delete converted_format;
    return type_converted_format;
  } else {
    return TypeConverter<ToType, ToIDType, ToNNZType, ToValueType>()(
            this);
  }
}

} // namespace format

} // namespace sparsebase
#ifdef _HEADER_ONLY
#include "sparsebase/format/format.cc"
#ifdef USE_CUDA
#include "cuda/format.cu"
#endif
#endif
#endif
