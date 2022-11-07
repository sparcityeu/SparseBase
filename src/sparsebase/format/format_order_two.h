#include <cxxabi.h>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <functional>
#include <memory>
#include <typeindex>
#include <typeinfo>
#include <vector>

#include "sparsebase/config.h"
#include "sparsebase/context/context.h"
#include "sparsebase/utils/exception.h"
#include "sparsebase/utils/utils.h"

#include "sparsebase/format/format_implementation.h"
#ifndef SPARSEBASE_PROJECT_FORMAT_ORDER_TWO_H
#define SPARSEBASE_PROJECT_FORMAT_ORDER_TWO_H
namespace sparsebase::format {

template <typename IDType, typename NNZType, typename ValueType>
class FormatOrderTwo
    : public FormatImplementation {
 public:
  FormatOrderTwo();

  //! Converts `this` to a FormatOrderTwo object of type ToType<IDType, NNZType,
  //! ValueType>
  /*!
   * @param to_context context used to carry out the conversion.
   * @param is_move_conversion whether to carry out a move conversion.
   * @return If `this` is of type `ToType<ValueType>` then it returns the same
   * object but as a different type. If not, it will convert `this` to a new
   * FormatOrderOne object and return a pointer to the new object.
   */
  template <template <typename, typename, typename> class ToType>
  ToType<IDType, NNZType, ValueType> *Convert(
      context::Context *to_context = nullptr, bool is_move_conversion = false);

  template <template <typename, typename, typename> class ToType>
  ToType<IDType, NNZType, ValueType> *Convert(
      const std::vector<context::Context *> &to_context,
      bool is_move_conversion = false);

  template <template <typename, typename, typename> class ToType,
      typename ToIDType, typename ToNNZType, typename ToValueType>
  ToType<ToIDType, ToNNZType, ToValueType> *Convert(
      bool is_move_conversion = false);
  template <template <typename, typename, typename> class ToType,
      typename ToIDType, typename ToNNZType, typename ToValueType>
  struct TypeConverter {
    ToType<ToIDType, ToNNZType, ToValueType> *operator()(
        FormatOrderTwo<IDType, NNZType, ValueType> *, bool) {
      static_assert(utils::always_false<ToIDType>,
                    "Cannot do type conversion for the requested type. Throw a "
                    "rock through one of our devs' windows");
    }
  };

  template <typename T>
  typename std::remove_pointer<T>::type *As() {
    static_assert(utils::always_false<T>,
                  "When casting a FormatOrderTwo, only pass the class name "
                  "without its types");
    return nullptr;
  }
  template <template <typename, typename, typename> typename T>
  typename std::remove_pointer<T<IDType, NNZType, ValueType>>::type *As() {
    static_assert(std::is_base_of_v<FormatOrderTwo<IDType, NNZType, ValueType>,
                  T<IDType, NNZType, ValueType>>,
    "Cannot cast to a non-FormatOrderTwo class");
    using TBase =
        typename std::remove_pointer<T<IDType, NNZType, ValueType>>::type;
    if (this->get_id() == std::type_index(typeid(TBase))) {
      return static_cast<TBase *>(this);
    }
    throw utils::TypeException(this->get_name(),
                               typeid(TBase).name());
  }
};
//! Coordinate List Sparse Data Format
/*!
 * Coordinate List format keeps 3 arrays row, col, vals.
 * The i-th element is at the coordinate row[i], col[i] in the matrix.
 * The value for the i-th element is vals[i].
 *
 * @tparam IDType type used for the dimensions
 * @tparam NNZType type used for non-zeros and the number of non-zeros
 * @tparam ValueType type used for the stored values
 *
 * N. Sato and W. F. Tinney, "Techniques for Exploiting the Sparsity or the
 * Network Admittance Matrix," in IEEE Transactions on Power Apparatus and
 * Systems, vol. 82, no. 69, pp. 944-950, Dec. 1963,
 * doi: 10.1109/TPAS.1963.291477.
 */
template <typename IDType, typename NNZType, typename ValueType>
class COO
    : public utils::IdentifiableImplementation<COO<IDType, NNZType, ValueType>,
        FormatOrderTwo<IDType, NNZType, ValueType>> {
 public:
  COO(IDType n, IDType m, NNZType nnz, IDType *row, IDType *col,
      ValueType *vals, Ownership own = kNotOwned, bool ignore_sort = false);
  COO(const COO<IDType, NNZType, ValueType> &);
  COO(COO<IDType, NNZType, ValueType> &&);
  COO<IDType, NNZType, ValueType> &operator=(
      const COO<IDType, NNZType, ValueType> &);
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
  std::unique_ptr<IDType, std::function<void(IDType *)>> col_;
  std::unique_ptr<IDType, std::function<void(IDType *)>> row_;
  std::unique_ptr<ValueType, std::function<void(ValueType *)>> vals_;
};


//! Compressed Sparse Row Sparse Data Format
/*!
 * Compressed Sparse Row format keeps 3 arrays row_ptr, col, vals.
 * The i-th element in the row_ptr array denotes the index the i-th row starts
 * in the col array.
 * The col, vals arrays are identical to the COO format.
 *
 * @tparam IDType type used for the dimensions
 * @tparam NNZType type used for non-zeros and the number of non-zeros
 * @tparam ValueType type used for the stored values
 *
 * Buluç, Aydın; Fineman, Jeremy T.; Frigo, Matteo; Gilbert, John R.; Leiserson,
 * Charles E. (2009). Parallel sparse matrix-vector and matrix-transpose-vector
 * multiplication using compressed sparse blocks. ACM Symp. on Parallelism
 * in Algorithms and Architectures. CiteSeerX 10.1.1.211.5256.
 */
template <typename IDType, typename NNZType, typename ValueType>
class CSR
    : public utils::IdentifiableImplementation<CSR<IDType, NNZType, ValueType>,
        FormatOrderTwo<IDType, NNZType, ValueType>> {
 public:
  CSR(IDType n, IDType m, NNZType *row_ptr, IDType *col, ValueType *vals,
      Ownership own = kNotOwned, bool ignore_sort = false);
  CSR(const CSR<IDType, NNZType, ValueType> &);
  CSR(CSR<IDType, NNZType, ValueType> &&);
  CSR<IDType, NNZType, ValueType> &operator=(
      const CSR<IDType, NNZType, ValueType> &);
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
  std::unique_ptr<NNZType, std::function<void(NNZType *)>> row_ptr_;
  std::unique_ptr<IDType, std::function<void(IDType *)>> col_;
  std::unique_ptr<ValueType, std::function<void(ValueType *)>> vals_;
};

//! Compressed Sparse Column Sparse Data Format
/*!
 * Compressed Sparse Column format keeps 3 arrays col_ptr, row, vals.
 * The i-th element in the col_ptr array denotes the index the i-th column
 * starts in the row array. The row, vals arrays are identical to the COO
 * format.
 *
 * @tparam IDType type used for the dimensions
 * @tparam NNZType type used for non-zeros and the number of non-zeros
 * @tparam ValueType type used for the stored values
 *
 * Buluç, Aydın; Fineman, Jeremy T.; Frigo, Matteo; Gilbert, John R.; Leiserson,
 * Charles E. (2009). Parallel sparse matrix-vector and matrix-transpose-vector
 * multiplication using compressed sparse blocks. ACM Symp. on Parallelism
 * in Algorithms and Architectures. CiteSeerX 10.1.1.211.5256.
 */
template <typename IDType, typename NNZType, typename ValueType>
class CSC
    : public utils::IdentifiableImplementation<CSC<IDType, NNZType, ValueType>,
        FormatOrderTwo<IDType, NNZType, ValueType>> {
 public:
  CSC(IDType n, IDType m, NNZType *col_ptr, IDType *col, ValueType *vals,
      Ownership own = kNotOwned, bool ignore_sort = false);
  CSC(const CSC<IDType, NNZType, ValueType> &);
  CSC(CSC<IDType, NNZType, ValueType> &&);
  CSC<IDType, NNZType, ValueType> &operator=(
      const CSC<IDType, NNZType, ValueType> &);
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
  std::unique_ptr<NNZType, std::function<void(NNZType *)>> col_ptr_;
  std::unique_ptr<IDType, std::function<void(IDType *)>> row_;
  std::unique_ptr<ValueType, std::function<void(ValueType *)>> vals_;
};

template <typename IDType, typename NNZType, typename ValueType>
template <typename ToIDType, typename ToNNZType, typename ToValueType>
struct format::FormatOrderTwo<IDType, NNZType, ValueType>::TypeConverter<
    CSR, ToIDType, ToNNZType, ToValueType> {
  CSR<ToIDType, ToNNZType, ToValueType> *operator()(
      FormatOrderTwo<IDType, NNZType, ValueType> *source,
      bool is_move_conversion) {
    CSR<IDType, NNZType, ValueType> *csr = source->template As<CSR>();
    auto dims = csr->get_dimensions();
    auto num_nnz = csr->get_num_nnz();
    ToNNZType *new_row_ptr;
    ToIDType *new_col;
    ToValueType *new_vals;

    if (!is_move_conversion || !std::is_same_v<NNZType, ToNNZType>) {
      new_row_ptr =
          utils::ConvertArrayType<ToNNZType>(csr->get_row_ptr(), dims[0] + 1);
    } else {
      if constexpr (std::is_same_v<NNZType, ToNNZType>) {
        new_row_ptr = csr->release_row_ptr();
      }
    }

    if (!is_move_conversion || !std::is_same_v<IDType, ToIDType>) {
      new_col = utils::ConvertArrayType<ToIDType>(csr->get_col(), num_nnz);
    } else {
      if constexpr (std::is_same_v<IDType, ToIDType>) {
        new_col = csr->release_col();
      }
    }
    if (!is_move_conversion || !std::is_same_v<ValueType, ToValueType>) {
      new_vals = utils::ConvertArrayType<ToValueType>(csr->get_vals(), num_nnz);
    } else {
      if constexpr (std::is_same_v<ValueType, ToValueType>) {
        new_vals = csr->release_vals();
      }
    }
    return new CSR<ToIDType, ToNNZType, ToValueType>(
        dims[0], dims[1], new_row_ptr, new_col, new_vals, kOwned);
  }
};
template <typename IDType, typename NNZType, typename ValueType>
template <typename ToIDType, typename ToNNZType, typename ToValueType>
struct format::FormatOrderTwo<IDType, NNZType, ValueType>::TypeConverter<
    CSC, ToIDType, ToNNZType, ToValueType> {
  CSC<ToIDType, ToNNZType, ToValueType> *operator()(
      FormatOrderTwo<IDType, NNZType, ValueType> *source,
      bool is_move_conversion) {
    CSC<IDType, NNZType, ValueType> *csc = source->template As<CSC>();
    auto dims = csc->get_dimensions();
    auto num_nnz = csc->get_num_nnz();
    ToNNZType *new_col_ptr;
    ToIDType *new_row;
    ToValueType *new_vals;

    if (!is_move_conversion || !std::is_same_v<NNZType, ToNNZType>) {
      new_col_ptr =
          utils::ConvertArrayType<ToNNZType>(csc->get_col_ptr(), dims[0] + 1);
    } else {
      if constexpr (std::is_same_v<NNZType, ToNNZType>)
      new_col_ptr = csc->release_col_ptr();
    }

    if (!is_move_conversion || !std::is_same_v<IDType, ToIDType>) {
      new_row = utils::ConvertArrayType<ToIDType>(csc->get_row(), num_nnz);
    } else {
      if constexpr (std::is_same_v<IDType, ToIDType>)
      new_row = csc->release_row();
    }
    if (!is_move_conversion || !std::is_same_v<ValueType, ToValueType>) {
      new_vals = utils::ConvertArrayType<ToValueType>(csc->get_vals(), num_nnz);
    } else {
      if constexpr (std::is_same_v<ValueType, ToValueType>)
      new_vals = csc->release_vals();
    }
    return new CSC<ToIDType, ToNNZType, ToValueType>(
        dims[0], dims[1], new_col_ptr, new_row, new_vals, kOwned);
  }
};
template <typename IDType, typename NNZType, typename ValueType>
template <typename ToIDType, typename ToNNZType, typename ToValueType>
struct format::FormatOrderTwo<IDType, NNZType, ValueType>::TypeConverter<
    COO, ToIDType, ToNNZType, ToValueType> {
  COO<ToIDType, ToNNZType, ToValueType> *operator()(
      FormatOrderTwo<IDType, NNZType, ValueType> *source,
      bool is_move_conversion) {
    COO<IDType, NNZType, ValueType> *coo = source->template As<COO>();
    auto dims = coo->get_dimensions();
    auto num_nnz = coo->get_num_nnz();
    ToIDType *new_col;
    ToIDType *new_row;
    ToValueType *new_vals;

    if (!is_move_conversion || !std::is_same_v<IDType, ToIDType>) {
      new_col = utils::ConvertArrayType<ToIDType>(coo->get_col(), num_nnz);
    } else {
      if constexpr (std::is_same_v<IDType, ToIDType>) {
        new_col = coo->release_col();
      }
    }

    if (!is_move_conversion || !std::is_same_v<IDType, ToIDType>) {
      new_row = utils::ConvertArrayType<ToIDType>(coo->get_row(), num_nnz);
    } else {
      if constexpr (std::is_same_v<IDType, ToIDType>) {
        new_row = coo->release_row();
      }
    }

    if (!is_move_conversion || !std::is_same_v<ValueType, ToValueType>) {
      new_vals = utils::ConvertArrayType<ToValueType>(coo->get_vals(), num_nnz);
    } else {
      if constexpr (std::is_same_v<ValueType, ToValueType>) {
        new_vals = coo->release_vals();
      }
    }
    return new COO<ToIDType, ToNNZType, ToValueType>(
        dims[0], dims[1], num_nnz, new_row, new_col, new_vals, kOwned);
  }
};
template <typename IDType, typename NNZType, typename ValueType>
template <template <typename, typename, typename> class ToType>
ToType<IDType, NNZType, ValueType>
*FormatOrderTwo<IDType, NNZType, ValueType>::Convert(
    context::Context *to_context, bool is_move_conversion) {
  static_assert(
      std::is_base_of<format::FormatOrderTwo<IDType, NNZType, ValueType>,
          ToType<IDType, NNZType, ValueType>>::value,
      "T must be an order two format");
  //converter::Converter* converter = this->converter_.get();
  auto converter = this->get_converter();
  context::Context *actual_context =
      to_context == nullptr ? this->get_context() : to_context;
  return converter
      ->Convert(this, ToType<IDType, NNZType, ValueType>::get_id_static(),
                actual_context, is_move_conversion)
      ->template AsAbsolute<ToType<IDType, NNZType, ValueType>>();
}

template <typename IDType, typename NNZType, typename ValueType>
template <template <typename, typename, typename> class ToType>
ToType<IDType, NNZType, ValueType>
*FormatOrderTwo<IDType, NNZType, ValueType>::Convert(
    const std::vector<context::Context *> &to_contexts,
    bool is_move_conversion) {
  static_assert(
      std::is_base_of<format::FormatOrderTwo<IDType, NNZType, ValueType>,
          ToType<IDType, NNZType, ValueType>>::value,
      "T must be an order two format");
  //auto* converter = this->converter_.get();
  auto converter = this->get_converter();
  std::vector<context::Context *> vec = {this->get_context()};
  std::vector<context::Context *> actual_contexts =
      to_contexts.empty() ? vec : to_contexts;
  return converter
      ->Convert(this, ToType<IDType, NNZType, ValueType>::get_id_static(),
                actual_contexts, is_move_conversion)
      ->template AsAbsolute<ToType<IDType, NNZType, ValueType>>();
}

template <typename IDType, typename NNZType, typename ValueType>
template <template <typename, typename, typename> class ToType,
    typename ToIDType, typename ToNNZType, typename ToValueType>
ToType<ToIDType, ToNNZType, ToValueType> *
FormatOrderTwo<IDType, NNZType, ValueType>::Convert(bool is_move_conversion) {
  static_assert(
      std::is_base_of<format::FormatOrderTwo<ToIDType, ToNNZType, ToValueType>,
          ToType<ToIDType, ToNNZType, ToValueType>>::value,
      "T must be an order two format");
  //auto* converter = this->converter_.get();
  auto converter = this->get_converter();
  if (this->get_id() !=
      ToType<IDType, NNZType, ValueType>::get_id_static()) {
    auto converted_format =
    converter->template Convert<ToType<IDType, NNZType, ValueType>>(
        this, this->get_context(), is_move_conversion);
    auto type_converted_format =
        TypeConverter<ToType, ToIDType, ToNNZType, ToValueType>()(
            converted_format, is_move_conversion);
    delete converted_format;
    return type_converted_format;
  } else {
    return TypeConverter<ToType, ToIDType, ToNNZType, ToValueType>()(
        this, is_move_conversion);
  }
}
}
#ifdef _HEADER_ONLY
#include "format_order_two.cc"
#endif
#endif  // SPARSEBASE_PROJECT_FORMAT_ORDER_TWO_H
