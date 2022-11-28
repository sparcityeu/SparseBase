#include "sparsebase/config.h"
#include "sparsebase/context/cpu_context.h"
#include "sparsebase/format/format_order_two.h"
#include "sparsebase/utils/exception.h"
#include "sparsebase/utils/utils.h"
#ifndef SPARSEBASE_PROJECT_CSC_H
#define SPARSEBASE_PROJECT_CSC_H

namespace sparsebase::format {

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
class CSC : public utils::IdentifiableImplementation<
                CSC<IDType, NNZType, ValueType>,
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
}  // namespace sparsebase::format

#ifdef _HEADER_ONLY
#include "csc.cc"
#endif
#endif  // SPARSEBASE_PROJECT_CSC_H
