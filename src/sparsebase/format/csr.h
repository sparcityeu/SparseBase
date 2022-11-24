#include "sparsebase/config.h"
#include "sparsebase/context/cpu_context.h"
#include "sparsebase/utils/exception.h"
#include "sparsebase/utils/utils.h"

#include "sparsebase/format/format_order_two.h"
#ifndef SPARSEBASE_PROJECT_CSR_H
#define SPARSEBASE_PROJECT_CSR_H

namespace sparsebase::format {

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
}

#ifdef _HEADER_ONLY
#include "csr.cc"
#endif

#endif  // SPARSEBASE_PROJECT_CSR_H
