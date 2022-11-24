#include "sparsebase/config.h"
#include "sparsebase/context/cpu_context.h"
#include "sparsebase/utils/exception.h"
#include "sparsebase/utils/utils.h"

#include "sparsebase/format/format_order_two.h"
#ifndef SPARSEBASE_PROJECT_COO_H
#define SPARSEBASE_PROJECT_COO_H
namespace sparsebase::format {

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
}

#ifdef _HEADER_ONLY
#include "coo.cc"
#endif
#endif  // SPARSEBASE_PROJECT_COO_H
