#include "sparsebase/config.h"
#include "sparsebase/utils/parameterizable.h"
#include "sparsebase/permute/permuter.h"
#include "sparsebase/format/array.h"
#include <vector>
#ifndef SPARSEBASE_PROJECT_PERMUTE_ORDER_TWO_H
#define SPARSEBASE_PROJECT_PERMUTE_ORDER_TWO_H
namespace sparsebase::permute {
//! The hyperparameters of the PermuteOrderTwo transformation.
/*!
 * The permutation vectors used for permuting the rows and the columns of a 2D
 * format.
 * @tparam IDType the data type of row and column numbers (vertex IDs in the
 */
template <typename IDType>
struct PermuteOrderTwoParams : utils::Parameters {
  //! Permutation vector for reordering the rows.
  IDType *row_order;
  //! Permutation vector for reordering the columns.
  IDType *col_order;
  explicit PermuteOrderTwoParams(IDType *r_order, IDType *c_order)
      : row_order(r_order), col_order(c_order){};
};
template <typename IDType, typename NNZType, typename ValueType>
class PermuteOrderTwo
    : public Permuter<
        format::FormatOrderTwo<IDType, NNZType, ValueType>,
      format::FormatOrderTwo<IDType, NNZType, ValueType>> {
public:
PermuteOrderTwo(IDType *, IDType *);
explicit PermuteOrderTwo(PermuteOrderTwoParams<IDType>);
//! Struct used to store permutation vectors used by each instance of
//! PermuteOrderTwo
typedef PermuteOrderTwoParams<IDType> ParamsType;

protected:
//! An implementation function that will transform a CSR format into another
//! CSR
/*!
 *
 * @param formats a vector containing a single Format object of type CSR
 * @param params a polymorphic pointer at a `TransformParams` object
 * @return a transformed Format object of type CSR
 */
static format::FormatOrderTwo<IDType, NNZType, ValueType> *PermuteOrderTwoCSR(
    std::vector<format::Format *> formats, utils::Parameters *);
};


}
#ifdef _HEADER_ONLY
#include "sparsebase/permute/permute_order_two.cc"
#endif

#endif  // SPARSEBASE_PROJECT_PERMUTE_ORDER_TWO_H
