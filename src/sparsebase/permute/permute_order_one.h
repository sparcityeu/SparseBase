#include "sparsebase/config.h"
#include "sparsebase/utils/parameterizable.h"
#include "sparsebase/permute/permuter.h"
#include "sparsebase/format/array.h"
#include <vector>

#ifndef SPARSEBASE_PROJECT_PERMUTE_ORDER_ONE_H
#define SPARSEBASE_PROJECT_PERMUTE_ORDER_ONE_H
namespace sparsebase::permute {

//! The hyperparameters of the PermuteOrderTwo transformation.
/*!
 * The permutation vectors used for permuting the rows and the columns of a 2D
 * format.
 * @tparam IDType the data type of row and column numbers (vertex IDs in the
 */
template <typename IDType>
struct PermuteOrderOneParams : utils::Parameters {
  //! Permutation vector
  IDType *order;
  explicit PermuteOrderOneParams(IDType *order) : order(order){};
};
template <typename IDType, typename ValueType>
class PermuteOrderOne
    : public Permuter<format::FormatOrderOne<ValueType>,
      format::FormatOrderOne<ValueType>> {
public:
PermuteOrderOne(IDType *);
//! Struct used to store permutation vectors used by each instance of
//! PermuteOrderTwo
typedef PermuteOrderOneParams<IDType> ParamsType;
explicit PermuteOrderOne(ParamsType);

protected:
//! An implementation function that will transform a CSR format into another
//! CSR
/*!
 *
 * @param formats a vector containing a single Format object of type Array
 * @param params a polymorphic pointer at a `TransformParams` object
 * @return a transformed Format object of type CSR
 */
static format::FormatOrderOne<ValueType> *PermuteArray(
    std::vector<format::Format *> formats, utils::Parameters *);
};


}
#ifdef _HEADER_ONLY
#include "sparsebase/permute/permute_order_one.cc"
#endif

#endif  // SPARSEBASE_PROJECT_PERMUTE_ORDER_ONE_H
