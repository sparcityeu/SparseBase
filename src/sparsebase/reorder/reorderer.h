#include <any>
#include <cmath>
#include <iostream>
#include <memory>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>
#include <vector>

#include "sparsebase/config.h"
#include "sparsebase/context/cpu_context.h"
#include "sparsebase/converter/converter.h"
#include "sparsebase/format/format.h"
#include "sparsebase/format/format_order_one.h"
#include "sparsebase/format/format_order_two.h"
#include "sparsebase/utils/extractable.h"
#include "sparsebase/utils/function_matcher_mixin.h"
#ifndef SPARSEBASE_PROJECT_REORDERING_H
#define SPARSEBASE_PROJECT_REORDERING_H

#ifdef USE_METIS
namespace sparsebase::metis {
#include <metis.h>
}
#endif

namespace sparsebase::reorder {

//! An abstract class representing reordering algorithms.
/*!
 * Class that generalizes reordering algorithms. It defines the API used for
 * reordering as well as the return type of reordering (IDType*).
 * @tparam IDType  the data type of row and column numbers (vertex IDs in the
 * case of graphs)
 */
template <typename IDType>
class Reorderer : public utils::FunctionMatcherMixin<IDType *> {
 protected:
 public:
  //! Generates a reordering inverse permutation of `format` using one of the
  //! contexts in `contexts`
  /*!
   *
   * @param format the Format object that will be reordered.
   * @param contexts vector of contexts that can be used for generating the
   * reordering.
   * @param convert_input whether or not to convert the input format if that is
   * needed.
   * @return the inverse permutation array `inv_perm` of the input format; an
   * array of size `format.get_dimensions()[0]` where `inv_per[i]` is the new ID
   * of row/column `i`.
   */
  IDType *GetReorder(format::Format *format,
                     std::vector<context::Context *> contexts,
                     bool convert_input);
  //! Generates a reordering inverse permutation of `format` with the given
  //! Parameters object and using one of the contexts in `contexts`
  /*!
   *
   * @param format the Format object that will be reordered.
   * @param params a polymorphic pointer at a `Parameters` object that
   * will contain hyperparameters used for reordering.
   * @param contexts vector of contexts that can be used for generating the
   * reordering.
   * @param convert_input whether or not to convert the input format if that is
   * needed.
   * @return the inverse permutation array `inv_perm` of the input format; an
   * array of size `format.get_dimensions()[0]` where `inv_per[i]` is the new ID
   * of row/column `i`.
   */
  IDType *GetReorder(format::Format *format, utils::Parameters *params,
                     std::vector<context::Context *> contexts,
                     bool convert_input);
  //! Generates a reordering using one of the contexts in `contexts`, and caches
  //! intermediate `Format` objects.
  /*!
   *
   * @param format the Format object that will be reordered.
   * @param contexts vector of contexts that can be used for generating the
   * reordering.
   * @param convert_input whether or not to convert the input format if that is
   * needed.
   * @return A tuple with the first element being a vector of Format*, where
   * each pointer in the output points at the format that the corresponds Format
   * object from the the input was converted to. If an input Format wasn't
   * converted, the output pointer will point at nullptr. The second element is
   * the inverse permutation array `inv_perm` of the input format; an array of
   * size `format.get_dimensions()[0]` where `inv_per[i]` is the new ID of
   * row/column `i`.
   */
  std::tuple<std::vector<std::vector<format::Format *>>, IDType *>
  GetReorderCached(format::Format *csr,
                   std::vector<context::Context *> contexts,
                   bool convert_input);
  //! Generates a reordering inverse permutation of `format` with the given
  //! Parameters object and using one of the contexts in `contexts`
  /*!
   *
   * @param format the Format object that will be reordered.
   * @param params a polymorphic pointer at a `Parameters` object that
   * will contain hyperparameters used for reordering.
   * @param contexts vector of contexts that can be used for generating the
   * reordering.
   * @param convert_input whether or not to convert the input format if that is
   * needed.
   * @return A tuple with the first element being a vector of Format*, where
   * each pointer in the output points at the format that the corresponds Format
   * object from the the input was converted to. If an input Format wasn't
   * converted, the output pointer will point at nullptr. The second element is
   * the inverse permutation array `inv_perm` of the input format; an array of
   * size `format.get_dimensions()[0]` where `inv_per[i]` is the new ID of
   * row/column `i`.
   */
  std::tuple<std::vector<std::vector<format::Format *>>, IDType *>
  GetReorderCached(format::Format *csr, utils::Parameters *params,
                   std::vector<context::Context *> contexts,
                   bool convert_input);
  virtual ~Reorderer();
};

}  // namespace sparsebase::reorder
#ifdef _HEADER_ONLY
#include "sparsebase/reorder/reorderer.cc"
#endif

#endif  // SPARSEBASE_PROJECT_REORDERING_H
