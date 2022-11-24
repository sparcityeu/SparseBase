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
#include "sparsebase/format/format.h"
#include "sparsebase/format/format_order_one.h"
#include "sparsebase/format/format_order_two.h"

#include "sparsebase/converter/converter.h"
#include "sparsebase/utils/function_matcher_mixin.h"
#include "sparsebase/utils/extractable.h"
#ifndef SPARSEBASE_PROJECT_PARTITIONER_H
#define SPARSEBASE_PROJECT_PARTITIONER_H

namespace sparsebase::partition {

//! An abstract class representing partitioning algorithms.
/*!
 * Class that generalizes partitioning algorithms. It defines the API used for
 * partitioning as well as the return type of partitioning (IDType*).
 * @tparam IDType  the data type of row and column numbers (vertex IDs in the
 * case of graphs)
 */
template <typename IDType>
class Partitioner : public utils::FunctionMatcherMixin<IDType *> {
 public:
  Partitioner();

  //! Performs a partition operation using the default parameters
  /*!
   * @param format the Format object that will be reordered.
   * @param contexts vector of contexts that can be used for generating the
   * partitioning.
   * @param convert_input whether or not to convert the input format if that is
   * needed.
   * @returns An IDType array where the i-th index contains the ID for the
   * partitioning i belongs to.
   */
  IDType *Partition(format::Format *format,
                    std::vector<context::Context *> contexts,
                    bool convert_input);

  //! Performs a partition operation using the parameters supplied by the user
  /*!
   * @param format the Format object that will be reordered.
   * @param contexts vector of contexts that can be used for generating the
   * partitioning.
   * @param convert_input whether or not to convert the input format if that is
   * needed.
   * @returns An IDType array where the i-th index contains the ID for the
   * partitioning i belongs to
   */
  IDType *Partition(format::Format *format, utils::Parameters *params,
                    std::vector<context::Context *> contexts,
                    bool convert_input);
  virtual ~Partitioner();
};


}  // namespace sparsebase::reorder
#ifdef _HEADER_ONLY
#include "sparsebase/partition/partitioner.cc"
#endif
#endif  // SPARSEBASE_PROJECT_PARTITIONER_H
