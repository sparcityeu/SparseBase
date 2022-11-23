#include <vector>

#include "sparsebase/config.h"
#include "sparsebase/format/csr.h"
#include "sparsebase/reorder/reorderer.h"
#include "sparsebase/utils/parameterizable.h"
#ifndef SPARSEBASE_PROJECT_GENERIC_REORDER_H
#define SPARSEBASE_PROJECT_GENERIC_REORDER_H

namespace sparsebase::reorder {
//! A generic reordering class that the user instantiate and then register their
//! own functions to.
template <typename IDType, typename NNZType, typename ValueType>
class GenericReorder : public Reorderer<IDType> {
 public:
  typedef utils::Parameters ParamsType;
  GenericReorder();
};
}
#ifdef _HEADER_ONLY
#include "sparsebase/reorder/generic_reorder.cc"
#endif

#endif  // SPARSEBASE_PROJECT_GENERIC_REORDER_H
