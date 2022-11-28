#include <vector>

#include "sparsebase/config.h"
#include "sparsebase/format/csr.h"
#include "sparsebase/reorder/reorderer.h"
#include "sparsebase/utils/parameterizable.h"

#ifndef SPARSEBASE_PROJECT_AMD_REORDER_H
#define SPARSEBASE_PROJECT_AMD_REORDER_H
namespace sparsebase::reorder {

#ifdef USE_AMD_ORDER
#ifdef __cplusplus
extern "C" {
#endif
#include <amd.h>
#ifdef __cplusplus
}
#endif

//! Parameters for AMDReordering
/*!
 * For the exact definitions, please consult the documentation given with the
 * code, which is available here:
 * https://dl.acm.org/doi/abs/10.1145/1024074.1024081
 */
struct AMDReorderParams : utils::Parameters {
  double dense = AMD_DEFAULT_DENSE;
  double aggressive = AMD_DEFAULT_AGGRESSIVE;
};

//! A wrapper for the AMD reordering algorithm
/*!
 * Wraps the AMD reordering algorithm library available here as supplemental
 * material: https://dl.acm.org/doi/abs/10.1145/1024074.1024081 The library must
 * be compiled with the USE_AMD_ORDER option turned on and the pre-built AMD
 * library should be available. See the Optional Dependencies page (under
 * Getting Started) in our documentation for more info.
 */
template <typename IDType, typename NNZType, typename ValueType>
class AMDReorder : public Reorderer<IDType> {
 public:
  typedef AMDReorderParams ParamsType;
  AMDReorder(ParamsType);
  AMDReorder();

 protected:
  static IDType* AMDReorderCSR(std::vector<format::Format*>,
                               utils::Parameters*);
};
#endif
}  // namespace sparsebase::reorder
#ifdef _HEADER_ONLY
#include "sparsebase/reorder/amd_reorder.cc"
#endif

#endif  // SPARSEBASE_PROJECT_AMD_REORDER_H
