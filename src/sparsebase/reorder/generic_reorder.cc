#include "sparsebase/reorder/generic_reorder.h"

#include "sparsebase/reorder/reorderer.h"

namespace sparsebase::reorder {
template <typename IDType, typename NNZType, typename ValueType>
GenericReorder<IDType, NNZType, ValueType>::GenericReorder() {}
#if !defined(_HEADER_ONLY)
#include "init/generic_reorder.inc"
#endif
}  // namespace sparsebase::reorder
