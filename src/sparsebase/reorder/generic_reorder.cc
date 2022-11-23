#include "sparsebase/reorder/generic_reorder.h"
#include "sparsebase/reorder/reorder.h"

namespace sparsebase::reorder {
template <typename IDType, typename NNZType, typename ValueType>
GenericReorder<IDType, NNZType, ValueType>::GenericReorder() {
}
#if !defined(_HEADER_ONLY)
#include "init/generic_reorder.inc"
#endif
}
