#include "sparsebase/partition/partitioner.h"

#include <algorithm>
#include <map>
#include <memory>
#include <queue>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

#include "sparsebase/converter/converter.h"
#include "sparsebase/format/array.h"
#include "sparsebase/format/coo.h"
#include "sparsebase/format/csc.h"
#include "sparsebase/format/csr.h"
#include "sparsebase/format/format.h"
#include "sparsebase/format/format_order_one.h"
#include "sparsebase/format/format_order_two.h"
#include "sparsebase/utils/extractable.h"
#include "sparsebase/utils/function_matcher_mixin.h"
#include "sparsebase/utils/logger.h"
#include "sparsebase/utils/parameterizable.h"
namespace sparsebase::partition {
template <typename IDType>
Partitioner<IDType>::Partitioner() = default;

template <typename IDType>
IDType *Partitioner<IDType>::Partition(format::Format *format,
                                       std::vector<context::Context *> contexts,
                                       bool convert_input) {
  return this->Execute(this->params_.get(), contexts, convert_input, format);
}

template <typename IDType>
IDType *Partitioner<IDType>::Partition(format::Format *format,
                                       utils::Parameters *params,
                                       std::vector<context::Context *> contexts,
                                       bool convert_input) {
  return this->Execute(params, contexts, convert_input, format);
}

template <typename IDType>
Partitioner<IDType>::~Partitioner() = default;

#if !defined(_HEADER_ONLY)
#include "init/partitioner.inc"
#endif
}  // namespace sparsebase::partition
