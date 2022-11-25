#include "sparsebase/reorder/reorderer.h"

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
namespace sparsebase::reorder {

template <typename IDType>
Reorderer<IDType>::~Reorderer() = default;
;

template <typename IDType>
IDType *Reorderer<IDType>::GetReorder(format::Format *format,
                                      std::vector<context::Context *> contexts,
                                      bool convert_input) {
  return this->Execute(this->params_.get(), contexts, convert_input, format);
}

template <typename IDType>
IDType *Reorderer<IDType>::GetReorder(format::Format *format,
                                      utils::Parameters *params,
                                      std::vector<context::Context *> contexts,
                                      bool convert_input) {
  return this->Execute(params, contexts, convert_input, format);
}

template <typename IDType>
std::tuple<std::vector<std::vector<format::Format *>>, IDType *>
Reorderer<IDType>::GetReorderCached(format::Format *format,
                                    std::vector<context::Context *> contexts,
                                    bool convert_input) {
  return this->CachedExecute(this->params_.get(), contexts, convert_input,
                             false, format);
}

template <typename IDType>
std::tuple<std::vector<std::vector<format::Format *>>, IDType *>
Reorderer<IDType>::GetReorderCached(format::Format *format,
                                    utils::Parameters *params,
                                    std::vector<context::Context *> contexts,
                                    bool convert_input) {
  return this->CachedExecute(params, contexts, convert_input, false, format);
}

#if !defined(_HEADER_ONLY)
#include "init/reorderer.inc"
#endif
}  // namespace sparsebase::reorder