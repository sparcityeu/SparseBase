#include <utility>
#include <vector>

#include "sparsebase/config.h"
#include "sparsebase/format/csr.h"
#include "sparsebase/reorder/reorderer.h"
#include "sparsebase/utils/parameterizable.h"
#ifndef SPARSEBASE_PROJECT_RABBIT_ORDER_H
#define SPARSEBASE_PROJECT_RABBIT_ORDER_H

namespace sparsebase::reorder {

#ifdef USE_RABBIT_ORDER

#define BOOST_ATOMIC_DETAIL_NO_CXX11_IS_TRIVIALLY_COPYABLE
#define BOOST_ATOMIC_DETAIL_NO_HAS_UNIQUE_OBJECT_REPRESENTATIONS
#define BOOST_ATOMIC_NO_CLEAR_PADDING

struct RabbitReorderParams : utils::Parameters {};

template <typename IDType, typename NNZType, typename ValueType>
class RabbitReorder : public Reorderer<IDType> {
 public:
  typedef RabbitReorderParams ParamsType;
  RabbitReorder();
  explicit RabbitReorder(RabbitReorderParams);

 protected:
  static IDType *CalculateReorderCSR(std::vector<format::Format *> formats,
                                     utils::Parameters *params);
};

#endif

}  // namespace sparsebase::reorder
#ifdef _HEADER_ONLY
#include "sparsebase/reorder/rabbit_reorder.cc"
#endif
#endif  // SPARSEBASE_PROJECT_RABBIT_ORDER_H
