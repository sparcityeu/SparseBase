#include "sparsebase/reorder/rabbit_reorder.h"

#include "sparsebase/reorder/reorderer.h"

#ifdef USE_RABBIT_ORDER
#include "rabbit_order.hpp"
#endif
namespace sparsebase::reorder {
#ifdef USE_RABBIT_ORDER

template <typename IDType, typename NNZType, typename ValueType>
RabbitReorder<IDType, NNZType, ValueType>::RabbitReorder() {
  this->RegisterFunction(
      {format::CSR<IDType, NNZType, ValueType>::get_id_static()},
      CalculateReorderCSR);
  this->params_ = std::make_unique<RabbitReorderParams>();
}

template <typename IDType, typename NNZType, typename ValueType>
RabbitReorder<IDType, NNZType, ValueType>::RabbitReorder(
    RabbitReorderParams params)
    : RabbitReorder() {}

template <typename IDType, typename NNZType, typename ValueType>
IDType *RabbitReorder<IDType, NNZType, ValueType>::CalculateReorderCSR(
    std::vector<format::Format *> formats, utils::Parameters *params) {
  using rabbit_order::vint;
  typedef std::vector<std::vector<std::pair<vint, float>>> adjacency_list;

  format::CSR<IDType, NNZType, ValueType> *csr =
      formats[0]->AsAbsolute<format::CSR<IDType, NNZType, ValueType>>();
  IDType n = csr->get_dimensions()[0];
  IDType *counts = new IDType[n]();
  auto *idx = csr->get_row_ptr();
  auto *adj = csr->get_col();

  adjacency_list G(n);
  for (size_t i = 0; i < n; i++) {
    for (size_t j = idx[i]; j < idx[i + 1]; j++) {
      vint source = i;
      vint target = adj[j];
      G[source].push_back(std::make_pair((vint)target, 1.0f));
    }
  }
  const auto g = rabbit_order::aggregate(std::move(G));
  const auto p = rabbit_order::compute_perm(g);
  IDType *ptr = new IDType[n];
  std::copy(p.get(), p.get() + n, ptr);
  return ptr;
}

#endif

#if !defined(_HEADER_ONLY)
#include "init/rabbit_reorder.inc"
#endif
}  // namespace sparsebase::reorder
