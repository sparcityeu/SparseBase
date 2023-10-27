#include "sparsebase/reorder/boba_reorder.h"

#include "sparsebase/reorder/reorderer.h"
#include "sparsebase/utils/logger.h"

#include "sparsebase/format/coo.h"
#include <unordered_set>

namespace sparsebase::reorder {
template <typename IDType, typename NNZType, typename ValueType>
BOBAReorder<IDType, NNZType, ValueType>::BOBAReorder() {
  this->RegisterFunction(
      {format::COO<IDType, NNZType, ValueType>::get_id_static()},
      GetReorderCOO);
}
template <typename IDType, typename NNZType, typename ValueType>
BOBAReorder<IDType, NNZType, ValueType>::BOBAReorder(BOBAReorderParams p) {
  this->RegisterFunction(
      {format::COO<IDType, NNZType, ValueType>::get_id_static()},
      GetReorderCOO);
}

template <typename IDType, typename NNZType, typename ValueType>
IDType *BOBAReorder<IDType, NNZType, ValueType>::GetReorderCOO(
    std::vector<format::Format *> formats, utils::Parameters *params) {
  format::COO<IDType, NNZType, ValueType> *coo =
      formats[0]->AsAbsolute<format::COO<IDType, NNZType, ValueType>>();

  context::CPUContext *cpu_context =
      static_cast<context::CPUContext *>(coo->get_context());

  IDType nodes = 0;

  if (coo->get_dimensions()[0] >= coo->get_dimensions()[1]) {
    nodes = coo->get_dimensions()[0];
  } else {
    nodes = coo->get_dimensions()[1];
  } 

  IDType *order = new IDType[nodes]();
  IDType *order2 = new IDType[nodes];
  auto coo_col = coo->get_col();
  auto coo_row = coo->get_row();

  int k=0;

  /* Goes through the values in the I array (rows) */
  std::unordered_set<IDType> copiedValues;
  for (int i = 0; i < coo->get_num_nnz(); i++) {
    IDType element_i = coo_row[i];
    if (copiedValues.find(element_i) == copiedValues.end()) {
      order[k++] = element_i;
      copiedValues.insert(element_i);
    }
  }
  if (k == nodes) {
    for (IDType i = 0; i < nodes; i++) {
      order2[order[i]] = i;
    }
    return order2;
  }

  /* Goes through the values in the J array (columns) */
  for (int i = 0; i < coo->get_num_nnz(); i++) {
    IDType element_j = coo_col[i];
    if (copiedValues.find(element_j) == copiedValues.end()) {
      order[k++] = element_j;
      copiedValues.insert(element_j);
    }
  }
  if (k == nodes) {
    for (IDType i = 0; i < nodes; i++) {
      order2[order[i]] = i;
    }
    return order2;
  }

  /* Goes through the single nodes that don't have any edge */
  for (IDType i = 0; i < nodes; i++) {
    if (copiedValues.find(i) == copiedValues.end()) {
      order[k++] = i;
      copiedValues.insert(i);
    }
    order2[order[i]] = i;
  }
  return order2;
}

#if !defined(_HEADER_ONLY)
#include "init/boba_reorder.inc"
#endif
}  // namespace sparsebase::reorder
