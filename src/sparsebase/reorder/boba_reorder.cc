#include "sparsebase/reorder/boba_reorder.h"

#include "sparsebase/reorder/reorderer.h"
#include "sparsebase/utils/logger.h"

#include "sparsebase/format/coo.h"
#include <unordered_set>
#include <queue>
#include <stack>

namespace sparsebase::reorder {
template <typename IDType, typename NNZType, typename ValueType>
BOBAReorder<IDType, NNZType, ValueType>::BOBAReorder(bool sequential) {

  auto params_struct = new BOBAReorderParams;
  params_struct->sequential = sequential;
  this->params_ = std::unique_ptr<BOBAReorderParams>(params_struct);

  this->RegisterFunction(
      {format::COO<IDType, NNZType, ValueType>::get_id_static()},
      GetReorderCOO);
}
template <typename IDType, typename NNZType, typename ValueType>
BOBAReorder<IDType, NNZType, ValueType>::BOBAReorder(BOBAReorderParams p)
    : BOBAReorder(p.sequential) {}
/*{
  this->RegisterFunction(
      {format::COO<IDType, NNZType, ValueType>::get_id_static()},
      GetReorderCOO);
}*/

template <typename IDType, typename NNZType, typename ValueType>
IDType *BOBAReorder<IDType, NNZType, ValueType>::GetReorderCOO(
    std::vector<format::Format *> formats, utils::Parameters *params) {
  format::COO<IDType, NNZType, ValueType> *coo =
      formats[0]->AsAbsolute<format::COO<IDType, NNZType, ValueType>>();

  context::CPUContext *cpu_context =
      static_cast<context::CPUContext *>(coo->get_context());

  BOBAReorderParams *params_ = static_cast<BOBAReorderParams *>(params);

  IDType nodes = 0;
  IDType nnzs = coo->get_num_nnz();

  if (coo->get_dimensions()[0] >= coo->get_dimensions()[1]) {
    nodes = coo->get_dimensions()[0];
  } else {
    nodes = coo->get_dimensions()[1];
  } 

  IDType *order = new IDType[nodes]();
  IDType *order2 = new IDType[nodes];
  auto coo_col = coo->get_col();
  auto coo_row = coo->get_row();

  int k = 0;

  std::vector<std::pair<int, int>> cooMatrix(nnzs);
  for (int i = 0; i < nnzs; i++) {
    cooMatrix[i] = std::make_pair(coo_row[i], coo_col[i]);
  }

  /* Sort the matrix based on columns */
  std::sort(cooMatrix.begin(), cooMatrix.end(), [](const auto& a, const auto& b) {
    return (a.second != b.second) ? (a.second < b.second) : (a.first < b.first);
  });

  if (params_->sequential) {

    /* Goes through the values in the I array (rows) */
    std::unordered_set<IDType> copiedValues;
    for (int i = 0; i < nnzs; i++) {
      IDType element_i = cooMatrix[i].first;
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
    for (int i = 0; i < nnzs; i++) {
      IDType element_j = cooMatrix[i].second;
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
  } else {

    std::priority_queue<std::pair<IDType, IDType>, std::vector<std::pair<IDType, IDType>>, std::greater<std::pair<IDType, IDType>>> PQ;
    for (IDType i = 0; i < nodes; i++) {
      order[i] = nnzs*2;
    }

    /* Goes through the values in the I++J array */
    #pragma omp parallel for
    for (IDType i = 0; i < nnzs*2; i++) {
      if ((i < nnzs) && (i < order[cooMatrix[i].first]))
        order[cooMatrix[i].first] = i;
      else if ((i >= nnzs) && (i < order[cooMatrix[i - nnzs].second]))
        order[cooMatrix[i - nnzs].second] = i; 
    }

    /* Places vertices by order of appearence */
    for (IDType i = 0; i < nodes; i++) {
      PQ.push(std::make_pair(order[i], i));
    }
    for (IDType i = 0; i < nodes; i++) {
      order2[PQ.top().second] = i;
      PQ.pop();
    }
    return order2;
  }
}

#if !defined(_HEADER_ONLY)
#include "init/boba_reorder.inc"
#endif
}  // namespace sparsebase::reorder
