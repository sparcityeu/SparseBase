#include "sparsebase/reorder/degree_reorder.h"

#include "sparsebase/format/csr.h"
#include "sparsebase/reorder/degree_reorder.h"

namespace sparsebase::reorder {

template <typename IDType, typename NNZType, typename ValueType>
DegreeReorder<IDType, NNZType, ValueType>::DegreeReorder(
    DegreeReorderParams params)
    : DegreeReorder(params.ascending) {}
template <typename IDType, typename NNZType, typename ValueType>
DegreeReorder<IDType, NNZType, ValueType>::DegreeReorder(bool ascending) {
  // this->map[{kCSRFormat}]= calculate_order_csr;
  // this->RegisterFunction({kCSRFormat}, CalculateReorderCSR);
  this->RegisterFunction(
      {format::CSR<IDType, NNZType, ValueType>::get_id_static()},
      CalculateReorderCSR);
  this->params_ = std::make_unique<DegreeReorderParams>(ascending);
}
template <typename IDType, typename NNZType, typename ValueType>
IDType *DegreeReorder<IDType, NNZType, ValueType>::CalculateReorderCSR(
    std::vector<format::Format *> formats, utils::Parameters *params) {
  format::CSR<IDType, NNZType, ValueType> *csr =
      formats[0]->AsAbsolute<format::CSR<IDType, NNZType, ValueType>>();
  DegreeReorderParams *cast_params = static_cast<DegreeReorderParams *>(params);
  bool ascending = cast_params->ascending;
  IDType n = csr->get_dimensions()[0];
  IDType *counts = new IDType[n + 1]();
  auto row_ptr = csr->get_row_ptr();
  auto col = csr->get_col();

  for (IDType u = 0; u < n; u++) {
    counts[row_ptr[u + 1] - row_ptr[u]]++;
  }
  for (IDType u = 1; u < n + 1; u++) {
    counts[u] += counts[u - 1];
  }
  IDType *sorted = new IDType[n];
  memset(sorted, -1, sizeof(IDType) * n);
  IDType *mr = new IDType[n]();
  for (IDType u = 0; u < n; u++) {
    IDType ec = counts[row_ptr[u + 1] - row_ptr[u]];
    sorted[ec - mr[ec] - 1] = u;
    mr[ec]++;
  }
  if (!ascending) {
    for (IDType i = 0; i < n / 2; i++) {
      IDType swp = sorted[i];
      sorted[i] = sorted[n - i - 1];
      sorted[n - i - 1] = swp;
    }
  }
  auto *inverse_permutation = new IDType[n];
  for (IDType i = 0; i < n; i++) {
    inverse_permutation[sorted[i]] = i;
  }
  delete[] mr;
  delete[] counts;
  delete[] sorted;
  return inverse_permutation;
}
#if !defined(_HEADER_ONLY)
#include "init/degree_reorder.inc"
#endif
}  // namespace sparsebase::reorder