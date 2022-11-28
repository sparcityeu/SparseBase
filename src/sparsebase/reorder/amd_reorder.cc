#include "sparsebase/reorder/amd_reorder.h"

#include "sparsebase/format/csr.h"

namespace sparsebase::reorder {
#ifdef USE_AMD_ORDER
#ifdef __cplusplus
extern "C" {
#endif
#include <amd.h>
#ifdef __cplusplus
}
#endif

template <typename IDType, typename NNZType, typename ValueType>
AMDReorder<IDType, NNZType, ValueType>::AMDReorder(AMDReorderParams p)
    : AMDReorder() {
  this->params_ = std::make_unique<AMDReorderParams>(p);
}
template <typename IDType, typename NNZType, typename ValueType>
AMDReorder<IDType, NNZType, ValueType>::AMDReorder() {
  this->params_ = std::make_unique<AMDReorderParams>();
  this->RegisterFunction(
      {format::CSR<IDType, NNZType, ValueType>::get_id_static()},
      AMDReorderCSR);
}

template <typename IDType, typename NNZType, typename ValueType>
IDType* AMDReorder<IDType, NNZType, ValueType>::AMDReorderCSR(
    std::vector<format::Format*> formats, utils::Parameters* p) {
  auto csr = formats[0]->AsAbsolute<format::CSR<IDType, NNZType, ValueType>>();
  format::CSR<long, long, ValueType>* csr_converted;
  try {
    csr_converted =
        csr->template Convert<format::CSR, long, long, ValueType>(false);
  } catch (const utils::TypeException& exception) {
    throw utils::TypeException(
        "AMD reordering requires IDType=long and NNZType=long");
  }
  auto params = static_cast<AMDReorderParams*>(p);
  long long n = csr->get_dimensions()[0];
  long *xadj_long = csr_converted->get_row_ptr(),
       *adj_long = csr_converted->get_col();
  long* i_order = new long[n];
  double* Control = new double[AMD_CONTROL];
  Control[AMD_DENSE] = params->dense;
  Control[AMD_AGGRESSIVE] = params->aggressive;
  double* Info = nullptr;  // new double[AMD_INFO]; // Auxiliary data
  int status = amd_l_order(n, xadj_long, adj_long, i_order, Control, Info);
  if (status != 0) {
    throw utils::ReorderException("AMD reordering failed");
  }
  IDType* order = new IDType[n];
  for (IDType i = 0; i < n; i++) order[i_order[i]] = i;
  return order;
}
#endif
#if !defined(_HEADER_ONLY)
#include "init/amd_reorder.inc"
#endif
}  // namespace sparsebase::reorder
