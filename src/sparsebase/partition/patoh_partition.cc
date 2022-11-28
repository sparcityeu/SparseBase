#include "sparsebase/partition/patoh_partition.h"

#include "sparsebase/format/csr.h"
#include "sparsebase/partition/partitioner.h"

namespace sparsebase::partition {
#ifdef USE_PATOH

#include <patoh.h>

template <typename IDType, typename NNZType, typename ValueType>
PatohPartition<IDType, NNZType, ValueType>::PatohPartition() {
  this->RegisterFunction(
      {format::CSR<IDType, NNZType, ValueType>::get_id_static()}, PartitionCSR);

  this->params_ =
      std::unique_ptr<PatohPartitionParams>(new PatohPartitionParams);
}

template <typename IDType, typename NNZType, typename ValueType>
PatohPartition<IDType, NNZType, ValueType>::PatohPartition(
    PatohPartitionParams params) {
  this->RegisterFunction(
      {format::CSR<IDType, NNZType, ValueType>::get_id_static()}, PartitionCSR);

  this->params_ =
      std::unique_ptr<PatohPartitionParams>(new PatohPartitionParams(params));
}

template <typename IDType, typename NNZType, typename ValueType>
IDType *PatohPartition<IDType, NNZType, ValueType>::PartitionCSR(
    std::vector<format::Format *> formats, utils::Parameters *params) {
  if constexpr (!(std::is_same_v<IDType, int> &&
                  std::is_same_v<NNZType, int>)) {
    throw utils::TypeException(
        "Patoh Partitioner requires IDType=int, NNZType=int");
  }

  format::CSR<IDType, NNZType, ValueType> *csr =
      formats[0]->AsAbsolute<format::CSR<IDType, NNZType, ValueType>>();

  int *ptrs = (int *)csr->get_row_ptr();
  int *js = (int *)csr->get_col();
  int m = csr->get_dimensions()[0];
  int n = csr->get_dimensions()[1];

  int *xpins, *pins, *cwghts, *nwghts;
  int i, p;

  cwghts = (int *)malloc(sizeof(int) * n);
  memset(cwghts, 0, sizeof(int) * n);
  for (i = 0; i < m; i++) {
    for (p = ptrs[i]; p < ptrs[i + 1]; p++) {
      cwghts[js[p]]++;
    }
  }

  nwghts = (int *)malloc(sizeof(int) * m);
  for (i = 0; i < m; i++) nwghts[i] = 1;

  xpins = (int *)malloc(sizeof(int) * (m + 1));
  memcpy(xpins, ptrs, sizeof(int) * (m + 1));

  pins = (int *)malloc(sizeof(int) * xpins[m]);
  for (i = 0; i < m; i++) {
    memcpy(pins + xpins[i], js + ptrs[i],
           sizeof(int) * (ptrs[i + 1] - ptrs[i]));
  }

  PatohPartitionParams *concrete_params =
      static_cast<PatohPartitionParams *>(params);
  PaToH_Parameters patoh_params;
  PaToH_Initialize_Parameters(&patoh_params, concrete_params->objective,
                              concrete_params->param_init);
  patoh_params._k = concrete_params->num_partitions;
  patoh_params.MemMul_Pins += 3;
  patoh_params.MemMul_CellNet += 3;
  patoh_params.final_imbal = concrete_params->final_imbalance;
  patoh_params.seed = concrete_params->seed;

  auto alloc_res =
      PaToH_Alloc(&patoh_params, m, n, 1, cwghts, nwghts, xpins, pins);

  if (alloc_res) {
    throw utils::AllocationException();
  }

  int *partition = new int[m];
  int *partwghts = new int[concrete_params->num_partitions];
  int cut = -1;

  PaToH_Part(&patoh_params, m, n, 1, 0, cwghts, nwghts, xpins, pins, nullptr,
             partition, partwghts, &cut);

  delete[] partwghts;
  free(xpins);
  free(pins);
  free(cwghts);
  free(nwghts);

  return (IDType *)partition;
}
#endif

#if !defined(_HEADER_ONLY)
#include "init/patoh_partition.inc"
#endif
}  // namespace sparsebase::partition
