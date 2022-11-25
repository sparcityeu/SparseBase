#include "sparsebase/partition/pulp_partition.h"

#include "sparsebase/format/csr.h"
#include "sparsebase/partition/partitioner.h"
#include "sparsebase/utils/logger.h"

namespace sparsebase::partition {
#ifdef USE_PULP

#include <pulp.h>

template <typename IDType, typename NNZType, typename ValueType>
PulpPartition<IDType, NNZType, ValueType>::PulpPartition() {
  this->RegisterFunction(
      {format::CSR<IDType, NNZType, ValueType>::get_id_static()}, PartitionCSR);

  this->params_ = std::unique_ptr<PulpPartitionParams>(new PulpPartitionParams);
}

template <typename IDType, typename NNZType, typename ValueType>
PulpPartition<IDType, NNZType, ValueType>::PulpPartition(
    PulpPartitionParams params) {
  this->RegisterFunction(
      {format::CSR<IDType, NNZType, ValueType>::get_id_static()}, PartitionCSR);

  this->params_ =
      std::unique_ptr<PulpPartitionParams>(new PulpPartitionParams(params));
}

template <typename IDType, typename NNZType, typename ValueType>
IDType *PulpPartition<IDType, NNZType, ValueType>::PartitionCSR(
    std::vector<format::Format *> formats, utils::Parameters *params) {
  format::CSR<IDType, NNZType, ValueType> *csr =
      formats[0]->AsAbsolute<format::CSR<IDType, NNZType, ValueType>>();

  PulpPartitionParams *pparams = static_cast<PulpPartitionParams *>(params);

  format::DimensionType n = csr->get_dimensions()[0];
  format::DimensionType m = csr->get_num_nnz();

  pulp_part_control_t con;
  con.vert_balance = pparams->vert_balance;
  con.edge_balance = pparams->edge_balance;
  con.pulp_seed = pparams->seed;
  con.do_lp_init = pparams->do_lp_init;
  con.do_bfs_init = pparams->do_bfs_init;
  con.do_repart = pparams->do_repart;
  con.do_edge_balance = pparams->do_edge_balance;
  con.do_maxcut_balance = pparams->do_maxcut_balance;

  int np = pparams->num_partitions;
  IDType *partition = new IDType[n];

  if constexpr (std::is_same_v<IDType, int> && std::is_same_v<NNZType, long>) {
    pulp_graph_t graph;
    graph.n = n;
    graph.m = m;
    graph.out_array = csr->get_col();
    graph.out_degree_list = csr->get_row_ptr();
    graph.vertex_weights = nullptr;
    graph.edge_weights = nullptr;
    graph.vertex_weights_sum = 0;
    pulp_run(&graph, &con, partition, np);
  } else {
    throw utils::TypeException(
        "Pulp Partitioner requires IDType=int, NNZType=long");
  }
  return partition;
}
#endif

#if !defined(_HEADER_ONLY)
#include "init/pulp_partition.inc"
#endif
}  // namespace sparsebase::partition
