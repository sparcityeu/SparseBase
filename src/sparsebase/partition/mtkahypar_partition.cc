#include "sparsebase/partition/mtkahypar_partition.h"

#include <fstream>
#include <thread>

#include "sparsebase/format/csr.h"
#include "sparsebase/partition/partitioner.h"

#ifdef USE_MTKAHYPAR
namespace sparsebase::mtkahypar {
#include <libmtkahypar.h>
}
#endif

namespace sparsebase::partition {
#ifdef USE_MTKAHYPAR

template <typename IDType, typename NNZType, typename ValueType>
MtkahyparPartition<IDType, NNZType, ValueType>::MtkahyparPartition() {
  this->RegisterFunction(
      {format::CSR<IDType, NNZType, ValueType>::get_id_static()}, PartitionCSR);

  this->params_ =
      std::unique_ptr<MtkahyparPartitionParams>(new MtkahyparPartitionParams);
}

template <typename IDType, typename NNZType, typename ValueType>
MtkahyparPartition<IDType, NNZType, ValueType>::MtkahyparPartition(
    MtkahyparPartitionParams params) {
  this->RegisterFunction(
      {format::CSR<IDType, NNZType, ValueType>::get_id_static()}, PartitionCSR);

  this->params_ = std::unique_ptr<MtkahyparPartitionParams>(
      new MtkahyparPartitionParams(params));
}

template <typename IDType, typename NNZType, typename ValueType>
IDType *MtkahyparPartition<IDType, NNZType, ValueType>::PartitionCSR(
    std::vector<format::Format *> formats, utils::Parameters *params) {
  auto *mt_params = reinterpret_cast<MtkahyparPartitionParams *>(params);

  sparsebase::mtkahypar::mt_kahypar_initialize_thread_pool(
      1 /* use all available cores */,
      true /* activate interleaved NUMA allocation policy */ );

  // Context preparing
  sparsebase::mtkahypar::mt_kahypar_context_t *context =
      sparsebase::mtkahypar::mt_kahypar_context_new();
  sparsebase::mtkahypar::mt_kahypar_load_preset(context, mt_params->preset);
  sparsebase::mtkahypar::mt_kahypar_set_partitioning_parameters(
      context, mt_params->num_partitions, mt_params->imbalance_parameter,
      mt_params->objective_function);
  sparsebase::mtkahypar::mt_kahypar_set_seed(mt_params->seed);
  mt_kahypar_set_context_parameter(context, mtkahypar::VERBOSE, "0");

  format::CSR<IDType, NNZType, ValueType> *csr =
      formats[0]->AsAbsolute<format::CSR<IDType, NNZType, ValueType>>();

  // Graph preparing
  auto num_vertex = csr->get_dimensions()[0];
  auto num_edges = csr->get_num_nnz();
  auto *row_ptr = csr->get_row_ptr();
  auto *col = csr->get_col();

  std::cout << "Init vertex:" << num_vertex << std::endl;
  std::cout << "Init edges:" << num_edges << std::endl;

  auto *edges =
      new sparsebase::mtkahypar::mt_kahypar_hypernode_id_t[num_edges * 2];

  std::size_t edge_index = 0;
  for (std::size_t i = 0; i < num_vertex; ++i) {
    for (std::size_t p = row_ptr[i]; p < row_ptr[i + 1]; ++p) {
      edges[edge_index] = i;
      edges[edge_index + 1] = col[p];
      edge_index += 2;
    }
  }

  auto graph = mtkahypar::mt_kahypar_create_graph(
      mt_params->preset, num_vertex, num_edges, edges, nullptr, nullptr);

  std::cout << "Number of Nodes: "
            << mtkahypar::mt_kahypar_num_hypernodes(graph) << std::endl;
  std::cout << "Number of Edges: "
            << mtkahypar::mt_kahypar_num_hyperedges(graph) << std::endl;

  // Partition Hypergraph
  sparsebase::mtkahypar::mt_kahypar_partitioned_hypergraph_t partitioned_graph =
      sparsebase::mtkahypar::mt_kahypar_partition(graph, context);

  IDType *partition = new IDType[num_vertex];

  // Extract Partition
  sparsebase::mtkahypar::mt_kahypar_get_partition(
      partitioned_graph,
      reinterpret_cast<mtkahypar::mt_kahypar_partition_id_t *>(partition));

  // Compute Metrics
  const double imbalance =
      sparsebase::mtkahypar::mt_kahypar_imbalance(partitioned_graph, context);
  const double cut = sparsebase::mtkahypar::mt_kahypar_cut(partitioned_graph);
  sparsebase::mtkahypar::mt_kahypar_cut(partitioned_graph);

  // Output Results
  std::cout << "Partitioning Results:" << std::endl;
  std::cout << "Imbalance         = " << imbalance << std::endl;
  std::cout << "Cut               = " << cut << std::endl;

  sparsebase::mtkahypar::mt_kahypar_free_context(context);
  sparsebase::mtkahypar::mt_kahypar_free_hypergraph(graph);
  sparsebase::mtkahypar::mt_kahypar_free_partitioned_hypergraph(
      partitioned_graph);
  delete[] edges;

  return partition;
}
#endif

#if !defined(_HEADER_ONLY)
#include "init/mtkahypar_partition.inc"
#endif
}  // namespace sparsebase::partition
