#include <vector>

#include "sparsebase/config.h"
#include "sparsebase/format/csr.h"
#include "sparsebase/partition/partitioner.h"
#include "sparsebase/utils/parameterizable.h"
#ifndef SPARSEBASE_PROJECT_PULP_PARTITION_H
#define SPARSEBASE_PROJECT_PULP_PARTITION_H
namespace sparsebase::partition {

#ifdef USE_PULP

//! Parameters for the PulpPartition class
struct PulpPartitionParams : utils::Parameters {
  double vert_balance = 1.1;
  double edge_balance = 1.5;
  bool do_lp_init = false;
  bool do_bfs_init = true;
  bool do_repart = false;
  bool do_edge_balance = false;
  bool do_maxcut_balance = false;
  bool verbose_output = false;
  int seed = 42;
  int num_partitions = 2;
};

//! A wrapper for the PULP graph partitioner
/* !
 * Wraps the PULP partitioner available here:
 * https://github.com/HPCGraphAnalysis/PuLP. The library must be compiled with
 * the USE_PULP option turned on and the pre-built PULP library should be
 * available. See the Optional Dependencies page (under Getting Started) in our
 * documentation for more info.
 */
template <typename IDType, typename NNZType, typename ValueType>
class PulpPartition : public Partitioner<IDType> {
 private:
  static IDType *PartitionCSR(std::vector<format::Format *> formats,
                              utils::Parameters *params);

 public:
  typedef PulpPartitionParams ParamsType;
  PulpPartition();
  PulpPartition(ParamsType params);
};
#endif
}  // namespace sparsebase::partition
#ifdef _HEADER_ONLY
#include "sparsebase/partition/pulp_partition.cc"
#endif

#endif  // SPARSEBASE_PROJECT_PULP_PARTITION_H
