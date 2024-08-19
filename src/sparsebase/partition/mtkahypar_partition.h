#include <vector>

#include "sparsebase/config.h"
#include "sparsebase/format/csr.h"
#include "sparsebase/partition/partitioner.h"
#include "sparsebase/utils/parameterizable.h"
#ifndef SPARSEBASE_PROJECT_MTKAHYPAR_PARTITION_H
#define SPARSEBASE_PROJECT_MTKAHYPAR_PARTITION_H
#ifdef USE_MTKAHYPAR
namespace sparsebase::mtkahypar {
#include <libmtkahypar.h>
}
#endif
namespace sparsebase::partition {
#ifdef USE_MTKAHYPAR

//! Parameters for mtkahypar partitioning
/*!
 * This struct replaces the options array of MTKAHYPAR
 * The names of the options are identical to the array
 * and can be found here:
 * https://github.com/kahypar/mt-kahypar
 */
struct MtkahyparPartitionParams : utils::Parameters {
  int32_t num_partitions = 2;
  float imbalance_parameter = 0.03;  // Allowed imbalance (%)
  std::size_t seed = 42;
  mtkahypar::mt_kahypar_preset_type_t preset = mtkahypar::DETERMINISTIC;
  mtkahypar::mt_kahypar_objective_t objective_function = mtkahypar::CUT;
};

//! A wrapper for the MTKAHYPAR partitioner
/* !
 * Wraps the MTKAHYPAR partitioner available here:
 * https://github.com/kahypar/mt-kahypar The library must be compiled with the
 * USE_METIS option turned on and the pre-built METIS library should be
 * available. See the Optional Dependencies page (under Getting Started) in our
 * documentation for more info. Detailed explanations of the options can be
 * found here: https://github.com/kahypar/mt-kahypar
 */
template <typename IDType, typename NNZType, typename ValueType>
class MtkahyparPartition : public Partitioner<IDType> {
 private:
  static IDType *PartitionCSR(std::vector<format::Format *> formats,
                              utils::Parameters *params);

 public:
  typedef MtkahyparPartitionParams ParamsType;
  MtkahyparPartition();
  MtkahyparPartition(ParamsType params);
};

#endif

}  // namespace sparsebase::partition
#ifdef _HEADER_ONLY
#include "sparsebase/partition/mtkahypar_partition.cc"
#endif

#endif  // SPARSEBASE_PROJECT_MTKAHYPAR_PARTITION_H
