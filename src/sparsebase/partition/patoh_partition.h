#include "sparsebase/config.h"
#include "sparsebase/utils/parameterizable.h"
#include "sparsebase/partition/partitioner.h"
#include "sparsebase/format/csr.h"
#include <vector>
#ifndef SPARSEBASE_PROJECT_PATOH_PARTITION_H
#define SPARSEBASE_PROJECT_PATOH_PARTITION_H
namespace sparsebase::partition {


#ifdef USE_PATOH

namespace patoh {
enum Objective {
  CON = 1,
  CUT = 2
};

enum ParameterInit {
  DEFAULT = 0,
  SPEED = 1,
  QUALITY = 2
};

}

//! Parameters for the PulpPartition class
struct PatohPartitionParams : utils::Parameters {
  patoh::Objective objective = patoh::CON;
  patoh::ParameterInit param_init = patoh::DEFAULT;
  int num_partitions = 2;
  int final_imbalance = -1;
  int seed = 42;
};

//! A wrapper for the Patoh graph partitioner
/* !
 * Wraps the Patoh partitioner available here:
 * https://faculty.cc.gatech.edu/~umit/software.html.
 * The library must be compiled with the
 * USE_PATOH option turned on and the pre-built PATOH library should be
 * available. See the Optional Dependencies page (under Getting Started) in our
 * documentation for more info.
 */
template <typename IDType, typename NNZType, typename ValueType>
class PatohPartition : public Partitioner<IDType> {
 private:
  static IDType *PartitionCSR(std::vector<format::Format *> formats,
                              utils::Parameters *params);

 public:
  typedef PatohPartitionParams ParamsType;
  PatohPartition();
  PatohPartition(ParamsType params);
};
#endif



}
#ifdef _HEADER_ONLY
#include "sparsebase/partition/patoh_partition.cc"
#endif

#endif  // SPARSEBASE_PROJECT_PATOH_PARTITION_H
