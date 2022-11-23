#include "sparsebase/config.h"
#include "sparsebase/utils/parameterizable.h"
#include "sparsebase/partition/partitioner.h"
#include "sparsebase/format/csr.h"
#include <vector>
#ifndef SPARSEBASE_PROJECT_METIS_PARTITION_H
#define SPARSEBASE_PROJECT_METIS_PARTITION_H
#ifdef USE_METIS
namespace sparsebase::metis {
#include <metis.h>
}
#endif
namespace sparsebase::partition {
#ifdef USE_METIS

//! Parameters for metis partitioning
/*!
 * This struct replaces the options array of METIS
 * The names of the options are identical to the array
 * and can be found here:
 * http://glaros.dtc.umn.edu/gkhome/fetch/sw/metis/manual.pdf
 */
struct MetisPartitionParams : utils::Parameters {
  int64_t num_partitions = 2;
  int64_t ptype = metis::METIS_PTYPE_KWAY;
  int64_t objtype = metis::METIS_OBJTYPE_CUT;
  int64_t ctype = metis::METIS_CTYPE_RM;
  int64_t iptype = metis::METIS_IPTYPE_GROW;
  int64_t rtype = metis::METIS_RTYPE_FM;
  int64_t ncuts = 1;
  int64_t nseps = 1;
  int64_t numbering = 0;
  int64_t niter = 10;
  int64_t seed = 42;
  int64_t minconn = 0;
  int64_t no2hop = 0;
  int64_t contig = 0;
  int64_t compress = 0;
  int64_t ccorder = 0;
  int64_t pfactor = 0;
  int64_t ufactor = 30;
};

//! A wrapper for the METIS partitioner
/* !
 * Wraps the METIS partitioner available here:
 * https://github.com/KarypisLab/METIS The library must be compiled with the
 * USE_METIS option turned on and the pre-built METIS library should be
 * available. See the Optional Dependencies page (under Getting Started) in our
 * documentation for more info. Detailed explanations of the options can be
 * found here: http://glaros.dtc.umn.edu/gkhome/fetch/sw/metis/manual.pdf
 */
template <typename IDType, typename NNZType, typename ValueType>
class MetisPartition : public Partitioner<IDType> {
 private:
  static IDType *PartitionCSR(std::vector<format::Format *> formats,
                              utils::Parameters *params);

 public:
  typedef MetisPartitionParams ParamsType;
  MetisPartition();
  MetisPartition(ParamsType params);
};

#endif

}
#ifdef _HEADER_ONLY
#include "sparsebase/partition/metis_partition.cc"
#endif

#endif  // SPARSEBASE_PROJECT_METIS_PARTITION_H
