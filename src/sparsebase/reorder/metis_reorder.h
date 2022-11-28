#include <utility>
#include <vector>

#include "sparsebase/config.h"
#include "sparsebase/format/csr.h"
#include "sparsebase/reorder/reorderer.h"
#include "sparsebase/utils/parameterizable.h"
#ifndef SPARSEBASE_PROJECT_METIS_REORDER_H
#define SPARSEBASE_PROJECT_METIS_REORDER_H

namespace sparsebase::reorder {

#ifdef USE_METIS

struct MetisReorderParams : utils::Parameters {
  int64_t ctype = metis::METIS_CTYPE_RM;
  int64_t rtype = metis::METIS_RTYPE_SEP2SIDED;
  int64_t nseps = 1;
  int64_t numbering = 0;
  int64_t niter = 10;
  int64_t seed = 42;
  int64_t no2hop = 0;
  int64_t compress = 0;
  int64_t ccorder = 0;
  int64_t pfactor = 0;
  int64_t ufactor = 30;
};

template <typename IDType, typename NNZType, typename ValueType>
class MetisReorder : public Reorderer<IDType> {
 public:
  typedef MetisReorderParams ParamsType;
  MetisReorder();
  MetisReorder(ParamsType params);
  static IDType *GetReorderCSR(std::vector<format::Format *> formats,
                               utils::Parameters *);
};

#endif

}  // namespace sparsebase::reorder
#ifdef _HEADER_ONLY
#include "sparsebase/reorder/metis_reorder.cc"
#endif

#endif  // SPARSEBASE_PROJECT_METIS_REORDER_H
