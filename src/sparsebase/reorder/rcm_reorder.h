#include <vector>

#include "sparsebase/config.h"
#include "sparsebase/format/csr.h"
#include "sparsebase/reorder/reorderer.h"
#include "sparsebase/utils/parameterizable.h"

#ifndef SPARSEBASE_PROJECT_RCM_REORDER_H
#define SPARSEBASE_PROJECT_RCM_REORDER_H

namespace sparsebase::reorder {

//! An empty struct used for the parameters of RCMReorder
struct RCMReorderParams : utils::Parameters {};

//! Reordering using the Reverse Cuthill-McKee algorithm:
//! https://en.wikipedia.org/wiki/Cuthill%E2%80%93McKee_algorithm
template <typename IDType, typename NNZType, typename ValueType>
class RCMReorder : public Reorderer<IDType> {
  typedef typename std::make_signed<IDType>::type SignedID;

 public:
  //! An empty struct used for the parameters of RCMReorder
  typedef RCMReorderParams ParamsType;
  RCMReorder();
  RCMReorder(ParamsType p);

 protected:
  static IDType peripheral(NNZType *xadj, IDType *adj, IDType n, IDType start,
                           SignedID *distance, IDType *Q);
  //! An implementation function that will reorder a CSR format
  /*!
   *
   * @param formats a vector containing a single Format object of type CSR
   * @param params a polymorphic pointer at a `RCMReorderParams` object
   * @return an inverse permutation of the input format; an array of size
   * `format.get_dimensions()[0]` where the ith element is the order of the ith
   * element in the original format object
   */
  static IDType *GetReorderCSR(std::vector<format::Format *> formats,
                               utils::Parameters *);
};


}
#ifdef _HEADER_ONLY
#include "sparsebase/reorder/rcm_reorder.cc"
#endif

#endif  // SPARSEBASE_PROJECT_RCM_REORDER_H
