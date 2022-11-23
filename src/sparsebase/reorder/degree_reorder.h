#include "sparsebase/config.h"
#include "sparsebase/utils/parameterizable.h"
#include "sparsebase/reorder/reorder.h"
#include "sparsebase/format/csr.h"
#include <vector>

#ifndef SPARSEBASE_PROJECT_DEGREE_REORDER_H
#define SPARSEBASE_PROJECT_DEGREE_REORDER_H

namespace sparsebase::reorder {

//! Parameters used in DegreeReorder, namely whether or not degrees are ordered
//! in ascending order.
struct DegreeReorderParams : utils::Parameters {
  bool ascending;
  DegreeReorderParams(bool ascending) : ascending(ascending) {}
};
//! Reordering preprocessing algorithm that reorders a format by representing it
//! as an adjacency matrix of a graph and ordering its vertices by degree
template <typename IDType, typename NNZType, typename ValueType>
class DegreeReorder : public Reorderer<IDType> {
 public:
  DegreeReorder(bool ascending);
  //! The hyperparameters used by the implementation functions of DegreeReorder
  typedef DegreeReorderParams ParamsType;
  DegreeReorder(DegreeReorderParams);

 protected:
  //! An implementation function that will reorder a CSR format
  /*!
   *
   * @param formats a vector containing a single Format object of type CSR
   * @param params a polymorphic pointer at a `DegreeReorderParams` object
   * @return an inverse permutation of the input format; an array of size
   * `format.get_dimensions()[0]` where the ith element is the order of the ith
   * element in the original format object
   */
  static IDType *CalculateReorderCSR(std::vector<format::Format *> formats,
                                     utils::Parameters *params);
};

}
#ifdef _HEADER_ONLY
#include "sparsebase/reorder/degree_reorder.cc"
#endif

#endif  // SPARSEBASE_PROJECT_DEGREE_REORDER_H
