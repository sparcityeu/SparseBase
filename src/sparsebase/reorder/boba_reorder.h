#include <utility>
#include <vector>

#include "sparsebase/config.h"
#include "sparsebase/format/coo.h"
#include "sparsebase/reorder/reorderer.h"
#include "sparsebase/utils/parameterizable.h"
#ifndef SPARSEBASE_PROJECT_BOBA_REORDER_H
#define SPARSEBASE_PROJECT_BOBA_REORDER_H

namespace sparsebase::reorder {

struct BOBAReorderParams : utils::Parameters {
  bool sequential;
  explicit BOBAReorderParams() {}
  BOBAReorderParams(bool sequential_)
      : sequential(sequential_) {}
};

template <typename IDType, typename NNZType, typename ValueType>
class BOBAReorder : public Reorderer<IDType> {
  typedef typename std::make_signed<IDType>::type SignedID;

 public:
  //! Parameter type for GrayReorder
  typedef BOBAReorderParams ParamsType;
  BOBAReorder(bool sequential);
  BOBAReorder(ParamsType p);

  protected: 
    //! An implementation function that will reorder a COO format
    /*!
    *
    * @param formats a vector containing a single Format object of type COO
    * @param params a polymorphic pointer at a `RCMReorderParams` object
    * @return an inverse permutation of the input format; an array of size
    * `format.get_dimensions()[0]` where the ith element is the order of the ith
    * element in the original format object
    */
    static IDType *GetReorderCOO(std::vector<format::Format *> formats,
                                utils::Parameters *);
};

}  // namespace sparsebase::reorder
#ifdef _HEADER_ONLY
#include "sparsebase/reorder/boba_reorder.cc"
#endif
#endif  // SPARSEBASE_PROJECT_BOBA_REORDER_H
