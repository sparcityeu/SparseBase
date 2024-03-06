#include <utility>
#include <vector>
#include <stack>

#include "sparsebase/config.h"
#include "sparsebase/format/csr.h"
#include "sparsebase/reorder/reorderer.h"
#include "sparsebase/utils/parameterizable.h"
#ifndef SPARSEBASE_PROJECT_SLASHBURN_REORDER_H
#define SPARSEBASE_PROJECT_SLASHBURN_REORDER_H

namespace sparsebase::reorder {

struct SlashburnReorderParams : utils::Parameters {
  int k_size;
  bool greedy;
  bool hub_order;
  explicit SlashburnReorderParams() {}
  SlashburnReorderParams(int hubset_k_size, bool greedy_alg, bool hub_ordering)
      : k_size(hubset_k_size),
        greedy(greedy_alg),
        hub_order(hub_ordering) {}
};

template <typename IDType, typename NNZType, typename ValueType>
class SlashburnReorder : public Reorderer<IDType> {
  typedef typename std::make_signed<IDType>::type SignedID;

 public:
  //! Parameter type for GrayReorder
  typedef SlashburnReorderParams ParamsType;
  SlashburnReorder(int k_size, bool greedy, bool hub_order);
  SlashburnReorder(ParamsType p);

  protected: 
    static void slashloop(NNZType *rptr, IDType *col, IDType n, IDType k,
                           IDType *v_flag, IDType *order, IDType level, IDType max_id);
    
    static IDType* removeKHubset(NNZType *rptr, IDType *col, IDType n, IDType k, IDType *v_flag, IDType *order, IDType level, IDType min_id);
    static IDType* removeKHubsetGreedy(NNZType *rptr, IDType *col, IDType n, IDType k, IDType *v_flag, IDType *order, IDType *degree, IDType level, IDType min_id);
    static IDType* computeDegree(NNZType *rptr, IDType *col, IDType n, IDType *v_flag, IDType level);
    
    static IDType findCC(NNZType *rptr, IDType *col, IDType *v_flag, IDType level, IDType root);
    static IDType orderCC(NNZType *rptr, IDType *col, IDType *v_flag, IDType *order, IDType level, IDType root, IDType max_id);
    
    //! An implementation function that will reorder a COO format
    /*!
    *
    * @param formats a vector containing a single Format object of type COO
    * @param params a polymorphic pointer at a `RCMReorderParams` object
    * @return an inverse permutation of the input format; an array of size
    * `format.get_dimensions()[0]` where the ith element is the order of the ith
    * element in the original format object
    */
    static IDType *GetReorderCSR(std::vector<format::Format *> formats,
                                utils::Parameters *);
};

}  // namespace sparsebase::reorder
#ifdef _HEADER_ONLY
#include "sparsebase/reorder/slashburn_reorder.cc"
#endif
#endif  // SPARSEBASE_PROJECT_SLASHBURN_REORDER_H
