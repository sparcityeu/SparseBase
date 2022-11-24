#include <vector>

#include "sparsebase/config.h"
#include "sparsebase/format/csr.h"
#include "sparsebase/utils/parameterizable.h"
#include "sparsebase/utils/function_matcher_mixin.h"

#ifndef SPARSEBASE_PROJECT_JACCARD_WEIGHTS_H
#define SPARSEBASE_PROJECT_JACCARD_WEIGHTS_H
namespace sparsebase::feature {
//! An empty struct used for the parameters of JaccardWeights
struct JaccardWeightsParams : utils::Parameters {};
//! Calculate the Jaccard Weights of the edges in a graph representation of a
//! format object
template <typename IDType, typename NNZType, typename ValueType,
    typename FeatureType>
class JaccardWeights : public utils::FunctionMatcherMixin<format::Format *> {
 public:
  //! An empty struct used for the parameters of JaccardWeights
  typedef JaccardWeightsParams ParamsType;
  JaccardWeights();
  JaccardWeights(ParamsType);
  //! Take a single Format object representating a graph and get the Jaccard
  //! Weights as a 1D format object
  /*!
   *
   * @param format input format object representing a graph
   * @param convert_input whether or not to convert the input format if that is
   * needed.
   * @return a 1D format (array) where element i in the array is the Jaccard
   * Weight of edge i in the graph (ith non-zero)
   */
  format::Format *GetJaccardWeights(format::Format *format,
                                    std::vector<context::Context *>,
                                    bool convert_input);
#ifdef USE_CUDA
  //! Take a CUDACSR representating a graph and get the Jaccard Weights as a
  //! CUDAArray
  /*!
   *
   * @param formats a vector of size 1 with formats[0] being CUDACSR
   * representing a graph
   * @param params a polymorphic pointer at a Parameters (not used)
   * @return a 1D array (CUDAArray) where element i in the array is the Jaccard
   * Weight of edge i in the graph (ith non-zero)
   */
  static format::Format *GetJaccardWeightCUDACSR(
      std::vector<format::Format *> formats, utils::Parameters *params);
#endif
  ~JaccardWeights();
};

}
#ifdef _HEADER_ONLY
#include "sparsebase/feature/jaccard_weights.cc"
#endif

#ifdef USE_CUDA
#include "sparsebase/feature/jaccard_weights_cuda.cuh"
#ifdef _HEADER_ONLY
#include "sparsebase/feature/jaccard_weights_cuda.cu"
#endif
#endif


#endif  // SPARSEBASE_PROJECT_JACCARD_WEIGHTS_H
