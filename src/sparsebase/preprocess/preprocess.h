/*******************************************************
 * Copyright (c) 2022 SparCity, Amro Alabsi Aljundi, Taha Atahan Akyildiz, Arda
 *Sener All rights reserved.
 *
 * This file is distributed under MIT license.
 * The complete license agreement can be obtained at:
 * https://sparcityeu.github.io/sparsebase/pages/license.html
 ********************************************************/
#ifndef SPARSEBASE_SPARSEBASE_PREPROCESS_PREPROCESS_H_
#define SPARSEBASE_SPARSEBASE_PREPROCESS_PREPROCESS_H_
#include <any>
#include <cmath>
#include <iostream>
#include <memory>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>
#include <vector>

#include "sparsebase/config.h"
#include "sparsebase/context/context.h"
#include "sparsebase/format/format.h"
#include "sparsebase/format/format_order_one.h"
#include "sparsebase/format/format_order_two.h"

#include "sparsebase/object/object.h"
#include "sparsebase/converter/converter.h"
#include "sparsebase/utils/function_matcher_mixin.h"
#include "sparsebase/utils/extractable.h"
#include "sparsebase/feature/feature_preprocess_type.h"

#ifdef USE_METIS
namespace sparsebase::metis {
#include <metis.h>
}
#endif

namespace sparsebase::preprocess {

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



}  // namespace sparsebase::preprocess
#ifdef _HEADER_ONLY
#include "sparsebase/preprocess/preprocess.cc"
#endif

#ifdef USE_CUDA
#include "cuda/preprocess.cuh"
#ifdef _HEADER_ONLY
#include "cuda/preprocess.cu"
#endif
#endif

#endif  // SPARSEBASE_SPARSEBASE_PREPROCESS_PREPROCESS_H_
