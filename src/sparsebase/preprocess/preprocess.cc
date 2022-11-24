#include "preprocess.h"

#include "sparsebase/converter/converter.h"
#include "sparsebase/feature/degrees.h"
#include "sparsebase/feature/degree_distribution.h"
#include "sparsebase/format/array.h"
#include "sparsebase/format/coo.h"
#include "sparsebase/format/csc.h"
#include "sparsebase/format/csr.h"
#include "sparsebase/format/format.h"
#include "sparsebase/format/format_order_one.h"
#include "sparsebase/format/format_order_two.h"
#include "sparsebase/utils/extractable.h"
#include "sparsebase/utils/function_matcher_mixin.h"
#include "sparsebase/utils/logger.h"
#include "sparsebase/utils/parameterizable.h"
#ifdef USE_CUDA
#include "sparsebase/preprocess/cuda/preprocess.cuh"
#endif
#include <algorithm>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <queue>
#include <set>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef USE_METIS
namespace sparsebase::metis {
#include <metis.h>
}
#endif
namespace sparsebase {

namespace preprocess {


template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
JaccardWeights<IDType, NNZType, ValueType, FeatureType>::JaccardWeights(
    ParamsType) {}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
JaccardWeights<IDType, NNZType, ValueType, FeatureType>::JaccardWeights() {
#ifdef USE_CUDA
  std::vector<std::type_index> formats = {
      format::CUDACSR<IDType, NNZType,
                            ValueType>::get_id_static()};
  this->RegisterFunction(formats, GetJaccardWeightCUDACSR);
#endif
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
JaccardWeights<IDType, NNZType, ValueType, FeatureType>::~JaccardWeights(){};

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
format::Format *
JaccardWeights<IDType, NNZType, ValueType, FeatureType>::GetJaccardWeights(
    format::Format *format, std::vector<context::Context *> contexts,
    bool convert_input) {
  return this->Execute(nullptr, contexts, convert_input,
                       format);  // func(sfs, this->params_.get());
}
#ifdef USE_CUDA
template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
format::Format *preprocess::JaccardWeights<
    IDType, NNZType, ValueType,
    FeatureType>::GetJaccardWeightCUDACSR(std::vector<format::Format *> formats,
                                          utils::Parameters *params) {
  auto cuda_csr =
      formats[0]
          ->AsAbsolute<format::CUDACSR<IDType, NNZType, ValueType>>();
  return preprocess::cuda::RunJaccardKernel<IDType, NNZType, ValueType,
                                            FeatureType>(cuda_csr);
}
#endif
#if !defined(_HEADER_ONLY)
#include "init/preprocess.inc"
#endif

}  // namespace preprocess

}  // namespace sparsebase
