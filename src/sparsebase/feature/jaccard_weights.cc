#include "sparsebase/feature/jaccard_weights.h"

#include "sparsebase/format/csr.h"
#include "sparsebase/utils/parameterizable.h"
#ifdef USE_CUDA
#include "sparsebase/format/cuda_csr_cuda.cuh"
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

namespace sparsebase::feature {

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
JaccardWeights<IDType, NNZType, ValueType, FeatureType>::JaccardWeights(
    ParamsType) {}

#ifdef USE_CUDA
template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
format::Format *JaccardWeights<IDType, NNZType, ValueType, FeatureType>::
    GetJaccardWeightCUDACSR(std::vector<format::Format *> formats,
                            utils::Parameters *params) {
  auto cuda_csr =
      formats[0]->AsAbsolute<format::CUDACSR<IDType, NNZType, ValueType>>();
  return RunJaccardKernel<IDType, NNZType, ValueType, FeatureType>(cuda_csr);
}
#endif

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
JaccardWeights<IDType, NNZType, ValueType, FeatureType>::JaccardWeights() {
#ifdef USE_CUDA
  std::vector<std::type_index> formats = {
      format::CUDACSR<IDType, NNZType, ValueType>::get_id_static()};
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
#if !defined(_HEADER_ONLY)
#include "init/jaccard_weights.inc"
#endif
}  // namespace sparsebase::feature
