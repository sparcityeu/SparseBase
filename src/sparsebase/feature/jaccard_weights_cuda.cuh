#include "sparsebase/format/cuda_csr_cuda.cuh"
#include "sparsebase/format/cuda_array_cuda.cuh"

#ifndef SPARSEBASE_PROJECT_JACCARD_WEIGHTS_CUDA_CUH
#define SPARSEBASE_PROJECT_JACCARD_WEIGHTS_CUDA_CUH

namespace sparsebase::feature {

template <typename IDType, typename NNZType, typename ValueType,
    typename FeatureType>
format::CUDAArray<FeatureType> *RunJaccardKernel(
    format::CUDACSR<IDType, NNZType, ValueType> *cuda_csr);
}

#endif  // SPARSEBASE_PROJECT_JACCARD_WEIGHTS_CUDA_CUH
