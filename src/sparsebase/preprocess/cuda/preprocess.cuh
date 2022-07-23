/*******************************************************
 * Copyright (c) 2022 SparCity, Amro Alabsi Aljundi, Taha Atahan Akyildiz, Arda Sener
 * All rights reserved.
 *
 * This file is distributed under MIT license.
 * The complete license agreement can be obtained at:
 * https://sparcityeu.github.io/sparsebase/pages/license.html
 ********************************************************/
#ifndef SPARSEBASE_SPARSEBASE_PREPROCESS_CUDA_PREPROCESS_H_
#define SPARSEBASE_SPARSEBASE_PREPROCESS_CUDA_PREPROCESS_H_
#include "sparsebase/format/cuda/format.cuh"
#include "sparsebase/format/format.h"
#include "sparsebase/preprocess/preprocess.h"
namespace sparsebase {

namespace preprocess {
namespace cuda {
template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
format::cuda::CUDAArray<FeatureType> *
RunJaccardKernel(format::cuda::CUDACSR<IDType, NNZType, ValueType> *cuda_csr);
}
} // namespace preprocess
} // namespace sparsebase
#endif // SPARSEBASE_SPARSEBASE_PREPROCESS_CUDA_PREPROCESS_H_