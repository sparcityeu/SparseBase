#include "sparsebase/format/format.h"
#include "sparsebase/format/cuda/format.cuh"
#include "sparsebase/preprocess/preprocess.h"
namespace sparsebase {

    namespace preprocess {
namespace cuda{
  template <typename IDType, typename NNZType, typename ValueType, typename FeatureType>
  format::cuda::CUDAArray<FeatureType>* RunJaccardKernel(format::cuda::CUDACSR<IDType, NNZType, ValueType>* cuda_csr);
  }
}
}