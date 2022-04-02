#include "sparsebase/format/format.h"
#include "sparsebase/preprocess/preprocess.h"
namespace sparsebase {

    namespace preprocess {
namespace cuda{
      template<typename IDType, typename NNZType, typename FeatureType>
      __global__ void jac_binning_gpu_u_per_grid_bst_kernel(const NNZType* xadj, const IDType* adj, NNZType n, FeatureType* emetrics, IDType SM_FAC);
  }
}
}