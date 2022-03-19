#include "sparsebase/sparse_format.h"
#include "sparsebase/sparse_preprocess.h"
#include "sparsebase/sparse_preprocess.h"
namespace sparsebase {
namespace preprocess {
template<typename IDType, typename NNZType, typename FeatureType>
__global__ void jac_binning_gpu_u_per_grid_bst_kernel(const NNZType* xadj, const IDType* adj, NNZType n, FeatureType* emetrics, IDType SM_FAC);
}
}