#include "sparsebase/format/cuda/format.cuh"
#include "sparsebase/format/format.h"
#include "sparsebase/preprocess/cuda/preprocess.cuh"
#include "sparsebase/preprocess/preprocess.h"
using namespace sparsebase;
using namespace sparsebase::format;
using namespace sparsebase::utils;
namespace sparsebase {

namespace preprocess {
namespace cuda {
template <typename IDType, typename NNZType, typename FeatureType>
__global__ void
jac_binning_gpu_u_per_grid_bst_kernel(const NNZType *xadj, const IDType *adj,
                                      NNZType n, FeatureType *emetrics,
                                      IDType SM_FAC);
template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
format::cuda::CUDAArray<FeatureType> *
RunJaccardKernel(format::cuda::CUDACSR<IDType, NNZType, ValueType> *cuda_csr) {

  context::cuda::CUDAContext *gpu_context =
      static_cast<context::cuda::CUDAContext *>(cuda_csr->get_context());
  cudaSetDevice(gpu_context->device_id);

  FeatureType *jaccard_weights;
  cudaMalloc(&jaccard_weights, cuda_csr->get_num_nnz() * sizeof(FeatureType));

#define MAX_GRID_DIM 65535
#define WARP_SIZE 32
  dim3 block(1, 1, 1);
  int g = 64;
  int a = 8;
  block.x = g;
  // block.y = max((size_t)1,(size_t)THREADS_PER_BLOCK/block.x);
  if (g < WARP_SIZE) {
    block.y = WARP_SIZE / g;
  }

  dim3 grid(1, 1, 1);
  // grid.x = max((size_t)1,(size_t)a/block.y);
  grid.x = a;
  grid.y = max((int)min((int)cuda_csr->get_dimensions()[0], (int)MAX_GRID_DIM),
               (int)1);
  grid.z =
      (int)min((int)max((int)cuda_csr->get_dimensions()[0] / grid.y, (int)1),
               (int)MAX_GRID_DIM);

  jac_binning_gpu_u_per_grid_bst_kernel<IDType, NNZType, FeatureType>
      <<<grid, block, 0>>>(cuda_csr->get_row_ptr(), cuda_csr->get_col(),
                           (NNZType)cuda_csr->get_dimensions()[0],
                           jaccard_weights, 0);
  cudaDeviceSynchronize();
  auto array = new format::cuda::CUDAArray<FeatureType>(
      cuda_csr->get_num_nnz(), jaccard_weights, *gpu_context);
  return array;
}
#define FULL_MASK 0xffffffff
__inline__ __device__ unsigned calculateMask(char length,
                                             unsigned long long thread_id) {
  if (length >= 32)
    return FULL_MASK;
  // unsigned mask = 0x80000000;
  unsigned mask = 1;
  for (char i = 1; i < length; i++) {
    mask = (mask << 1) | 1;
    //  mask = (mask>>1)|0x80000000;
  }
  int group_in_warp = (thread_id % WARP_SIZE) / length;
  mask = mask << (group_in_warp * length);
  // mask = mask >> (group_in_warp*length);
  return mask;
}
template <typename NNZType, typename IDType>
__device__ inline NNZType bst(const NNZType *__restrict__ xadj,
                              const IDType *__restrict__ adj, IDType neighbor,
                              IDType target) {
  NNZType match = (NNZType)(-1);
  NNZType left = xadj[neighbor] + 1;
  NNZType right = xadj[neighbor + 1];
  IDType curr;
  NNZType middle;
  while (left <= right) {
    middle = ((unsigned long long)left + (unsigned long long)right) >> 1;
    curr = adj[middle - 1];
    if (curr > target) {
      right = middle - 1;
    } else if (curr < target) {
      left = middle + 1;
    } else {
      match = middle - 1;
      break;
    }
  }
  return match;
}
__inline__ __device__ int warpReduce(int val, unsigned int length,
                                     unsigned mask) {
  for (int delta = length / 2; delta > 0; delta /= 2) {
    val += __shfl_down_sync(mask, val, delta, length);
  }
  return val;
}
template <typename IDType, typename NNZType, typename FeatureType>
__global__ void
jac_binning_gpu_u_per_grid_bst_kernel(const NNZType *xadj, const IDType *adj,
                                      NNZType n, FeatureType *emetrics,
                                      IDType SM_FAC) {
  const bool directed = false;
  // int no_threads = blockDim.z * blockDim.y * blockDim.x * gridDim.x;
  unsigned int block_local_id = blockDim.x * blockDim.y * threadIdx.z +
                                blockDim.x * threadIdx.y + threadIdx.x;
  unsigned int grid_id =
      gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;
  unsigned long long tid =
      (long long)blockDim.z * blockDim.x * blockDim.y * grid_id +
      (unsigned long long)block_local_id;
  unsigned mask =
      calculateMask(blockDim.x, tid); // threadIdx.y*blockDim.x+threadIdx.x);

  for (NNZType ptr = blockIdx.y + blockIdx.z * gridDim.y; ptr < n;
       ptr += gridDim.y * gridDim.z) {
    IDType u = ptr;
    NNZType degu = xadj[u + 1] - xadj[u];

    for (NNZType neigh_ptr = threadIdx.y + blockDim.y * threadIdx.z +
                             blockIdx.x * blockDim.y * blockDim.z;
         neigh_ptr < degu; neigh_ptr += blockDim.y * blockDim.z * gridDim.x) {
      IDType v = adj[neigh_ptr + xadj[u]];
      bool skippable = (xadj[v + 1] - xadj[v] < degu ||
                        (xadj[v + 1] - xadj[v] == degu && v > u));
      if (!directed && skippable)
        continue;
      NNZType other_ptr = bst(xadj, adj, v, u);
      if (directed && other_ptr != (NNZType)-1 && skippable)
        continue;
      NNZType intersection_size = 0;

      for (NNZType t_ptr = threadIdx.x; t_ptr < degu; t_ptr += blockDim.x) {
        NNZType loc = bst(xadj, adj, v, adj[xadj[u] + t_ptr]);
        intersection_size += (loc != (NNZType)-1);
      }

      intersection_size = warpReduce(intersection_size, blockDim.x, mask);
      // intersection_size = warpReduce(intersection_size, blockDim.x, tid%32,
      // threadIdx.x==0);
      if (threadIdx.x == 0) {
        FeatureType J =
            float(intersection_size) /
            float(degu + (xadj[v + 1] - xadj[v]) - intersection_size);
        emetrics[(xadj[u] + neigh_ptr)] = J;
        if (other_ptr != (NNZType)-1)
          emetrics[other_ptr] = J;
      }
    }
    //    __syncthreads();
  }
}

#if !defined(_HEADER_ONLY)
#include "init/external/cuda/preprocess.inc"
#endif
} // namespace cuda
} // namespace preprocess

} // namespace sparsebase