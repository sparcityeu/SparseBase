#ifndef SPARSEBASE_PROJECT_UTILS_CUDA_CUH
#define SPARSEBASE_PROJECT_UTILS_CUDA_CUH

namespace sparsebase::utils {

template <typename T>
struct CUDADeleter {
  void operator()(T *obj) { cudaFree(obj); }
};

}

#endif  // SPARSEBASE_PROJECT_UTILS_CUDA_CUH
