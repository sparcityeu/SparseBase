#include "sparsebase/format/format.h"
#include "sparsebase/io/iobase.h"
#include "sparsebase/preprocess/preprocess.h"
#include "sparsebase/context/context.h"
#include <string>
#include <iostream>
#include <chrono>
#include <algorithm>
#include <random>

typedef unsigned int id_type;
typedef unsigned int nnz_type;
typedef float value_type;

using namespace sparsebase;
using namespace io;
using namespace preprocess;
using namespace format;

#define THREADS_PER_BLOCK 512
#define NUM_BLOCKS 512
#define NUM_THREADS (THREADS_PER_BLOCK*NUM_BLOCKS)
#define WARP_SIZE 32

__global__ void spmv_1w1r(nnz_type * matrix_row_ptr, id_type * matrix_col, value_type * matrix_vals, value_type * vec_vals, value_type * res, id_type n);
void filL_randomly(value_type *ptr, id_type size);

int main(int argc, char * argv[]){

  // The name of the edge list file in disk
  std::string matrix_filename(argv[1]);
  // Read the edge list file into a CSR object
  CSR<id_type, nnz_type, value_type>* csr = IOBase::ReadMTXToCSR<id_type, nnz_type, value_type>(matrix_filename);

  // get_dimensions() returns a vector with the dimension of
  // each order of the format object
  id_type num_rows = csr->get_dimensions()[0];
  id_type num_columns = csr->get_dimensions()[1];
  nnz_type num_non_zeros = csr->get_num_nnz();

  std::cout << "Matrix has "
            << num_rows << " rows, "
            << num_columns << " columns, and "
            << num_non_zeros << " non-zeros" << std::endl;

  // Raw C++ array that will contain the data
  value_type * vec_ptr = new value_type[num_columns];
  // Fill the vector with random values
  filL_randomly(vec_ptr, num_columns);
  // Create a SparseBase array as a wrapper around the raw array `ptr`.
  Array<value_type>* vec = new Array<value_type>(num_columns, vec_ptr, kOwned);
  // We can access metadata of the Array
  std::cout << "Vector has "
            << vec->get_dimensions()[0] << " elements " <<  std::endl;

  // Context representing the GPU with ID 0 in the system
  context::CUDAContext gpu0{0};

  // The conversion target is passed as a template parameter,
  // and the context to convert it to is the parameter.
  cuda::CUDACSR<id_type, nnz_type, value_type>* cuda_csr = csr->Convert<cuda::CUDACSR>(&gpu0);
  cuda::CUDAArray<value_type>* cuda_array = vec->Convert<cuda::CUDAArray>(&gpu0);

  value_type * result_ptr;
  // Allocate the memory using the native CUDA call
  cudaMalloc(&result_ptr, num_rows * sizeof(value_type));

  // These pointers point at data on the GPU
  nnz_type* matrix_row_ptr = cuda_csr->get_row_ptr();
  id_type* matrix_col = cuda_csr->get_col();
  value_type* matrix_vals = cuda_csr->get_vals();

  value_type* vector_vals = cuda_array->get_vals();

  auto start = std::chrono::system_clock::now();
  spmv_1w1r<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(matrix_row_ptr, matrix_col, matrix_vals, vector_vals, result_ptr, num_rows);
  cudaDeviceSynchronize();
  if (!cudaPeekAtLastError() == cudaSuccess){
    std::cout << "Kernel failed: " << cudaGetErrorString(cudaPeekAtLastError()) << std::endl;
    return 1;
  }
  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> total_time = end-start;
  std::cout << "SpMV without reordering takes: " << total_time.count() << " seconds" << std::endl;

  // A context representing the host system
  context::CPUContext cpu;
  // Create a parameters object to store special parameters specific
  // Gray reordering
  GrayReorderParams gray_params(BitMapSize::BitSize32, 32, 32);
  // Create an inverse permutation array  of the matrix `csr`
  id_type * gray_reorder = ReorderBase::Reorder<GrayReorder>(gray_params, csr, {&cpu}, true);

  // `Permute2D` returns a CSR object but stores it in a polymorphic
  // pointer at the superclass for two-dimensional formats, FormatOrderTwo.
  FormatOrderTwo<id_type, nnz_type, value_type>* gray_reordered_csr = ReorderBase::Permute2D(gray_reorder, csr, {&cpu}, true);
  // We move the reordered CSR to the GPU.
  cuda::CUDACSR<id_type, nnz_type, value_type>* cuda_gray_reordered_csr = gray_reordered_csr ->Convert<cuda::CUDACSR>(&gpu0);
  // Rather than get the generic pointer to `FormatOrderOne`, we can cast the output
  // to the correct type in the same call to `Permute1D`
  Array<value_type>* gray_reordered_vec = ReorderBase::Permute1D<Array>(gray_reorder, vec, {&cpu}, true);
  // We move this array to the GPU.
  cuda::CUDAArray<value_type>* cuda_gray_reordered_vec = gray_reordered_vec->Convert<cuda::CUDAArray>(&gpu0);

  start = std::chrono::system_clock::now();
  spmv_1w1r<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(cuda_gray_reordered_csr ->get_row_ptr(),
                                         cuda_gray_reordered_csr->get_col(),
                                         cuda_gray_reordered_csr->get_vals(),
                                         cuda_gray_reordered_vec->get_vals(),
                                         result_ptr, num_rows);
  cudaDeviceSynchronize();
  if (!cudaPeekAtLastError() == cudaSuccess){
    std::cout << "Kernel failed: " << cudaGetErrorString(cudaPeekAtLastError()) << std::endl;
    return 1;
  }
  end = std::chrono::system_clock::now();
  total_time = end-start;
  std::cout << "SpMV with Gray reordering takes: " << total_time.count() << " seconds" << std::endl;

  // We can pass the CUDACSR and the function will automatically convert it to CSR for reordering
  id_type* rcm_reorder = ReorderBase::Reorder<RCMReorder>({}, cuda_csr, {&cpu}, true);

  // A list of available contexts
  std::vector<context::Context*> contexts = {&cpu, &gpu0};
  // We can apply the permutation to the CUDACSR directly, as well, but the returned
  // object will be a CSR since permutation will run on a CSR rathar than a CUDACSR
  auto rcm_reordered_csr = ReorderBase::Permute2D<CSR>(rcm_reorder, cuda_csr, contexts, true);
  auto cuda_rcm_reordered_csr = rcm_reordered_csr->Convert<cuda::CUDACSR>(&gpu0);

  auto rcm_reordered_vec = ReorderBase::Permute1D<Array>(rcm_reorder, cuda_array, contexts, true);
  auto cuda_rcm_reordered_vec = rcm_reordered_vec->Convert<cuda::CUDAArray>(&gpu0);

  start = std::chrono::system_clock::now();
  spmv_1w1r<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(cuda_rcm_reordered_csr->get_row_ptr(),
                                         cuda_rcm_reordered_csr->get_col(),
                                         cuda_rcm_reordered_csr->get_vals(),
                                         cuda_rcm_reordered_vec->get_vals(),
                                         result_ptr, num_rows);
  cudaDeviceSynchronize();
  if (!cudaPeekAtLastError() == cudaSuccess){
    std::cout << "Kernel failed: " << cudaGetErrorString(cudaPeekAtLastError()) << std::endl;
    return 1;
  }
  end = std::chrono::system_clock::now();
  total_time = end-start;
  std::cout << "SpMV with RCM reordering takes: " << total_time.count() << " seconds" << std::endl;

  // `reverse_reorder` can be used to undo the permutation done using `rcm_reorder`
  id_type * reverse_rcm_reorder = ReorderBase::InversePermutation(rcm_reorder, num_rows);

  // We place the raw array in a CUDAArray to use Permute1D
  cuda::CUDAArray<value_type> cuda_rcm_reordered_result(num_rows, result_ptr, gpu0, kOwned);
  // Since `Permute1D` is only implemented for `Array`, the output will be an `Array`
  FormatOrderOne<value_type>* foo_result = ReorderBase::Permute1D(reverse_rcm_reorder, &cuda_rcm_reordered_result, contexts, true);
  // Safely cast the pointer at the parent `FormatOrderOne` to `Array`
  Array<value_type>* arr_result = foo_result->As<Array>();



  return 0;
}

__global__ void spmv_1w1r(nnz_type * matrix_row_ptr, id_type * matrix_col, value_type * matrix_vals, value_type * vec_vals, value_type * res, id_type n) {
  int thread_id = threadIdx.x + (blockDim.x * blockIdx.x);
  int rid = thread_id / WARP_SIZE;
  int ridx = thread_id % WARP_SIZE;
  id_type s;
  for(; rid < n; rid += (NUM_THREADS)/WARP_SIZE){
    s = 0;
    for(unsigned int j = matrix_row_ptr[rid]+ridx; j < matrix_row_ptr[rid+1]; j+=WARP_SIZE){
      s += matrix_vals[j]*vec_vals[matrix_col[j]];
    }
    atomicAdd(res+rid, s);
    //res[rid] += s;
  }
}

void filL_randomly(value_type *ptr, id_type size) {
  std::default_random_engine rd;
  std::uniform_real_distribution<value_type> dist(-10, 10);

  for (id_type i = 0; i < size; i++){
    ptr[i] = dist(rd);
  }
}