#include "sparsebase/format/csr.h"
#include "sparsebase/format/cuda_csr_cuda.cuh"
#include "sparsebase/format/array.h"
#include "sparsebase/format/cuda_array_cuda.cuh"
#include "sparsebase/reorder/gray_reorder.h"
#include "sparsebase/reorder/rcm_reorder.h"
#include "sparsebase/bases/iobase.h"
#include "sparsebase/bases/reorder_base.h"
#include "sparsebase/context/cpu_context.h"
#include "sparsebase/context/cuda_context_cuda.cuh"
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
using namespace format;
using namespace reorder;

#define THREADS_PER_BLOCK 512
#define NUM_BLOCKS 512
#define NUM_THREADS (THREADS_PER_BLOCK*NUM_BLOCKS)
#define WARP_SIZE 32

__global__ void spmv_1w1r(nnz_type * matrix_row_ptr, id_type * matrix_col, value_type * matrix_vals, value_type * vec_vals, value_type * res, id_type n);
void filL_randomly(value_type *ptr, id_type size);

int main(int argc, char * argv[]){
  if (argc < 2){
    std::cout << "Please enter the name of the edgelist file as a parameter\n";
    return 1;
  }


  ///// YOUR CODE GOES HERE /////

  ///////////////////////////////
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
