#include <iostream>

#include <unistd.h>

#include "sparsebase/format/format.h"
#include "sparsebase/preprocess/preprocess.h"
#include "sparsebase/utils/io/reader.h"
#include "sparsebase/experiment/experiment.h"

using namespace std;
using namespace sparsebase;
using namespace format;
using namespace preprocess;
using namespace utils;
using namespace io;
using namespace context;

using row_type = unsigned int;
using nnz_type = unsigned int;
using value_type = unsigned int;

#define NUM_RUNS 1

// single threaded spmv
std::any spmv(unordered_map<string, Format *> & data, std::any params){
  auto v = any_cast<row_type*>(params);
  auto spm = data["processed_format"]->As<CSR<row_type, nnz_type, value_type>>();
  auto dimensions = spm->get_dimensions();
  auto num_rows = dimensions[0];
  auto rows = spm->get_row_ptr();
  auto cols = spm->get_col();

  auto * res = new row_type[num_rows];
  for(unsigned int i = 0; i < num_rows; i++){
    row_type s = 0;
    for(unsigned int j = rows[i]; j < rows[i+1]; j++){
      auto idx = cols[j];
      s += v[idx];
    }
    res[i] = s;
  }

  return res;
}


//omp parallel spmv
std::any spmv_par(unordered_map<string, Format*> & data, std::any params){
  auto v = any_cast<row_type*>(params);
  auto spm = data["processed_format"]->As<CSR<row_type, nnz_type, value_type>>();
  auto dimensions = spm->get_dimensions();
  auto num_rows = dimensions[0];
  auto rows = spm->get_row_ptr();
  auto cols = spm->get_col();

  auto * res = new row_type[num_rows];
  #pragma omp parallel for schedule(dynamic, 32)
  for(unsigned int i = 0; i < num_rows; i++){
    row_type s = 0;
    for(unsigned int j = rows[i]; j < rows[i+1]; j++){
      auto idx = cols[j];
      s += v[idx];
    }
    res[i] = s;
  }

  return res;
}

#ifdef USE_CUDA

using namespace format::cuda;
using namespace context::cuda;

#define THREADS_PER_BLOCK 1
#define NUM_BLOCKS 1
#define NUM_THREADS (THREADS_PER_BLOCK*NUM_BLOCKS)
#define WARP_SIZE 32

__global__ void spmv_kernel(nnz_type * row_ptr, row_type * cols, row_type * v, row_type * res, row_type n){
  int thread_id = threadIdx.x + (blockDim.x * threadIdx.y);
  row_type s = 0;
  for(unsigned int j = row_ptr[thread_id]; j < row_ptr[thread_id+1]; j++){
    auto idx = cols[j];
    s += v[idx];
  }
  res[thread_id] = s;
}

// many-core accelerated spmv
std::any spmv_gpu(unordered_map<string, Format *> & data, std::any params){
  CUDAContext gpu_context{0};
  auto array_converter = new utils::converter::ConverterOrderOne<row_type>();

  auto r =  new row_type[958]();
  auto s_r = Array<row_type>(958, r);
  auto c_r = s_r.Convert<CUDAArray>(&gpu_context);

  auto v = any_cast<row_type*>(params);
  auto s_v = Array<row_type>(958, v);
  auto c_v = s_v.Convert<CUDAArray>(&gpu_context);

  auto spm = data["processed_format"]->As<CSR<row_type, nnz_type, value_type>>();
  auto c_spm = spm->Convert<CUDACSR>(&gpu_context);
  auto dimensions = c_spm->get_dimensions();
  auto num_rows = dimensions[0];
  auto rows = c_spm->get_row_ptr();
  auto cols = c_spm->get_col();
  dim3 tpb(num_rows, 1);

  spmv_kernel<<<NUM_BLOCKS, tpb>>>(rows, cols, c_v->get_vals(), c_r->get_vals(), num_rows);

  CPUContext cpu_context;
  auto r_r = c_r->Convert<Array>(&cpu_context);
  return r_r->get_vals();
}
#endif

int main(int argc, char **argv){

  experiment::ConcreteExperiment exp;

  // load all the matrices
  for(unsigned int i = 1; i < argc; i++){
    exp.AddDataLoader(experiment::LoadCSR<MTXReader, row_type, nnz_type, value_type>, {argv[i]});
  }

  // reorder matrices
  exp.AddPreprocess(experiment::Pass<row_type, nnz_type, value_type>); // add dummy preprocessing to run kernels without reordering
  exp.AddPreprocess(experiment::Reorder<RCMReorder, CSR, CPUContext, row_type, nnz_type, value_type>);

  // init the vector for the kernels
  auto v = new row_type[958];
  std::fill_n(v, 958, 1);

  // add kernels that will carry out the sparse matrix vector multiplication
  exp.AddKernel(spmv, v);
  exp.AddKernel(spmv_par, v);
#ifdef USE_CUDA
 exp.AddKernel(spmv_gpu, v);
#endif

  // start the experiment
  exp.Run(NUM_RUNS);

  // check if results are correct
  std::vector<std::any> res = exp.GetResults();
  for(auto r: res){
    auto result = any_cast<row_type*>(r);
    for(unsigned int t = 0; t < 958; t++){
      cout << result[958-t] << " ";
    }
    cout << endl;
  }

  // get auxiliary data

  // display runtimes
  auto secs = exp.GetRunTimes();
  for(auto s: secs){
    cout << "Run time: " << s << endl;
  }

  return 0;
}
