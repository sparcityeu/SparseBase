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
std::any spmv(unordered_map<string, Format *> & data, std::any fparams, std::any pparams, std::any kparams){
  auto v_arr = any_cast<Array<row_type>>(fparams);
  auto v = v_arr.get_vals();
  auto spm = data["processed_format"]->AsAbsolute<CSR<row_type, nnz_type, value_type>>();
  auto dimensions = spm->get_dimensions();
  auto num_rows = dimensions[0];
  auto rows = spm->get_row_ptr();
  auto cols = spm->get_col();

  auto res = new row_type[num_rows]();
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
std::any spmv_par(unordered_map<string, Format*> & data, std::any fparams, std::any pparams, std::any kparams){
  auto v_arr = any_cast<Array<row_type>>(fparams);
  auto v = v_arr.get_vals();
  auto spm = data["processed_format"]->AsAbsolute<CSR<row_type, nnz_type, value_type>>();
  auto dimensions = spm->get_dimensions();
  auto num_rows = dimensions[0];
  auto rows = spm->get_row_ptr();
  auto cols = spm->get_col();

  auto res = new row_type[num_rows]();
  #pragma omp parallel for // schedule(dynamic)
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

#define THREADS_PER_BLOCK 1024
#define NUM_BLOCKS 512
#define NUM_THREADS (THREADS_PER_BLOCK*NUM_BLOCKS)
#define WARP_SIZE 32

__global__ void spmv_1t1r(nnz_type * row_ptr, row_type * cols, row_type * v, row_type * res, row_type n){
  int thread_id = threadIdx.x + (blockDim.x * blockIdx.y);
  row_type s;
  for(unsigned int t = thread_id; t < n; t += THREADS_PER_BLOCK*NUM_BLOCKS){
    s = 0;
    for(unsigned int j = row_ptr[t]; j < row_ptr[t+1]; j++){
      auto idx = cols[j];
      s += v[idx];
    }
    res[t] = s;
  }
}

__global__ void spmv_1w1r(nnz_type * row_ptr, row_type * cols, row_type * v, row_type * res, row_type n){
  int thread_id = threadIdx.x + (blockDim.x * blockIdx.x);
  int rid = thread_id / WARP_SIZE;
  int ridx = thread_id % WARP_SIZE;
  row_type s;
  for(; rid < n; rid += (NUM_THREADS)/WARP_SIZE){
    s = 0;
    for(unsigned int j = row_ptr[rid]+ridx; j < row_ptr[rid+1]; j+=WARP_SIZE){
      s += v[cols[j]];
    }
    atomicAdd(res+rid, s);
  }
}

// many-core accelerated spmv
std::any spmv_gpu(unordered_map<string, Format *> & data, std::any fparams, std::any pparams, std::any kparams){
  CUDAContext gpu_context{0};
  auto s_v = any_cast<Array<row_type>>(fparams);
  auto c_v = s_v.Convert<CUDAArray>(&gpu_context);

  auto size = s_v.get_dimensions()[0];
  auto r =  new row_type[size]();
  auto s_r = Array<row_type>(size, r);
  auto c_r = s_r.Convert<CUDAArray>(&gpu_context);

  auto spm = data["processed_format"]->AsAbsolute<CSR<row_type, nnz_type, value_type>>();
  auto c_spm = spm->Convert<CUDACSR>(&gpu_context);
  auto dimensions = c_spm->get_dimensions();
  auto num_rows = dimensions[0];
  auto rows = c_spm->get_row_ptr();
  auto cols = c_spm->get_col();

  spmv_1t1r<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(rows, cols, c_v->get_vals(), c_r->get_vals(), num_rows);

  CPUContext cpu_context;
  auto r_r = c_r->Convert<Array>(&cpu_context);
  return r_r->get_vals();
}

// many-core accelerated spmv
std::any spmv_gpu2(unordered_map<string, Format *> & data, std::any fparams, std::any pparams, std::any kparams){
  CUDAContext gpu_context{0};
  auto s_v = any_cast<Array<row_type>>(fparams);
  auto c_v = s_v.Convert<CUDAArray>(&gpu_context);

  auto size = s_v.get_dimensions()[0];
  auto r =  new row_type[size]();
  auto s_r = Array<row_type>(size, r);
  auto c_r = s_r.Convert<CUDAArray>(&gpu_context);

  auto spm = data["processed_format"]->AsAbsolute<CSR<row_type, nnz_type, value_type>>();
  auto c_spm = spm->Convert<CUDACSR>(&gpu_context);
  auto dimensions = c_spm->get_dimensions();
  auto num_rows = dimensions[0];
  auto rows = c_spm->get_row_ptr();
  auto cols = c_spm->get_col();

  spmv_1w1r<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(rows, cols, c_v->get_vals(), c_r->get_vals(), num_rows);

  CPUContext cpu_context;
  auto r_r = c_r->Convert<Array>(&cpu_context);
  return r_r->get_vals();
}
#endif
int main(int argc, char **argv){

  experiment::ConcreteExperiment exp;

  // load matrices
  // init the vector for the kernels
  auto v = new row_type[958];
  std::fill_n(v, 958, 1);
  auto ash_v = Array<row_type>(958, v);
  v = new row_type[317080];
  std::fill_n(v, 317080, 1);
  auto dblp_v = Array<row_type>(317080, v);
  vector<string> files = {argv[1]};
  exp.AddDataLoader(experiment::LoadCSR<MTXReader, row_type, nnz_type, value_type>, {make_pair(files, ash_v)});
  files = {argv[2]};
  exp.AddDataLoader(experiment::LoadCSR<EdgeListReader, row_type, nnz_type, value_type>, {make_pair(files, dblp_v)});

  // reorder matrices
  exp.AddPreprocess("original", experiment::Pass, {}); // add dummy preprocessing to run kernels without reordering
  RCMReorder<row_type, nnz_type, value_type>::ParamsType params = {};
  exp.AddPreprocess("RCM", experiment::Reorder<RCMReorder, CSR, CPUContext, row_type, nnz_type, value_type>, params);

  // add kernels that will carry out the sparse matrix vector multiplication
  exp.AddKernel("single-threaded", spmv, {});
  exp.AddKernel("omp-parallel", spmv_par, {});
#ifdef USE_CUDA
   exp.AddKernel("cuda-1t1r", spmv_gpu, {});
   exp.AddKernel("cuda-1w1r", spmv_gpu2, {});
#endif

  // start the experiment
  exp.Run(NUM_RUNS, true);

  cout << endl;

  // check results
  cout << "Results: " << endl;
  auto res = exp.GetResults();
  for(auto r: res){
    cout << r.first << ": ";
    auto result = any_cast<row_type*>(r.second[0]);
    for(unsigned int t = 0; t < 50; t++){
      cout << result[t] << " ";
    }
    cout << endl;
  }

  cout << endl;

  // get auxiliary data created during the experiment
  auto auxiliary = exp.GetAuxiliary();
  cout << "Auxiliary Data: " << endl;
  for(const auto & a: auxiliary){
    cout << a.first << endl;
  }

  cout << endl;

  // display runtimes
  auto secs = exp.GetRunTimes();
  cout << "Runtimes: " << endl;
  for(const auto & s: secs){
    cout << s.first << ": ";
    for(auto sr: s.second){
      cout << sr << " ";
    }
    cout << endl;
  }

  return 0;
}
