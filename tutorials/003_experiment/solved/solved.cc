#include <iostream>

#include <omp.h>
#include <unistd.h>
#include <random>

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
std::any spmv(unordered_map<string, Format *> & data, std::any & fparams, std::any & pparams, std::any & kparams){
  auto v_arr = any_cast<Array<float>>(fparams);
  auto vals_arr = any_cast<Array<float>>(kparams);
  auto vals = vals_arr.get_vals();
  auto v = v_arr.get_vals();
  auto spm = data["processed_format"]->AsAbsolute<CSR<row_type, nnz_type, value_type>>();
  auto dimensions = spm->get_dimensions();
  auto num_rows = dimensions[0];
  auto rows = spm->get_row_ptr();
  auto cols = spm->get_col();

  auto res = new float[num_rows]();
  for(unsigned int i = 0; i < num_rows; i++){
    float s = 0;
    for(unsigned int j = rows[i]; j < rows[i+1]; j++){
      s += vals[j] * v[cols[j]];
    }
    res[i] = s;
  }

  return res;
}


//omp parallel spmv
std::any spmv_par(unordered_map<string, Format*> & data, std::any & fparams, std::any & pparams, std::any & kparams){
  auto v_arr = any_cast<Array<float>>(fparams);
  auto vals_arr = any_cast<Array<float>>(kparams);
  auto vals = vals_arr.get_vals();
  auto v = v_arr.get_vals();
  auto spm = data["processed_format"]->AsAbsolute<CSR<row_type, nnz_type, value_type>>();
  auto dimensions = spm->get_dimensions();
  auto num_rows = dimensions[0];
  auto rows = spm->get_row_ptr();
  auto cols = spm->get_col();

  auto res = new float[num_rows]();
#pragma omp parallel for schedule(dynamic)
  for(unsigned int i = 0; i < num_rows; i++){
    float s = 0;
    for(unsigned int j = rows[i]; j < rows[i+1]; j++){
      s += vals[j] * v[cols[j]];
    }
    res[i] = s;
  }

  return res;
}

void fill_r(float * arr, unsigned int size){
  default_random_engine rnd{std::random_device{}()};
  uniform_real_distribution<float> dist(-1, 1);
  for(unsigned int i = 0; i < size; i++){
    arr[i] = dist(rnd);
  }
}

int main(int argc, char **argv){

  experiment::ConcreteExperiment exp;

  // add data loaders for the files we will use
  // also init the vector for the kernels and pass it as a file specific parameter
  auto v = new float[958];
  fill_r(v, 958);
  auto ash_v = Array<float>(958, v);
  v = new float[317080];
  fill_r(v, 317080);
  auto dblp_v = Array<float>(317080, v);
  vector<string> files = {argv[1]};
  exp.AddDataLoader(experiment::LoadCSR<MTXReader, row_type, nnz_type, value_type>, {make_pair(files, ash_v)});
  files = {argv[2]};
  exp.AddDataLoader(experiment::LoadCSR<EdgeListReader, row_type, nnz_type, value_type>, {make_pair(files, dblp_v)});

  // add reordering as a preprocess
  // also add the experiment::Pass function as a preprocessing function to be able to run your kernels on the original data
  exp.AddPreprocess("original", experiment::Pass, {}); // add dummy preprocessing to run kernels without reordering
  RCMReorder<row_type, nnz_type, value_type>::ParamsType params = {};
  exp.AddPreprocess("RCM", experiment::Reorder<RCMReorder, CSR, CPUContext, row_type, nnz_type, value_type>, params);

  // add kernels that will carry out the SPMV
  // init random vals large enough for all the files and pass it as a kernel specific parameter
  auto vals = new float[1049866];
  fill_r(vals, 1049866);
  auto vals_v = Array<float>(1049866, vals);
  exp.AddKernel("single-threaded", spmv, vals_v);
  exp.AddKernel("omp-parallel", spmv_par, vals_v);

  // run the experiment
  exp.Run(NUM_RUNS, true);

  cout << endl;

  // check results
  cout << "Results: " << endl;
  auto res = exp.GetResults();
  for(auto r: res){
    cout << r.first << ": ";
    auto result = any_cast<float*>(r.second[0]);
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
