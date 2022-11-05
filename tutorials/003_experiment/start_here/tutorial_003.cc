#include <iostream>

#include <omp.h>
#include <unistd.h>
#include <random>

#include "sparsebase/format/format.h"
#include "sparsebase/preprocess/preprocess.h"
#include "sparsebase/io/io.h"
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

  // create the experiment
  /*
   * YOUR CODE GOES HERE
   */

  // add data loaders for the files we will use
  // also init the vector for the kernels and pass it as a file specific parameter

  /*
   * YOUR CODE GOES HERE
   */

  // add reordering as a preprocess
  // also add the experiment::Pass function as a preprocessing function to be able to run your kernels on the original data

  /*
   * YOUR CODE GOES HERE
   */

  // add kernels that will carry out the SPMV
  // init random vals large enough for all the files and pass it as a kernel specific parameter

  /*
   * YOUR CODE GOES HERE
   */

  // run the experiment

  /*
   * YOUR CODE GOES HERE
   */

  // check results

  /*
   * YOUR CODE GOES HERE
   */

  // get auxiliary data created during the experiment

  /*
   * YOUR CODE GOES HERE
   */

  // display runtimes

  /*
   * YOUR CODE GOES HERE
   */

  return 0;
}
