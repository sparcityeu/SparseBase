#include <iostream>

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

using vertex_type = unsigned int;
using edge_type = unsigned int;
using value_type = unsigned int;

#define NUM_RUNS 1

std::any spmv(unordered_map<string, Format *> & data, std::any params){
  auto v = any_cast<double*>(params);
  auto spm = data["processed_format"]->As<CSR<vertex_type, edge_type, value_type>>();
  auto dimensions = spm->get_dimensions();
  auto num_rows = dimensions[0];
  auto rows = spm->get_row_ptr();
  auto cols = spm->get_col();

  auto * res = new double[num_rows];
  for(unsigned int i = 0; i < num_rows; i++){
    double s = 0;
    for(unsigned int j = rows[i]; j < rows[i+1]; j++){
      auto idx = cols[j];
      s += v[idx];
    }
    res[i] = s;
  }

  return res;
}

std::any spmv_gpu(unordered_map<string, Format *> & data, std::any params){
  std::any r;
  return r;
}

int main(int argc, char **argv){

  experiment::ConcreteExperiment cpu_exp;
  for(unsigned int i = 1; i < argc; i++){
    cpu_exp.AddDataLoader(experiment::LoadCSR<MTXReader, vertex_type, edge_type, value_type>, {argv[i]});
  }
  cpu_exp.AddPreprocess(experiment::Pass<vertex_type, edge_type, value_type>); // add dummy preprocessing to run kernels without preprocessing
  cpu_exp.AddPreprocess(experiment::ReorderCSR<RCMReorder, CPUContext, vertex_type, edge_type, value_type>);
  cpu_exp.AddPreprocess(experiment::Reorder<RCMReorder, CSR, CPUContext, vertex_type, edge_type, value_type>);

  auto v = new double[958];
  std::fill_n(v, 958, 1);
  cpu_exp.AddKernel(spmv, v);

  cpu_exp.Run(NUM_RUNS);

  std::vector<std::any> res = cpu_exp.GetResults();
  for(auto r: res){
    auto result = any_cast<double*>(r);
    for(unsigned int t = 0; t < 958; t++){
      cout << result[958-t] << " ";
    }
    cout << endl;
  }

  auto secs = cpu_exp.GetRunTimes();
  for(auto s: secs){
    cout << "Run time: " << s << endl;
  }
  return 0;
}
