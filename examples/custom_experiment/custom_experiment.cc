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

using vertex_type = unsigned int;
using edge_type = unsigned int;
using value_type = unsigned int;

#define NUM_RUNS 1

// create custom dataLoader
unordered_map<string, Format*> get_data_f( string & file_name ) {
  CSR<vertex_type, edge_type, value_type> * csr = MTXReader<vertex_type, edge_type, value_type>(file_name)
      .ReadCSR();
  unordered_map<string, Format*> r;
  r.emplace("graph", csr);
  return r;
};

// create custom preprocess function
void preprocess_f(unordered_map<string, Format*> & data) {
  context::CPUContext cpu_context;
  auto perm = ReorderBase::Reorder<RCMReorder>({}, data["graph"]->As<CSR<vertex_type, edge_type, value_type>>(), {&cpu_context}, true);
  auto A_reordered = ReorderBase::Permute2D<CSR>(perm, data["graph"]->As<CSR<vertex_type, edge_type, value_type>>(), {&cpu_context}, true);
  //auto *A_csr = A_reordered->Convert<CSR>();
  data.emplace("ordered", A_reordered);
};

// kernel function is always provided by the user
// this kernel carries out an spmv
std::any kernel_f(unordered_map<string, Format*> & data, std::any params) {
  auto v = any_cast<double*>(params);
  auto spm = data["ordered"]->As<CSR<vertex_type, edge_type, value_type>>();
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
};

int main(int argc, char **argv){

  experiment::ConcreteExperiment exp;
  string file_name = argv[1];
  // add custom dataLoader
  exp.AddDataLoader(get_data_f, {file_name});
  // add custom preprocessing
  exp.AddPreprocess(preprocess_f);

  auto v = new double[958];
  std::fill_n(v, 958, 1);
  // since the vector is all 1s, this kernel calculates the number of nnz per row
  exp.AddKernel(kernel_f, v);

  exp.Run(NUM_RUNS);

  std::vector<std::any> res = exp.GetResults();
  cout << "Size = " << res.size() << endl;
  auto secs = exp.GetRunTimes();
  for(auto s: secs){
    cout << "Run time: " << s << endl;
  }
  return 0;
}
