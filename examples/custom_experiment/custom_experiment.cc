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

// create custom dataLoader
unordered_map<string, Format*> get_data_f( vector<string> & file_names ) {
  CSR<vertex_type, edge_type, value_type> * csr = MTXReader<vertex_type, edge_type, value_type>(file_names[0])
      .ReadCSR();
  unordered_map<string, Format*> r;
  r.emplace("graph", csr);
  return r;
};

// create custom preprocess function
void preprocess_f(unordered_map<string, Format*> & data, std::any params) {
  context::CPUContext cpu_context;
  auto perm = ReorderBase::Reorder<RCMReorder>({}, data["graph"]->AsAbsolute<CSR<vertex_type, edge_type, value_type>>(), {&cpu_context}, true);
  auto A_reordered = ReorderBase::Permute2D<CSR>(perm, data["graph"]->AsAbsolute<CSR<vertex_type, edge_type, value_type>>(), {&cpu_context}, true);
  //auto *A_csr = A_reordered->Convert<CSR>();
  data.emplace("ordered", A_reordered);
};

// kernel function is always provided by the user
// this kernel carries out an spmv
std::any kernel_f(unordered_map<string, Format*> & data, std::any fparams, std::any pparams, std::any kparams) {
  auto v = any_cast<double*>(fparams);
  auto spm = data["ordered"]->AsAbsolute<CSR<vertex_type, edge_type, value_type>>();
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
  vector<string> file_names = {argv[1]};
  string file_name = argv[1];
  auto v = new double[958];
  std::fill_n(v, 958, 1);
  // add custom dataLoader
  exp.AddDataLoader(get_data_f, {make_pair(file_names, v)});

  // experiment functions are wrapped with std::function, thus can be any allable function that abides by the function definition.
  auto get_data_l = [file_name] ( vector<string> & file_names ) {
    CSR<vertex_type, edge_type, value_type> * csr = MTXReader<vertex_type, edge_type, value_type>(file_name)
        .ReadCSR();
    unordered_map<string, Format*> r;
    r.emplace("graph", csr);
    return r;
  };
  exp.AddDataLoader(get_data_l, {make_pair(file_names, v)});
  // add custom preprocessing
  exp.AddPreprocess("mypreprocess", preprocess_f, {});

  // since the vector is all 1s, this kernel calculates the number of nnz per row
  exp.AddKernel("mykernel", kernel_f, {});

  exp.Run();

  auto res = exp.GetResults();
  cout << "# of results = " << res.size() << endl;
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
