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

int main(int argc, char **argv){

  experiment::ConcreteExperiment exp;
  auto get_data_f = [] ( string file_name ) {
    CSR<vertex_type, edge_type, value_type> * csr = MTXReader<vertex_type, edge_type, value_type>(file_name)
        .ReadCSR();
    unordered_map<string, Format*> r;
    r.emplace("graph", csr);
    auto format = csr->get_format_id();
    auto dimensions = csr->get_dimensions();
    auto row_ptr2 = csr->get_row_ptr();
    auto col2 = csr->get_col();
    auto vals = csr->get_vals();
    cout << "Format: " << format.name() << endl;
    cout << "# of dimensions: " << dimensions.size() << endl;
    for (int i = 0; i < dimensions.size(); i++) {
      cout << "Dim " << i << " size " << dimensions[i] << endl;
    }
    return r;
  };
  string file_name = argv[1];
  auto data = get_data_f(file_name);
  exp.AddDataLoader(get_data_f, {file_name});

  auto preprocess_f = [] (unordered_map<string, Format*> data) {
    cout << "PREPROCESS LAMBDA FUNCTION!!" << endl;
    unordered_map<string, Format*> r;
    context::CPUContext cpu_context;
    auto *perm = ReorderBase::Reorder<RCMReorder>({}, data["graph"]->As<CSR<vertex_type, edge_type, value_type>>(), {&cpu_context});
    //r.emplace("permutation", perm);
    auto * A_reordered = ReorderBase::Permute2D<CSR>(perm, data["graph"]->As<CSR<vertex_type, edge_type, value_type>>(), {&cpu_context});
    auto *A_csc = A_reordered->Convert<CSR>();
    r.emplace("ordered", A_csc);
    return r;
  };
  data = preprocess_f(data);
  exp.AddPreprocess(preprocess_f);

  auto kernel_f = [] (unordered_map<string, Format*> data) {
    cout << "DO WHATEVER YOU WANT WITH DATA!!!" << endl;
    context::CPUContext cpu_context;
    auto *perm = ReorderBase::Reorder<RCMReorder>({}, data["ordered"]->As<CSR<vertex_type, edge_type, value_type>>(), {&cpu_context});
    auto * A_reordered = ReorderBase::Permute2D<CSR>(perm, data["ordered"]->As<CSR<vertex_type, edge_type, value_type>>(), {&cpu_context});
    auto * A_csc = A_reordered->Convert<CSC>();
    std::any rt = A_csc;
    return rt;
  };
  auto r = kernel_f(data);
  exp.AddKernel(kernel_f);
  exp.AddKernel(kernel_f);
  exp.Run(2);
  std::vector<std::any> res = exp.GetResults();
  cout << "Size = " << res.size() << endl;
  auto secs = exp.GetRunTimes();
  for(auto s: secs){
    cout << "Run time: " << s << endl;
  }
  return 0;
}
