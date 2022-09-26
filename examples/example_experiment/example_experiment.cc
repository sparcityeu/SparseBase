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

std::any kernel_f2(unordered_map<string, Format *> data){
  cout << "In function kernel_f2" << endl;
  std::any r;
  return r;
}

int main(int argc, char **argv){

  experiment::ConcreteExperiment exp;
  string file_name = argv[1];
  exp.AddDataLoader(experiment::LoadCSR<CSR, MTXReader, vertex_type, edge_type, value_type>, {file_name});
  exp.AddPreprocess(experiment::ReorderCSR<RCMReorder, vertex_type, edge_type, value_type>);

  auto kernel_f = [] (unordered_map<string, Format*> data) {
    cout << "DO WHATEVER YOU WANT WITH DATA!!!" << endl;
    context::CPUContext cpu_context;
    auto *perm = ReorderBase::Reorder<RCMReorder>({}, data["ordered"]->As<CSR<vertex_type, edge_type, value_type>>(), {&cpu_context}, true);
    auto * A_reordered = ReorderBase::Permute2D<CSR>(perm, data["ordered"]->As<CSR<vertex_type, edge_type, value_type>>(), {&cpu_context}, true);
    auto * A_csc = A_reordered->Convert<CSC>();
    std::any rt = A_csc;
    return rt;
  };
  exp.AddKernel(kernel_f);
  exp.AddKernel(kernel_f2);
  exp.Run(2);
  std::vector<std::any> res = exp.GetResults();
  cout << "Size = " << res.size() << endl;
  auto secs = exp.GetRunTimes();
  for(auto s: secs){
    cout << "Run time: " << s << endl;
  }
  return 0;
}
