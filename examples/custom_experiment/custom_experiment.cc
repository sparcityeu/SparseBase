#include <iostream>

#include "sparsebase/bases/reorder_base.h"
#include "sparsebase/experiment/concrete_experiment.h"
#include "sparsebase/format/csr.h"
#include "sparsebase/format/format.h"
#include "sparsebase/format/format_order_one.h"
#include "sparsebase/format/format_order_two.h"
#include "sparsebase/io/mtx_reader.h"
#include "sparsebase/reorder/reorderer.h"

using namespace std;
using namespace sparsebase;
using namespace format;
;
using namespace utils;
using namespace io;

using vertex_type = unsigned int;
using edge_type = unsigned int;
using value_type = unsigned int;

// create custom preprocess function
void preprocess_f(unordered_map<string, Format *> &data, std::any &fparams,
                  std::any &params) {
  context::CPUContext cpu_context;
  auto perm = bases::ReorderBase::Reorder<reorder::RCMReorder>(
      {}, data["graph"]->AsAbsolute<CSR<vertex_type, edge_type, value_type>>(),
      {&cpu_context}, true);
  auto A_reordered = bases::ReorderBase::Permute2D<CSR>(
      perm,
      data["graph"]->AsAbsolute<CSR<vertex_type, edge_type, value_type>>(),
      {&cpu_context}, true);
  data.emplace("ordered", A_reordered);
};

// kernel function is always provided by the user
// this kernel counts the number of nnz in each row of the matrix or number of
// rows of a graph
std::any kernel_f(unordered_map<string, Format *> &data, std::any &fparams,
                  std::any &pparams, std::any &kparams) {
  auto spm =
      data["ordered"]->AsAbsolute<CSR<vertex_type, edge_type, value_type>>();
  auto dimensions = spm->get_dimensions();
  auto num_rows = dimensions[0];
  auto rows = spm->get_row_ptr();
  auto cols = spm->get_col();

  auto *res = new double[num_rows];
  for (unsigned int i = 0; i < num_rows; i++) {
    double s = 0;
    for (unsigned int j = rows[i]; j < rows[i + 1]; j++) {
      s++;
    }
    res[i] = s;
  }

  return res;
};

int main(int argc, char **argv) {
  experiment::ConcreteExperiment exp;
  vector<string> file_names = {argv[1]};
  string file_name = argv[1];

  // experiment functions are wrapped with std::function, thus can be any
  // callable function that abides by the function definition. even though
  // passing file names as a target makes perfect sense, this example lambda
  // shows anything can be passed as a context to experiment all functions
  auto get_data_l = [file_name](vector<string> &file_names) {
    CSR<vertex_type, edge_type, value_type> *csr =
        MTXReader<vertex_type, edge_type, value_type>(file_name).ReadCSR();
    unordered_map<string, Format *> r;
    r.emplace("graph", csr);
    return r;
  };
  std::any empty;
  exp.AddDataLoader(get_data_l, {make_pair(file_names, empty)});
  // add custom preprocessing
  exp.AddPreprocess("mypreprocess", preprocess_f, {});

  exp.AddKernel("mykernel", kernel_f, {});

  exp.Run(2);

  auto res = exp.GetResults();
  cout << "# of results = " << res.size() << endl;
  auto secs = exp.GetRunTimes();
  cout << "Runtimes: " << endl;
  for (const auto &s : secs) {
    cout << s.first << ": ";
    for (auto sr : s.second) {
      cout << sr << " ";
    }
    cout << endl;
  }
  return 0;
}
