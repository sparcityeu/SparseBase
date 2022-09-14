#include <iostream>

#include "sparsebase/format/format.h"
#include "sparsebase/preprocess/preprocess.h"
#include "sparsebase/utils/io/reader.h"
#include "sparsebase/experiment/experiment.h"

using namespace std;
using namespace sparsebase;

using vertex_type = unsigned int;
using edge_type = unsigned int;
using value_type = unsigned int;

int main(int argc, char **argv){
    
  unsigned int row_ptr[4] = {0, 2, 3, 4};
  unsigned int col[4] = {1, 2, 0, 0};

  format::CSR<vertex_type, edge_type, value_type> csr(3, 3, row_ptr,
                                                              col, nullptr);
  auto format = csr.get_format_id();
  auto dimensions = csr.get_dimensions();
  auto row_ptr2 = csr.get_row_ptr();
  auto col2 = csr.get_col();
  auto vals = csr.get_vals();
  cout << "Format: " << format.name() << endl;
  cout << "# of dimensions: " << dimensions.size() << endl;
  for (int i = 0; i < dimensions.size(); i++) {
      cout << "Dim " << i << " size " << dimensions[i] << endl;
  }
  
  experiment::ConcreteExperiment exp;
  exp.AddFormat(&csr);

  preprocess::RCMReorder<vertex_type, edge_type, value_type> o1;
  exp.AddProcess(&o1);

}
