#include <iostream>

#include "sparsebase/sparse_format.h"
#include "sparsebase/sparse_object.h"
#include "sparsebase/sparse_reader.h"
#include "sparsebase/sparse_preprocess.h"
#include "sparsebase/sparse_exception.h"

using namespace std;
using namespace sparsebase;

int main(int argc, char * argv[]){
  if (argc < 2){
    cout << "Usage: ./csr_coo <matrix_market_format>\n";
    cout << "Hint: You can use the matrix market file: examples/data/ash958.mtx\n";
    return 1;
  }
  cout << "F t re  s sp r e!" << endl;
  unsigned int row_ptr[4] = {0, 2, 3, 4};
  unsigned int col[4] = {1, 2, 0, 0};
  {
    format::CSR<unsigned int, unsigned int, unsigned int> csr(3, 3, row_ptr, col, nullptr);
    auto format = csr.get_format_id();
    auto dimensions = csr.get_dimensions();
    auto row_ptr2 = csr.get_row_ptr();
    auto col2 = csr.get_col();
    auto vals = csr.get_vals();
    cout << "Format: " << format.name() << endl;
    cout << "# of dimensions: " << dimensions.size() << endl;
    for(int i = 0; i < dimensions.size(); i++){
      cout << "Dim " << i << " size " << dimensions[i] << endl; 
    }
  }

  cout << endl;
  cout <<  "************************" << endl;
  cout <<  "************************" << endl;
  cout << endl;

  {
    string file_name = argv[1];
    utils::MTXReader<unsigned int, unsigned int, unsigned int> reader(file_name);
    format::COO<unsigned int, unsigned int, unsigned int> * coo = reader.ReadCOO();
    auto format = coo->get_format_id();
    auto dimensions = coo->get_dimensions();
    auto coo_col = coo->get_col();
    auto coo_row = coo->get_row();
    auto coo_vals = coo->get_vals();
    cout << "Format: " << coo->get_format_id().name() << endl;
    cout << "# of dimensions: " << dimensions.size() << endl;
    for(int i = 0; i < dimensions.size(); i++){
      cout << "Dim " << i << " size " << dimensions[i] << endl; 
    }
  }

  return 0;
}
