#include <iostream>

#include "sparsebase/sparse_format.h"
#include "sparsebase/sparse_object.h"
#include "sparsebase/sparse_reader.h"
#include "sparsebase/sparse_preprocess.h"
#include "sparsebase/sparse_exception.h"

using namespace std;
using namespace sparsebase;

int main(int argc, char * argv[]){
  cout << "F t re  s sp r e!" << endl;
  unsigned int row_ptr[4] = {0, 2, 3, 4};
  unsigned int col[4] = {1, 2, 0, 0};
  {
    CSR<unsigned int, unsigned int, void> csr(3, 3, row_ptr, col, nullptr);
    auto format = csr.get_format();
    auto dimensions = csr.get_dimensions();
    auto row_ptr2 = csr.get_row_ptr();
    auto col2 = csr.get_col();
    auto vals = csr.get_vals();
    cout << "Format: " << format << endl;
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
    MTXReader<unsigned int, unsigned int, void> reader(file_name);
    COO<unsigned int, unsigned int, void> * coo = reader.ReadCOO();
    auto format = coo->get_format();
    auto dimensions = coo->get_dimensions();
    auto coo_col = coo->get_col();
    try{ //if the data member is invalid, sparse format throws an exception
      auto coo_row_ptr = coo->get_row_ptr();
    }
    catch(InvalidDataMember& ex){
      cout << ex.what() << endl;
    }
    auto coo_row = coo->get_row();
    auto coo_vals = coo->get_vals();
    cout << "Format: " << coo->get_format() << endl;
    cout << "# of dimensions: " << dimensions.size() << endl;
    for(int i = 0; i < dimensions.size(); i++){
      cout << "Dim " << i << " size " << dimensions[i] << endl; 
    }
  }

  return 0;
}
