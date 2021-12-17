#include <iostream>

#include "sparsebase/SparseFormat.hpp"
#include "sparsebase/SparseObject.hpp"
#include "sparsebase/SparseReader.hpp"
#include "sparsebase/SparseProcess.hpp"
#include "sparsebase/SparseException.hpp"

using namespace std;
using namespace sparsebase;

int main(int argc, char * argv[]){
  cout << "F t re  s sp r e!" << endl;
  unsigned int xadj[4] = {0, 2, 3, 4};
  unsigned int adj[4] = {1,2,0,0};
  {
    CSR<unsigned int, unsigned int, void> csr(3, 3, xadj, adj, nullptr);
    auto format = csr.get_format();
    auto dimensions = csr.get_dimensions();
    auto xadj = csr.get_xadj();
    auto adj = csr.get_adj();
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
    COO<unsigned int, unsigned int, void> * coo = reader.read_coo();
    auto format = coo->get_format();
    auto dimensions = coo->get_dimensions();
    auto adj = coo->get_adj();
    try{ //if the data member is invalid, sparse format throws an exception
      auto xadj = coo->get_xadj();
    }
    catch(InvalidDataMember ex){
      cout << ex.what() << endl;
    }
    auto is = coo->get_is();
    auto vals = coo->get_vals();
    cout << "Format: " << coo->get_format() << endl;
    cout << "# of dimensions: " << dimensions.size() << endl;
    for(int i = 0; i < dimensions.size(); i++){
      cout << "Dim " << i << " size " << dimensions[i] << endl; 
    }
  }

  return 0;
}
