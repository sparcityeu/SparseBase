#include <iostream>

#include "sparsebase/SparseFormat.hpp"
#include "sparsebase/SparseObject.hpp"
#include "sparsebase/SparseReader.hpp"
#include "sparsebase/SparsePreprocess.hpp"

using namespace std;
using namespace sparsebase;

int main(int argc, char * argv[]){
  cout << "F t re  s sp r e!" << endl;
  unsigned int xadj[4] = {0, 2, 3, 4};
  unsigned int adj[4] = {1,2,0,0};
  CSR<unsigned int, unsigned int, void> csr(3, 3, xadj, adj, nullptr);
  cout << "Format: " << csr.get_format() << endl;


  //unsigned int
  //COO<unsigned int, unsigned int, float> coo(
  return 0;
}
