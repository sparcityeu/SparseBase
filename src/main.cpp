#include <iostream>

#include "Tensor.hpp"
#include "SparseObject.hpp"

using namespace std;
using namespace sparsebase;

int main(int argc, char * argv[]){
  cout << "F t re  s sp r e!" << endl;
  Tensor * csf = new CSF(3); 
  Tensor * csr = new CSR<unsigned int, unsigned int>(); 
  Tensor * coo = new COO(3); 
  COO * coo2 = new COO(3); 
  coo->get_rank(); 
  SparseObject * graph = new Graph<int, int>();
  string fname = "/data/GE/graphs/uedgelist/com-dblp_c.graph";
  Graph<int, int> * graph2 = new Graph<int, int>(fname);
  
  return 0;
}
