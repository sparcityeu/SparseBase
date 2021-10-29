#include <iostream>

#include "Tensor.hpp"
#include "SparseObject.hpp"

using namespace std;
using namespace sparsebase;

int main(int argc, char * argv[]){
  cout << "F t re  s sp r e!" << endl;
  Tensor * csf = new CSF(); 
  Tensor * coo = new COO(); 
  COO * coo2 = new COO(); 
  coo->get_rank(); 
  coo2->go_crazy(); 
  SparseObject * graph = new Graph<int, int>();
  
  return 0;
}
