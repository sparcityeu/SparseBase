#include <iostream>

#include "Tensor.hpp"
#include "SparseObject.hpp"

using namespace std;
using namespace sparsebase;

int main(int argc, char * argv[]){
  cout << "F t re  s sp r e!" << endl;
  Tensor * csf = new CSF(); 
  Tensor * coo = new COO(); 
  SparseObject * graph = new Graph();
  
  return 0;
}
