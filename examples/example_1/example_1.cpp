#include <iostream>

#include "sparsebase/SparseFormat.hpp"
#include "sparsebase/SparseObject.hpp"
#include "sparsebase/SparseReader.hpp"
#include "sparsebase/SparseProcess.hpp"

using namespace std;
using namespace sparsebase;

int main(int argc, char * argv[]){
  cout << "F t re  s sp r e!" << endl;
  unsigned int xadj[4] = {0, 2, 3, 4};
  unsigned int adj[4] = {1,2,0,0};
  CSR<unsigned int, unsigned int, void> csr(3, 3, xadj, adj, nullptr);
  Graph<unsigned int, unsigned int> g(&csr);
  string file_name = argv[1];
  Graph<unsigned int, unsigned int> g2(new UedgelistReader<unsigned int, unsigned int, void>(file_name));
  //DegreeOrder<unsigned int, unsigned int> orderer;
  //ExecutableDegreeOrdering<unsigned int, unsigned int> orderer(1);
  ExecutableOrdering<unsigned int, unsigned int, DegreeOrder<unsigned int, unsigned int, void>> orderer(1);
  SparseFormat<unsigned int, unsigned int> * con = g2.get_connectivity();
  CSR<unsigned int, unsigned int, void> * tmp = reinterpret_cast<CSR<unsigned int, unsigned int, void>*>(con);
  //CSR<unsigned int, unsigned int, void> * tmp = reinterpret_cast<CSR<unsigned int, unsigned int, void>*>(g.get_connectivity());
  unsigned int* order = orderer.get_order(con);
  cout << "Order: " << order << endl;
  cout << "Order: " << order[csr.get_dimensions()[0]-1] << endl;
  cout << "Order: " << order[0] << endl;
  //CSR<unsigned int, unsigned int, void> * tmp = g2.get_connectivity();
  //unsigned int * order = orderer.get_order<void>(tmp);
  //unsigned int * order2 = orderer.get_order<void>(g2.get_connectivity());
  //auto o = tmp->get_order();
  //cout << "Order: " << o << endl;
  int n = tmp->get_dimensions()[0];
  cout << "Number of vertices: " << n << endl;
  cout << "edges "<< tmp->xadj[order[n-1]+1] - tmp->xadj[order[n-1]] << endl;
  cout << "edges "<< order[n-2] << endl;
  return 0;
}
