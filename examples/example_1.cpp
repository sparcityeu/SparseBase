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
  Graph<unsigned int, unsigned int> g2(new UedgelistReader<unsigned int, unsigned int, void>("/Users/taa/research/graphs/com-dblp_c.graph"));
  DegreeOrder<unsigned int, unsigned int> orderer;
  CSR<unsigned int, unsigned int, void> * tmp = reinterpret_cast<CSR<unsigned int, unsigned int, void>*>(g2.get_connectivity());
  //CSR<unsigned int, unsigned int, void> * tmp = g2.get_connectivity();
  unsigned int * order = orderer.get_order<void>(tmp);
  unsigned int * order2 = orderer.get_order<void>(g2.get_connectivity());
  auto o = tmp->get_order();
  cout << "Order: " << o << endl;
  int n = tmp->get_dimensions()[0];
  cout << "Number of vertices: " << n << endl;
  //cout << "edges "<< tmp->xadj[order2[n-1]+1] - tmp->xadj[order2[n-1]] << endl;
  cout << "edges "<< tmp->xadj[order[n-1]+1] - tmp->xadj[order[n-1]] << endl;
  //cout << "edges "<< order[n-2];
  return 0;
}
