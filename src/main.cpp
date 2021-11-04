#include <iostream>

#include "SparseFormat.hpp"
#include "SparseObject.hpp"
#include "SparseReader.hpp"
#include "SparseProcess.hpp"

using namespace std;
using namespace sparsebase;

int main(int argc, char * argv[]){
  cout << "F t re  s sp r e!" << endl;
  unsigned int xadj[4] = {0, 2, 3, 4};
  unsigned int adj[4] = {1,2,0,0};
  CSR<unsigned int, unsigned int, void> csr(3, 3, xadj, adj, nullptr);
  Graph<unsigned int, unsigned int> g(&csr);
  Graph<unsigned int, unsigned int> g2(UedgelistReader<unsigned int, unsigned int, void>("graphs/com-dblp_c.graph"));
  DegreeOrder<unsigned int, unsigned int> orderer;
  unsigned int * order = orderer.get_order<void>(&csr);
  int n = csr.get_dimensions()[0];
  //cout << "edges "<< csr.xadj[order[n-1]+1] - csr.xadj[order[n-1]];
  cout << "edges "<< order[n-2];
  return 0;
}
