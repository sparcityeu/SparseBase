#include <iostream>

#include "sparsebase/SparseFormat.hpp"
#include "sparsebase/SparseObject.hpp"
#include "sparsebase/SparseReader.hpp"
#include "sparsebase/SparseProcess.hpp"

using namespace std;
using namespace sparsebase;

using vertex_type = unsigned int;
using edge_type = unsigned int;
using value_type = void;

int main(int argc, char * argv[]){
  cout << "F t re  s sp r e!" << endl;
  string file_name = argv[1];

  cout << "********************************" << endl;

  cout << "Reading graph from " << file_name << "..." << endl;
  Graph<vertex_type, edge_type, value_type> g(new UedgelistReader<vertex_type, edge_type, value_type>(file_name));
  cout << "Number of vertices: " << g.n << endl; 
  cout << "Number of edges: " << g.m << endl; 

  cout << "********************************" << endl;

  cout << "Sorting the vertices according to degree (degree ordering)..." << endl;
  DegreeReorderInstance<vertex_type, edge_type, value_type> orderer(1);
  //ExecutableOrdering<vertex_type, edge_type, DegreeOrder<vertex_type, edge_type, value_type>> orderer(1);
  SparseFormat<vertex_type, edge_type, value_type> * con = g.get_connectivity();
  vertex_type * order = orderer.get_reorder(con);
  auto xadj = con->get_xadj();
  auto adj = con->get_adj();
  vertex_type n = con->get_dimensions()[0];
  cout << "According to degree order: " << endl;
  cout << "First vertex, ID: " << order[0] << ", Degree: " << xadj[order[0]] - xadj[order[0]] << endl;
  cout << "Last vertex, ID: " << order[n-1] << ", Degree: " << xadj[order[n-1]+1] - xadj[order[n-1]] << endl;

  cout << "********************************" << endl;

  cout << "Checking the correctness of the ordering..." << endl;
  bool order_is_correct = true;
  for(vertex_type i; i < n-1 && order_is_correct; i++){
    vertex_type v = order[i]; 
    vertex_type u = order[i+1];
    if(xadj[v+1] - xadj[v] > xadj[u+1] - xadj[u])
    {
      cout << "Degree Order is incorrect!" << endl;
      order_is_correct = false;
    }
  }
  if(order_is_correct){
    cout << "Order is correct!" << endl;
  }
  return 0;
}
