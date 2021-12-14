#include <iostream>
#include <set>

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

  cout << "Generating RCM ordering..." << endl;
  ReorderInstance<vertex_type, edge_type, value_type, RCMReorder<vertex_type, edge_type, value_type>> orderer;
  SparseFormat<vertex_type, edge_type, value_type> * con = g.get_connectivity();
  vertex_type * order = orderer.get_reorder(con);
  auto xadj = con->get_xadj();
  auto adj = con->get_adj();
  vertex_type n = con->get_dimensions()[0];

  cout << "********************************" << endl;

  cout << "Checking the correctness of the ordering..." << endl;
  bool order_is_correct = true;
  set<vertex_type> ids;
  for(vertex_type i = 0; i < n && order_is_correct; i++){
    vertex_type i_order = order[i]; 
    if(i_order < n && ids.find(i_order) == ids.end()){
      ids.insert(i_order);
    }
    else {
      cout << "RCM ordering is incorrect!";
      order_is_correct = false;
    }
  }
  if (ids.size() > n){
      cout << "RCM ordering is incorrect!";
      order_is_correct = false;
  }
  if(order_is_correct){
    cout << "Order is correct!" << endl;
  }
  return 0;
}
