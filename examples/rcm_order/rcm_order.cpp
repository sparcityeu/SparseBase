#include <iostream>
#include <set>

#include "sparsebase/sparse_format.hpp"
#include "sparsebase/sparse_object.hpp"
#include "sparsebase/sparse_reader.hpp"
#include "sparsebase/sparse_preprocess.hpp"

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
  Graph<vertex_type, edge_type, value_type> g;
  g.read_connectivity_from_edgelist_to_csr(file_name);
  cout << "Number of vertices: " << g.n_ << endl; 
  cout << "Number of edges: " << g.m_ << endl; 

  cout << "********************************" << endl;

  cout << "Generating RCM ordering..." << endl;
  //ReorderInstance<vertex_type, edge_type, value_type, RCMReorder<vertex_type, edge_type, value_type>> orderer(1,4);
  ReorderInstance<vertex_type, edge_type, value_type, RCMReorder> orderer(1,4);
  SparseFormat<vertex_type, edge_type, value_type> * con = g.get_connectivity();
  vertex_type * order = orderer.get_reorder(con);
  auto xadj = con->get_row_ptr();
  auto adj = con->get_col();
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
