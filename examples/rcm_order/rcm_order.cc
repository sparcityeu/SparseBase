#include <iostream>
#include <set>

#include "sparsebase/format/format.h"
#include "sparsebase/object/object.h"
#include "sparsebase/utils/io/reader.h"
#include "sparsebase/preprocess/preprocess.h"

using namespace std;
using namespace sparsebase;

using vertex_type = unsigned int;
using edge_type = unsigned int;
using value_type = unsigned int;

int main(int argc, char * argv[]){
  if (argc < 2){
    cout << "Usage: ./rcm_order <uedgelist_file>\n";
    cout << "Hint: You can use the edgelist: examples/data/com-dblp.uedgelist\n";
    return 1;
  }
  cout << "F t re  s sp r e!" << endl;
  string file_name = argv[1];
  context::CPUContext cpu_context;

  cout << "********************************" << endl;

  cout << "Reading graph from " << file_name << "..." << endl;
  object::Graph<vertex_type, edge_type, value_type> g;
  g.ReadConnectivityFromEdgelistToCSR(file_name);
  cout << "Number of vertices: " << g.n_ << endl; 
  cout << "Number of edges: " << g.m_ << endl; 

  cout << "********************************" << endl;

  cout << "Generating RCM ordering..." << endl;

  preprocess::RCMReorder<vertex_type, edge_type, value_type> orderer(1,4);
  auto * con = g.get_connectivity()->As<format::CSR<vertex_type,edge_type,value_type>>();
  vertex_type * order = orderer.GetReorder(con, {&cpu_context});
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
