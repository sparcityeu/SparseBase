#include <iostream>

#include "sparsebase/sparse_format.h"
#include "sparsebase/sparse_object.h"
#include "sparsebase/sparse_reader.h"
#include "sparsebase/sparse_preprocess.h"

#include <set>

using namespace std;
using namespace sparsebase;

using vertex_type = unsigned int;
using edge_type = unsigned int;
using value_type = void;

int main(int argc, char * argv[]){
  if (argc < 2){
    cout << "Usage: ./degree_order <uedgelist_file>\n";
    cout << "Hint: You can use the edgelist: examples/data/com-dblp.uedgelist\n";
    return 1;
  }
  cout << "F t re  s sp r e!" << endl;
  string file_name = argv[1];

  cout << "********************************" << endl;

  cout << "Reading graph from " << file_name << "..." << endl;
  Graph<vertex_type, edge_type, value_type> g;
  g.ReadConnectivityFromEdgelistToCSR(file_name);
  cout << "Number of vertices: " << g.n_ << endl; 
  cout << "Number of edges: " << g.m_ << endl; 

  cout << "********************************" << endl;

  cout << "Sorting the vertices according to degree (degree ordering)..." << endl;
  DegreeReorderInstance<vertex_type, edge_type, value_type> orderer(1);
  //ReorderInstance<vertex_type, edge_type, value_type, DegreeReorder<vertex_type, edge_type, value_type>> orderer(1);
  //ExecutableOrdering<vertex_type, edge_type, DegreeOrder<vertex_type, edge_type, value_type>> orderer(1);
  SparseFormat<vertex_type, edge_type, value_type> * con = g.get_connectivity();
  vertex_type * order = orderer.GetReorder(con);
  vertex_type n = con->get_dimensions()[0];
  auto row_ptr = con->get_row_ptr();
  auto col = con->get_col();
  cout << "According to degree order: " << endl;
  cout << "First vertex, ID: " << order[0] << ", Degree: " << row_ptr[order[0] + 1] - row_ptr[order[0]] << endl;
  cout << "Last vertex, ID: " << order[n-1] << ", Degree: " << row_ptr[order[n - 1] + 1] - row_ptr[order[n - 1]] << endl;

  cout << "********************************" << endl;

  cout << "Checking the correctness of the ordering..." << endl;
  bool order_is_correct = true;
  set<vertex_type> check;
  for(vertex_type i = 0; i < n-1 && order_is_correct; i++){
    vertex_type v = order[i]; 
    if(check.find(v) == check.end()){
      check.insert(v);
    }
    else{
      order_is_correct = false;
    }
    vertex_type u = order[i+1];
    if(row_ptr[v + 1] - row_ptr[v] > row_ptr[u + 1] - row_ptr[u])
    {
      cout << "Degree Order is incorrect!" << endl;
      order_is_correct = false;
    }
  }
  vertex_type v = order[n-1]; 
  if(check.find(v) == check.end()){
    check.insert(v);
  }
  else{
      order_is_correct = false;
  }
  if(order_is_correct){
    cout << "Order is correct." << endl;
  }

  TransformInstance<vertex_type, edge_type, value_type, Transform> transformer;
  SparseFormat<vertex_type, edge_type, value_type> * csr = transformer.GetTransformation(con, order);
  auto * n_row_ptr = csr->get_row_ptr();
  auto * n_col = csr->get_col();
  cout << "Checking the correctness of the transformation..." << endl;
  bool transform_is_correct = true;
  for(vertex_type i = 0; i < n-1 && transform_is_correct; i++){
    if(n_row_ptr[i + 2] - n_row_ptr[i + 1] < n_row_ptr[i + 1] - n_row_ptr[i])
    {
      cout << "Transformation is incorrect!" << endl;
      transform_is_correct = false;
    }
  }
  if(transform_is_correct){
    cout << "Transformation is correct." << endl;
  }
  return 0;
}
