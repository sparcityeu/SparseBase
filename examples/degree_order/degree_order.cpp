#include <iostream>

#include "sparsebase/SparseFormat.hpp"
#include "sparsebase/SparseObject.hpp"
#include "sparsebase/SparseReader.hpp"
#include "sparsebase/SparsePreprocess.hpp"

#include <set>

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
  cout << "Number of vertices: " << g.n << endl; 
  cout << "Number of edges: " << g.m << endl; 

  cout << "********************************" << endl;

  cout << "Sorting the vertices according to degree (degree ordering)..." << endl;
  DegreeReorderInstance<vertex_type, edge_type, value_type> orderer(1);
  //ReorderInstance<vertex_type, edge_type, value_type, DegreeReorder<vertex_type, edge_type, value_type>> orderer(1);
  TransformInstance<vertex_type, edge_type, value_type, Transform<vertex_type, edge_type, value_type>> transformer(1);
  //ExecutableOrdering<vertex_type, edge_type, DegreeOrder<vertex_type, edge_type, value_type>> orderer(1);
  SparseFormat<vertex_type, edge_type, value_type> * con = g.get_connectivity();
  vertex_type * order = orderer.get_reorder(con);
  vertex_type n = con->get_dimensions()[0];
  auto xadj = con->get_xadj();
  auto adj = con->get_adj();
  cout << "According to degree order: " << endl;
  cout << "First vertex, ID: " << order[0] << ", Degree: " << xadj[order[0]+1] - xadj[order[0]] << endl;
  cout << "Last vertex, ID: " << order[n-1] << ", Degree: " << xadj[order[n-1]+1] - xadj[order[n-1]] << endl;

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
    if(xadj[v+1] - xadj[v] > xadj[u+1] - xadj[u])
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

  SparseFormat<vertex_type, edge_type, value_type> * csr = transformer.get_transformation(con, order);
  auto * nxadj = csr->get_xadj();
  auto * nadj = csr->get_adj();
  cout << "Checking the correctness of the transformation..." << endl;
  bool transform_is_correct = true;
  for(vertex_type i = 0; i < n-1 && order_is_correct; i++){
    if(nxadj[i+2] - nxadj[i+1] < nxadj[i+1] - nxadj[i])
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
