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

struct customParam : ReorderParams{
  customParam(int h):_hyperparameter(h){}
  int _hyperparameter;
};
vertex_type* degree_reorder_csr(std::vector<SparseFormat<vertex_type, edge_type, value_type> *> formats, ReorderParams *params)
{
  CSR<vertex_type, edge_type, value_type> *csr = static_cast<CSR<vertex_type, edge_type, value_type> *>(formats[0]);
  customParam  *cast_params = static_cast<customParam *>(params);
  cout << cast_params->_hyperparameter << endl;
  vertex_type n = csr->get_dimensions()[0];
  vertex_type *counts = new vertex_type[n]();
  for (vertex_type u = 0; u < n; u++)
  {
    counts[csr->xadj[u + 1] - csr->xadj[u] + 1]++;
  }
  for (vertex_type u = 1; u < n; u++)
  {
    counts[u] += counts[u - 1];
  }
  vertex_type *sorted = new vertex_type[n];
  memset(sorted, -1, sizeof(vertex_type) * n);
  vertex_type *mr = new vertex_type[n]();
  for (vertex_type u = 0; u < n; u++)
  {
    vertex_type ec = counts[csr->xadj[u + 1] - csr->xadj[u]];
    sorted[ec + mr[ec]] = u;
    mr[ec]++;
  }
  vertex_type *inv_sorted = new vertex_type[n];
  for (vertex_type i = 0; i < n; i++)
    inv_sorted[sorted[i]] = i;
  delete[] mr;
  delete[] counts;
  delete[] sorted;
  return inv_sorted;
}
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
  SparseFormat<vertex_type, edge_type, value_type> * con = g.get_connectivity();

  ReorderInstance<vertex_type, edge_type, value_type, GenericReorder> orderer;
  orderer.register_function({CSR_f}, degree_reorder_csr);
  customParam params{10};
  vertex_type * order = orderer.get_reorder(con, &params);

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

  TransformInstance<vertex_type, edge_type, value_type, Transform> transformer(1);
  SparseFormat<vertex_type, edge_type, value_type> * csr = transformer.get_transformation(con, order);
  auto * nxadj = csr->get_xadj();
  auto * nadj = csr->get_adj();
  cout << "Checking the correctness of the transformation..." << endl;
  bool transform_is_correct = true;
  for(vertex_type i = 0; i < n-1 && transform_is_correct; i++){
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