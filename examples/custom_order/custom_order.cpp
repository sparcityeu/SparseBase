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

struct customParam : ReorderParams{
  customParam(int h):hyperparameter(h){}
  int hyperparameter;
};
vertex_type* degree_reorder_csr(std::vector<SparseFormat<vertex_type, edge_type, value_type> *> formats, ReorderParams *params)
{
  CSR<vertex_type, edge_type, value_type> *csr = static_cast<CSR<vertex_type, edge_type, value_type> *>(formats[0]);
  customParam  *cast_params = static_cast<customParam *>(params);
  cout << "Custom hyperparameter: " << cast_params->hyperparameter << endl;
  vertex_type n = csr->get_dimensions()[0];
  vertex_type *counts = new vertex_type[n]();
  auto row_ptr = csr->get_row_ptr();
  for (vertex_type u = 0; u < n; u++)
  {
    counts[row_ptr[u + 1] - row_ptr[u] + 1]++;
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
    vertex_type ec = counts[row_ptr[u + 1] - row_ptr[u]];
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
  if (argc < 2){
    cout << "Usage: ./custom_order <uedgelist_file>\n";
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
  SparseFormat<vertex_type, edge_type, value_type> * con = g.get_connectivity();

  ReorderInstance<vertex_type, edge_type, value_type, GenericReorder> orderer;
  orderer.RegisterFunction({kCSRFormat}, degree_reorder_csr);
  customParam params{10};
  vertex_type * order = orderer.GetReorder(con, &params);

  vertex_type n = con->get_dimensions()[0];
  auto xadj = con->get_row_ptr();
  auto adj = con->get_col();
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
