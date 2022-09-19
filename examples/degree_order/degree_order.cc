#include <iostream>

#include "sparsebase/format/format.h"
#include "sparsebase/object/object.h"
#include "sparsebase/preprocess/preprocess.h"
#include "sparsebase/utils/io/reader.h"

#include <set>

using namespace std;
using namespace sparsebase;

using vertex_type = unsigned int;
using edge_type = unsigned int;
using value_type = unsigned int;

int main(int argc, char *argv[]) {
  if (argc < 2) {
    cout << "Usage: ./degree_order <uedgelist_file>\n";
    cout
        << "Hint: You can use the edgelist: examples/data/com-dblp.uedgelist\n";
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

  cout << "Sorting the vertices according to degree (degree ordering)..."
       << endl;

  preprocess::DegreeReorder<vertex_type, edge_type, value_type> orderer(1);
  format::Format *con = g.get_connectivity();
  vertex_type *permutation = orderer.GetReorder(con, {&cpu_context}, false);
  vertex_type n = con->get_dimensions()[0];
  auto row_ptr =
      con->As<format::CSR<vertex_type, edge_type, value_type>>()->get_row_ptr();
  auto col =
      con->As<format::CSR<vertex_type, edge_type, value_type>>()->get_col();
  cout << "According to degree order: " << endl;
  cout << "First vertex, ID: " << permutation[0]
       << ", Degree: " << row_ptr[permutation[0] + 1] - row_ptr[permutation[0]] << endl;
  cout << "Last vertex, ID: " << permutation[n - 1]
       << ", Degree: " << row_ptr[permutation[n - 1] + 1] - row_ptr[permutation[n - 1]]
       << endl;

  cout << "********************************" << endl;

  cout << "Checking the correctness of the ordering..." << endl;
  // We can easily get the inverse of our permutation (to reverse the ordering)
  auto inv_permutation = preprocess::ReorderBase::InversePermutation(permutation, con->get_dimensions()[0]);
  bool order_is_correct = true;
  set<vertex_type> check;
  for (vertex_type new_u = 0; new_u < n - 1 && order_is_correct; new_u++) {
    vertex_type u = inv_permutation[new_u];
    if (check.find(u) == check.end()) {
      check.insert(u);
    } else {
      order_is_correct = false;
    }
    vertex_type v = inv_permutation[new_u + 1];
    if (row_ptr[u + 1] - row_ptr[u] > row_ptr[v + 1] - row_ptr[v]) {
      cout << "Degree Order is incorrect!" << endl;
      order_is_correct = false;
      return 1;
    }
  }
  vertex_type v = inv_permutation[n - 1];
  if (check.find(v) == check.end()) {
    check.insert(v);
  } else {
    order_is_correct = false;
  }
  if (order_is_correct) {
    cout << "Order is correct." << endl;
  }
  preprocess::PermuteOrderTwo<vertex_type, edge_type, value_type> transformer(permutation, permutation);
  format::Format *perm_csr = transformer.GetTransformation(con, {&cpu_context}, true);
  auto *n_row_ptr =
      perm_csr->As<format::CSR<vertex_type, edge_type, value_type>>()->get_row_ptr();
  auto *n_col =
      perm_csr->As<format::CSR<vertex_type, edge_type, value_type>>()->get_col();
  cout << "Checking the correctness of the transformation..." << endl;
  bool transform_is_correct = true;
  for (vertex_type i = 0; i < n - 1 && transform_is_correct; i++) {
    if (n_row_ptr[i + 2] - n_row_ptr[i + 1] < n_row_ptr[i + 1] - n_row_ptr[i]) {
      cout << "Transformation is incorrect!" << endl;
      transform_is_correct = false;
      return 1;
    }
  }
  if (transform_is_correct) {
    cout << "Transformation is correct." << endl;
  }
  // We can reverse the ordering easily
  preprocess::PermuteOrderTwo<vertex_type, edge_type, value_type> inv_trans(inv_permutation, inv_permutation);
  auto perm_then_inv_perm_csr = inv_trans.GetTransformation(perm_csr, {&cpu_context}, true)->As<format::CSR<vertex_type, edge_type, value_type>>();
  auto orig_csr = con->As<format::CSR<vertex_type, edge_type, value_type>>();
  for (vertex_type i = 0; i < n; i++){
    for (edge_type e = perm_then_inv_perm_csr->get_row_ptr()[i]; e < perm_then_inv_perm_csr->get_row_ptr()[i+1]; e++){
      if (orig_csr->get_col()[e]!= perm_then_inv_perm_csr->get_col()[e]){
        cout << "Bad inverse reordering!\n";
        return 1;
      }
    }
  }
  cout << "Inversion is correct\n";
  delete[] permutation;
  delete[] inv_permutation;
  return 0;
}
