#include <iostream>
#include <set>

#include "sparsebase/bases/iobase.h"
#include "sparsebase/bases/reorder_base.h"
#include "sparsebase/format/csr.h"
#include "sparsebase/format/format_order_one.h"
#include "sparsebase/format/format_order_two.h"
#include "sparsebase/object/object.h"
#include "sparsebase/reorder/gray_reorder.h"

using namespace std;
using namespace sparsebase;
using namespace io;
using namespace bases;
using namespace format;
using namespace reorder;

using vertex_type = int;
using edge_type = int;
using value_type = float;

int main(int argc, char *argv[]) {
  if (argc < 2) {
    cout << "Usage: ./gray_order <matrix_market_file>\n";
    cout
        << "Hint: You can use the edgelist: examples/data/com-dblp.mtx\n";
    return 1;
  }
  cout << "F t re  s sp r e!" << endl;
  string file_name = argv[1];

  cout << "********************************" << endl;

  cout << "Reading graph from " << file_name << "..." << endl;
  auto csr =
      sparsebase::bases::IOBase::ReadMTXToCSR<vertex_type, edge_type,
          value_type>(file_name, true);
  cout << "Number of vertices: " << csr->get_dimensions()[0] << endl;
  cout << "Number of edges: " << csr->get_num_nnz() << endl;

  vertex_type num_rows = csr->get_dimensions()[0];
  edge_type num_non_zeros = csr->get_num_nnz();

  cout << "********************************" << endl;
  cout << "Generating Gray ordering..." << endl;

  // A context representing the host system
  context::CPUContext cpu_context;

  // Create a parameters object to store special parameters specific
  GrayReorderParams gray_params(
      BitMapSize::BitSize16, 20,
      ((num_non_zeros / num_rows) / BitMapSize::BitSize16));

  vertex_type *gray_reorder = bases::ReorderBase::Reorder<GrayReorder>(
      gray_params, csr, {&cpu_context}, true);
  cout << "********************************" << endl;

  cout << "Checking the correctness of the ordering..." << endl;
  bool order_is_correct = true;
  set<vertex_type> ids;
  for (vertex_type i = 0; i < num_rows && order_is_correct; i++) {
    vertex_type i_order = gray_reorder[i];
    if (i_order < num_rows && ids.find(i_order) == ids.end()) {
      ids.insert(i_order);
    } else {
      cout << "Gray ordering is incorrect!";
      order_is_correct = false;
      return 1;
    }
  }
  if (ids.size() > num_rows) {
    cout << "Gray ordering is incorrect!";
    order_is_correct = false;
  }
  if (order_is_correct) {
    cout << "Order is correct!" << endl;
  } else {
    cout << "Gray ordering is incorrect!";
    order_is_correct = false;
    return 1;
  }

  delete gray_reorder;

  return 0;
}