#include <iostream>

#include "sparsebase/sparse_format.h"
#include "sparsebase/sparse_reader.h"
#include "sparsebase/sparse_object.h"

using vertex_type = unsigned int;
using edge_type = unsigned int;
using value_type = void;

using namespace std;

int main(int argc, char * argv[]){
  if (argc < 2){
    cout << "Usage: ./sparse_reader <matrix_market_format>\n";
    cout << "Hint: You can use the matrix market file: examples/data/ash958.mtx\n";
    return 1;
  }
    string file_name = argv[1];

    // Reading the mtx into a graph object
    //auto reader = new sparsebase::MTXReader<vertex_type, edge_type, value_type>(file_name);
    sparsebase::Graph<vertex_type, edge_type, value_type> g;
    g.ReadConnectivityToCOO(sparsebase::MTXReader<vertex_type, edge_type, value_type>(file_name));

    cout << "Number of vertices: " << g.n_ << endl;
    cout << "Number of edges: " << g.m_ << endl;

    // Extracting connectivity information from a graph and casting it
    auto coo = dynamic_cast<sparsebase::COO<vertex_type,edge_type,value_type>*>(g.get_connectivity());

    vertex_type nnz = coo->get_num_nnz();
    vertex_type * col = coo->get_col();
    vertex_type * row = coo->get_row();

    cout << "NNZ: " << nnz << endl;

}