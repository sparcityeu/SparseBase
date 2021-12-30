#include <iostream>

#include "sparsebase/sparse_format.hpp"
#include "sparsebase/sparse_reader.hpp"
#include "sparsebase/sparse_object.hpp"

using vertex_type = unsigned int;
using edge_type = unsigned int;
using value_type = void;

using namespace std;

int main(int argc, char * argv[]){
    string file_name = argv[1];

    // Reading the mtx into a graph object
    //auto reader = new sparsebase::MTXReader<vertex_type, edge_type, value_type>(file_name);
    sparsebase::Graph<vertex_type, edge_type, value_type> g;
    g.read_connectivity_to_coo(sparsebase::MTXReader<vertex_type, edge_type, value_type>(file_name));

    cout << "Number of vertices: " << g.n_ << endl;
    cout << "Number of edges: " << g.m_ << endl;

    // Extracting connectivity information from a graph and casting it
    auto coo = dynamic_cast<sparsebase::COO<vertex_type,edge_type,value_type>*>(g.get_connectivity());

    vertex_type nnz = coo->get_num_nnz();
    vertex_type * col = coo->get_col();
    vertex_type * row = coo->get_row();

    cout << "NNZ: " << nnz << endl;

}