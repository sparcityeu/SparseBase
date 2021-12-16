#include <iostream>

#include "sparsebase/SparseFormat.hpp"
#include "sparsebase/SparseReader.hpp"
#include "sparsebase/SparseObject.hpp"

using vertex_type = unsigned int;
using edge_type = unsigned int;
using value_type = void;

using namespace std;

int main(int argc, char * argv[]){
    string file_name = argv[1];

    // Reading the mtx into a graph object
    auto reader = new sparsebase::MTXReader<vertex_type, edge_type, value_type>(file_name);
    sparsebase::Graph<vertex_type, edge_type, value_type> g(reader);
    cout << "Number of vertices: " << g.n << endl;
    cout << "Number of edges: " << g.m << endl;

    // Extracting connectivity information from a graph and casting it
    auto coo = dynamic_cast<sparsebase::COO<vertex_type,edge_type,value_type>*>(g.get_connectivity());

    vertex_type nnz = coo->get_num_nnz();
    vertex_type * adj = coo->get_adj();
    vertex_type * is = coo->get_is();

    cout << "NNZ: " << nnz << endl;

}