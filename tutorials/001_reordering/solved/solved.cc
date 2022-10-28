#include "sparsebase/format/format.h"
#include "sparsebase/utils/io/iobase.h"
#include "sparsebase/preprocess/preprocess.h"
#include "sparsebase/context/context.h"
#include <string>
#include <iostream>

typedef unsigned int id_type;
typedef unsigned int nnz_type;
typedef void value_type;

using namespace sparsebase;
using namespace utils::io;
using namespace preprocess;
using namespace format;

int main(int argc, char * argv[]){
    if (argc < 2){
        std::cout << "Please enter the name of the edgelist file as a parameter\n";
        return 1;
    }
    // The name of the edge list file in disk
    std::string filename(argv[1]);
    // Read the edge list file into a CSR object
    CSR<id_type, nnz_type, value_type>* csr = IOBase::ReadEdgeListToCSR<id_type, nnz_type, value_type>(filename);
    std::cout << "Original graph:" << std::endl;
    // get a array representing the dimensions of the matrix represented by `csr`,
    // i.e, the adjacency matrix of the graph
    std::cout << "Number of vertices: " << csr->get_dimensions()[0] << std::endl;
    // Number of non-zeros in the matrix represented by `csr`
    std::cout << "Number of edges: " << csr->get_num_nnz() << std::endl;

    // row_ptr contains the starting indices of the adjacency lists of the vertices in `csr`
    auto row_ptr = csr->get_row_ptr();
    std::cout << "Degree of vertex 0: " << row_ptr[1]-row_ptr[0] << std::endl;
    std::cout << "Degree of vertex 1: " << row_ptr[2]-row_ptr[1] << std::endl;
    std::cout << "Degree of vertex 2: " << row_ptr[3]-row_ptr[2] << std::endl;
    std::cout << std::endl;

    // Create a CPU context
    context::CPUContext cpu_context;
    // We would like to order the vertices by degrees in descending order
    bool ascending = false;
    DegreeReorderParams params(ascending);
    // Create a permutation array of `csr` using one of the passed contexts
    // (in this case, only one is passed)
    // The last argument tells the function to convert the input format if needed
    id_type* new_order = ReorderBase::Reorder<DegreeReorder>(params, csr, {&cpu_context}, true);

    // Permute2D permutes the rows and columns of `csr` according to `new_order`
    // Similar to `Reorder`, we specify the contexts to use,
    // and whether the library can convert the input if needed
    FormatOrderTwo<id_type, nnz_type, value_type>* new_format =
        ReorderBase::Permute2D(new_order, csr, {&cpu_context}, true);
    // Cast the polymorphic pointer to a pointer at CSR
    CSR<id_type, nnz_type, value_type>* new_csr = new_format->As<CSR>();
    std::cout << "Reordered graph:" << std::endl;
    std::cout << "Number of vertices: " << new_csr->get_dimensions()[0] << std::endl;
    std::cout << "Number of edges: " << new_csr->get_num_nnz() << std::endl;

    auto new_row_ptr = new_csr->get_row_ptr();
    std::cout << "Degree of vertex 0: " << new_row_ptr[1]-new_row_ptr[0] << std::endl;
    std::cout << "Degree of vertex 1: " << new_row_ptr[2]-new_row_ptr[1] << std::endl;
    std::cout << "Degree of vertex 2: " << new_row_ptr[3]-new_row_ptr[2] << std::endl;

    return 0;
}