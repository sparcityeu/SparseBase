#include "sparsebase/format/format.h"
#include "sparsebase/utils/io/reader.h"
#include "sparsebase/preprocess/preprocess.h"
#include "sparsebase/context/context.h"
#include <string>
#include <iostream>

typedef unsigned int IDType;
typedef unsigned int NNZType;
typedef unsigned int ValueType;

using namespace sparsebase;
int main(int argc, char * argv[]){
    if (argc < 2){
        std::cout << "Please enter the name of the edgelist file as a parameter\n";
        return 1;
    }
    // The name of the edge list file in disk
    std::string filename(argv[1]); 
    // Create a reader object and set the name of the file it will read
    utils::io::EdgeListReader<IDType, NNZType, ValueType> reader(filename);
    // Read the file into a CSR format
    format::CSR<IDType, NNZType, ValueType>* csr = reader.ReadCSR();

    std::cout << "Original graph:" << std::endl; 
    // get a vector representing the dimensions of the matrix represented by `csr`, 
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

    // Create a DegreeReorder object and tell it to sort in descending order
    bool ascending = false;
    preprocess::DegreeReorder<IDType, NNZType, ValueType> reorderer(ascending);
    // Create a CPU context
    context::CPUContext cpu_context;
    // Create a reordering of `csr` using one of the passed contexts 
    // (in this case, only one is passed)
    IDType* new_order = reorderer.GetReorder(csr, {&cpu_context});

    // Transform object takes the reordering as an argument
    preprocess::Transform<IDType, NNZType, ValueType> transform(new_order);
    // The transformer will use `new_order` to restructure `csr`
    format::Format* format = transform.GetTransformation(csr, {&cpu_context});
    // The abstract `Format` pointer is casted into a `CSR` pointer
    format::CSR<IDType, NNZType, ValueType>* new_csr = 
      format->As<format::CSR<IDType, NNZType, ValueType>>();

    std::cout << "Reordered graph:" << std::endl; 
    std::cout << "Number of vertices: " << new_csr->get_dimensions()[0] << std::endl;
    std::cout << "Number of edges: " << new_csr->get_num_nnz() << std::endl;

    auto new_row_ptr = new_csr->get_row_ptr();
    std::cout << "Degree of vertex 0: " << new_row_ptr[1]-new_row_ptr[0] << std::endl;
    std::cout << "Degree of vertex 1: " << new_row_ptr[2]-new_row_ptr[1] << std::endl;
    std::cout << "Degree of vertex 2: " << new_row_ptr[3]-new_row_ptr[2] << std::endl;

    return 0;
}