# Reordering a graph

## Objective
Read a graph from a file and reorder it. 

## Overview

In this tutorial, you will use SparseBase to do the following:

1. Read a graph from an unordered edge list file.
2. Reorder the vertices of the graph according to their degree.
3. Permute the graph according to the generated ordering.
4. Compile the program and execute it.

## Preliminaries
Start by navigating to the directory `tutorials/001_reordering/start_here/`. Open the file `tutorial_001.cc` using a code editor and follow along with the tutorial. The file contains some boilerplate code that includes the appropriate headers, creates some type definitions, and uses the `sparsebase` namespace.

The completed tutorial can be found in `tutorials/001_reordering/solved/solved.cc`. We will use the unordered edge list file `tutorials/001_reordering/chesapeake.edgelist`. 

## Steps

### 1. Read the graph from disk
Begin your main program by reading the undirected edge list file into a `CSR` object using the `ReadEdgeListToCSR` function in `IOBase`. 

```c++
    // The name of the edge list file in disk
    std::string filename(argv[1]);
    // Read the edge list file into a CSR object
    CSR<id_type, nnz_type, value_type>* csr = IOBase::ReadEdgeListToCSR<id_type, nnz_type, value_type>(filename);
```

The three templated type parameters of the `CSR` and `ReadEdgeListToCSR` objects determine the data types that will store the IDs, the number of non-zeros, and the values of the weights of the graph, respectively. These types are defined at the beginning of the file. Notice that, since the graph we read is unweighted, there will be no values in the `CSR` format object, only connectivity information. That is why we set `value_type` to void, though we could have set it to other types.

You will find that these three template types are used by most classes of the library.

To get a feel for the graph you just read, print some of its statistics:

```c++
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
```

### 2. Reorder the graph
Next, create a degree reordering of the graph:
```c++
// Create a CPU context
context::CPUContext cpu_context;
// We would like to order the vertices by degrees in descending order
bool ascending = false;
DegreeReorderParams params(ascending);
// Create a permutation array of `csr` using one of the passed contexts 
// (in this case, only one is passed)
// The last argument tells the function to convert the input format if needed
id_type* new_order = ReorderBase::Reorder<DegreeReorder>(params, csr, {&cpu_context}, true);
```

When calling the `Reorder` function, we pass the reordering class we want to use as a template parameter, in this case, that would be `DegreeReorder`. We also pass a struct containing the hyperparameters of this specific reordering as the first argument of the function.

The array `new_order` is an array containing the inverse permutation of all the vertices of `csr`. In other words, `new_order[i] = j` indicates that the vertex `i` in `csr` is at location `j` after reordering.

### 3. Use the reordering to restructure the graph
Finally, use the permutation array `new_order` to restructure the graph and apply the new order to it.

```c++
// Permute2D permutes the rows and columns of `csr` according to `new_order`
// Similar to `Reorder`, we specify the contexts to use, 
// and whether the library can convert the input if needed
FormatOrderTwo<id_type, nnz_type, value_type>* new_format = 
        ReorderBase::Permute2D(new_order, csr, {&cpu_context}, true);
// Cast the polymorphic pointer to a pointer at CSR
CSR<id_type, nnz_type, value_type>* new_csr = new_format->As<CSR>();
```

The `Permute2D()` call returns a `CSR` object, but it returns it as a polymorphic pointer to the superclass `FormatOrderTwo` which is the parent of `CSR` and other order-2 formats. The `As` function will safely cast that pointer to the correct type.

Let's print some statistics about the reordered graph.

```c++
std::cout << "Reordered graph:" << std::endl; 
std::cout << "Number of vertices: " << new_csr->get_dimensions()[0] << std::endl;
std::cout << "Number of edges: " << new_csr->get_num_nnz() << std::endl;

auto new_row_ptr = new_csr->get_row_ptr();
std::cout << "Degree of vertex 0: " << new_row_ptr[1]-new_row_ptr[0] << std::endl;
std::cout << "Degree of vertex 1: " << new_row_ptr[2]-new_row_ptr[1] << std::endl;
std::cout << "Degree of vertex 2: " << new_row_ptr[3]-new_row_ptr[2] << std::endl;
```

### 4. Compile the program and execute it
Compile the code using `g++`. We assume SparseBase has been compiled without `CUDA` support and without any optional libraries. Also, we assume it has already been installed in the compiled setting (as opposed to header-only installation).

While in the directory `tutorials/001_reordering/start_here`, execute the following commands:
```bash
g++ -std=c++17 tutorial_001.cc -lsparsebase -fopenmp -lgomp -std=c++17 -o reorder.out
./reorder.out ../chesapeake.edgelist
```

You should see the following output:

```
Original graph:
Number of vertices: 40
Number of edges: 340
Degree of vertex 0: 0
Degree of vertex 1: 11
Degree of vertex 2: 11

Reordered graph:
Number of vertices: 40
Number of edges: 340
Degree of vertex 0: 33
Degree of vertex 1: 29
Degree of vertex 2: 18
```