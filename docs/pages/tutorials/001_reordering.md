# Reordering a graph

## Objective
Read a graph from a file and reorder it. 

## Overview

In this tutorial, you will use SparseBase to do the following:

1. Read a graph from an unordered edge list file.
2. Reorder the vertices of the graph according to their degree.
3. Restructure the graph according to the generated ordering.

## Preliminaries
Start by navigating to the directory `tutorials/001_reordering/start_here/`. Open the file `tutorial_001.cc` using a code editor and follow along with the tutorial. The file contains some boilerplate code that includes the appropriate headers, creates some type definitions, and uses the `sparsebase` namespace.

The completed tutorial can be found in `tutorials/001_reordering/solved/solved.cc`. We will use the unordered edge list file `tutorials/001_reordering/chesapeake.edgelist`. 

## Steps

### 1. Read the graph from disk
Begin your main program by reading the undirected edge list file into a `CSR` object using an `EdgeListReader` object. 

```c++
// The name of the edge list file in disk
std::string filename(argv[1]); 
// Create a reader object and set the name of the file it will read
utils::io::EdgeListReader<IDType, NNZType, ValueType> reader(filename);
// Read the file into a CSR format
format::CSR<IDType, NNZType, ValueType>* csr = reader.ReadCSR();
```

The three templated type parameters of the `CSR` and `EdgeListReader` objects determine the data types that will store the IDs, the number of non-zeros, and the values of the weights of the graph, respectively. These types are defined at the beginning of the file. Notice that, since the graph we read is unweighted, there will be no values in the `CSR` format object, only connectivity information. However, the type of the cannot be set to `void` due to internal details with smart pointers. For this reason, we set it to `unsigned int`. 

You will find that these three template types are used by most classes of the library.

To get a feel for the graph you just read, print some of its statistics:

```c++
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
```

### 2. Reorder the graph
Next, create a degree reordering of the graph:
```c++
// Create a DegreeReorder object and tell it to sort in descending order
bool ascending = false;
preprocess::DegreeReorder<IDType, NNZType, ValueType> reorderer(ascending);
// Create a CPU context
context::CPUContext cpu_context;
// Create a reordering of `csr` using one of the passed contexts 
// (in this case, only one is passed)
IDType* new_order = reorderer.GetReorder(csr, {&cpu_context});
```

The array `new_order` is an array containing the inverse permutation of all the vertices of `csr`. In other words, `new_order[i] = j` indicates that the vertex `i` in `csr` is at location `j` after reordering.

### 3. Use the reordering to restructure the graph
Finally, use the reordering array `new_order` to restrucuture the graph and apply the new order to it.

```c++
// Transform object takes the reordering as an argument
preprocess::Transform<IDType, NNZType, ValueType> transform(new_order);
// The transformer will use `new_order` to restructure `csr`
format::Format* format = transform.GetTransformation(csr, {&cpu_context});
// The abstract `Format` pointer is cast into a `CSR` pointer
format::CSR<IDType, NNZType, ValueType>* new_csr = 
  format->As<format::CSR<IDType, NNZType, ValueType>>();
```

The `GetTransformation()` call returns a `CSR` object, but it returns it as a polymorphic pointer to the superclass `Format`. The `As` function will statically cast that pointer to the correct type.

Let's print some statistics about the reordered graph.

```c++
std::cout << "Reordered graph:" << std::endl; 
std::cout << "Number of vertices: " << new_csr->get_dimensions()[0] << std::endl;
std::cout << "Number of edges: " << new_csr->get_num_nnz() << std::endl;

auto new_row_ptr = new_csr->get_row_ptr();
std::cout << "Degree of vertex 0: " << row_ptr[1]-row_ptr[0] << std::endl;
std::cout << "Degree of vertex 1: " << row_ptr[2]-row_ptr[1] << std::endl;
std::cout << "Degree of vertex 2: " << row_ptr[3]-row_ptr[2] << std::endl;
```

### 4. Compile the program and execute it
Compile the code using `g++`. We assume SparseBase has already been installed in the compiled setting (as opposed to header-only installation).

While in the directory `tutorials/001_reordering/start_here`, execute the following commands:
```bash
g++ -std=c++17 tutorial_001.cc -lsparsebase -fopenmp -std=c++17 -o reorder.out
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