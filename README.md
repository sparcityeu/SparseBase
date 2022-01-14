# SparseBase

SparseBase is a library built in C++ that encapsulates, preprocesses, and performs I/O operations on sparse data structures seamlessly and optimally to provide a backbone for algorithms that use these structures.

It is designed with HPC (High Performance Computing) usage at the forefront. It is meant as a container of sparse objects such as tensors, graphs, multi-graphs, and hypergraphs. It mainly focuses on re-ordering, partitioning, and coarsening sparse objects. Also, it supports many different formats (representations) of sparse data and enables smooth conversion between formats.

The library is still in early stages of development. As a result, the API is not stable and likely to change in the near future.

Please check our [documentation](https://sabanciparallelcomputing.github.io/sparsebase_docs/index.html) for tutorials, how-to guides, and a reference for the API.

Below you will find [instructions](#compiling) on building and installing the library, as well as some quick [tutorials](#getting-started) on its basic operations.

## Requirements

Compilers listed below are the ones we used during development and testing. 
Other compilers are not supported at the current release.
Lowest versions are included based on the feature sets of the tools/compilers
and not all of them are tested.
We suggest using more recent versions when possible.

- Supported Compilers:
  - clang++ (version 6 or above)
  - g++ (version 7 or above)

- Build tools:
  - cmake (version 3.0 or above)
  - GNU make (version 4.1 or above)

> We had some trouble with g++ version 11 on GNU/Linux. 
> For now we suggest using clang or a different version of g++ on GNU/Linux.

## Compiling
```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```

This will generate the library as a static library. In addition, the example codes are compiled and their binaries are located in `build/examples`.

Due to optimizations and templates, this process might take several minutes.

## Installation

To install the library, compile the library as shown in the previous section. Afterwards, you can install the library files either to the systems global location or to a custom location. To install the library to the default system location:
```bash
cd build
cmake --install .
```

To install the library to a custom directory, do the following:
```bash
cd build
cmake --install . --prefix "/custom/location"
```
**Note:** We suggest using **absolute paths** for prefixes as relative paths have been known to cause problems from time to time.

## Usage
SparseBase can be easily added to your project either through CMake's `find_package()` command, or by directly linking it at compilation time.
### Adding SparseBase through CMake
If you installed SparseBase to the default system directory, use the following the command in your `CMakeLists.txt` file to add the library to your project:
```cmake
find_package(sparsebase 0.1.3 REQUIRED)
```
However, if you installed the library to a different path, say `/custom/location/`, you must specify that path in the command:
```cmake
find_package(sparsebase 0.1.3 REQUIRED PATHS /custom/location/)
```
After the library is added to your project, you can simply link your targets to `sparsebase::sparsebase`:

```cmake
target_link_libraries(your_target sparsebase::sparsebase)
```

### Linking to SparseBase at compile time 
You can link SparseBase directly to your targets by passing the appropriate flag for your compiler. For example, for `g++`, add the `-lsparsebase` flag:
```bash
g++ source.cpp -lsparsebase
```
If the library was installed to a different location, say `/custom/location/`, then make sure to guide the compiler to the locations of the headers and the binary:
```bash
g++ source.cpp -I/custom/location/include -L/custom/location/lib -lsparsebase
```

## Tests

Users can run unit tests easily after building the project. To do so, they must configure CMake to compile tests:
```bash
mkdir build # if a build directory doesn't exist
cd build
cmake -DRUN_TESTS ..
```
Once its built, while in the build directory, do the following:
```bash
ctest -V
```
## Formatting
Source files can be automatically formatted using `clang-format`. After installing `clang-format`, generate the build system using CMake and build the target `format`. This example shows its usage with `make`:
```bash
mkdir build
cd build
cmake ..
make format
``` 

# Getting Started

## Creating a SparseFormat Object

Currently two sparse data formats are supported:
- COO (Coordinate List)
- CSR (Compressed Sparse Row)

In the code snippet below you can see the creation of a CSR type object
which only contains connectivity information. As a result the value type and parameter
are set as `void` and `nullptr respectively.
```cpp
unsigned int row_ptr[4] = {0, 2, 3, 4};
unsigned int col[4] = {1, 2, 0, 0};

sparsebase::CSR<unsigned int, unsigned int, void> csr(3, 3, row_ptr, col, nullptr);
```

In the code snippet below you can see the creation of a COO type object which also contains value information.

```cpp
int row[6] = {0, 0, 1, 1, 2, 2};
int col[6] = {0, 1, 1, 2, 3, 3};
int vals[6] = {10, 20, 30, 40, 50, 60};

auto coo = new sparsebase::COO<int,int,int>(6, 6, 6, row, col, vals);
```


## Input

Currently, we support two sparse data file formats:
- Matrix Market Files (.mtx)
- Undirected Edge List Files (.uedgelist)

We can perform a read operation on these formats as shown below: (use `UedgeListReader` for .uedgelist files)
```cpp
auto reader = new sparsebase::MTXReader<vertex_type, edge_type, value_type>(file_name);
auto data = reader->read_coo();
```

There are certain limitations to readers, which will be addressed in future releases:
- `UedgeListReader` can only read to CSR format
- `MTXReader` can only read to COO format
- Reading multiple tensors, matrices or graphs from a single file is not supported.


## Converting Formats

As explained in the previous section, readers will read to different formats.

However, we can convert the data into the format we desire using ``SparseConverter``:
```cpp
auto converter = sparsebase::SparseConverter<vertex_type, edge_type, value_type>();
auto converted = converter.convert(result, CSR_f);
auto csr = dynamic_cast<sparsebase::CSR<vertex_type, edge_type, value_type>>(converted);
```

## Working with Graphs

Graphs can be created using any SparseFormat as the connectivity information of the graph.

```cpp
auto reader = new sparsebase::MTXReader<vertex_type, edge_type, value_type>(file_name);
auto data = reader->read();
auto g = sparsebase::Graph<vertex_type, edge_type, value_type>(data);
```

Alternatively we can create a graph by directly passing the reader.

```cpp
 Graph<vertex_type, edge_type, value_type> g;
 g.read_connectivity_to_coo(MTXReader<vertex_type, edge_type, value_type>(file_name));
```

As of the current version of the library, graphs function as containers of sparse data. However, there are plans to expand this in future releases.

## Ordering

Ordering can be handled using the ``ReorderInstance`` class. Note that this class takes a ``ReorderPreprocessType`` as a template parameter.
As of the current version three such ``ReorderPreprocessType`` classes exist ``RCMReorder``, ``DegreeReorder``, ``GenericReorder``.

Below you can see an example of an RCM reordering of a graph.
```cpp
sparsebase::ReorderInstance<vertex_type, edge_type, value_type, sparsebase::RCMReorder> orderer;
sparsebase::SparseFormat<vertex_type, edge_type, value_type> * con = g.get_connectivity();
vertex_type * order = orderer.get_reorder(con);
```

For these operations, we also support an alternative syntax (without the template parameter) using the ``RCMReorderInstance`` and ``DegreeReorderInstance`` wrapper classes.

Below you can see this alternative syntax being used to reorder the same graph.
```cpp
sparsebase::RCMReorderInstance<vertex_type, edge_type, value_type> orderer;
sparsebase::SparseFormat<vertex_type, edge_type, value_type> * con = g.get_connectivity();
vertex_type * order = orderer.get_reorder(con);
```

Orders are returned as arrays which describe the transformation that needs to take place for the graph to be reordered.
So by default, reordering won't actually mutate the graph. If the user wishes to do so, they can use the `TransformInstance` class

```cpp
sparsebase::TransformInstance<vertex_type, edge_type, value_type, sparsebase::Transform> transformer;
auto csr = transformer.get_transformation(con, order);
```

For instructions on contributing to the repository, check out the [contribution guide](CONTRIBUTING.md).