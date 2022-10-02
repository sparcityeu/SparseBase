# Usage
SparseBase can be easily added to your project either through CMake's `find_package()` command, or by directly linking it at compilation time.


## Adding SparseBase through CMake

> In the commands below replace 0.1.5 with the version of SparseBase you installed.

If you installed SparseBase to the default system directory, use the following the command in your `CMakeLists.txt` file to add the library to your project:
```cmake
find_package(sparsebase 0.1.5 REQUIRED)
```
However, if you installed the library to a different path, say `/custom/location/`, you must specify that path in the command:
```cmake
find_package(sparsebase 0.1.5 REQUIRED PATHS /custom/location/)
```
After the library is added to your project, you can simply link your targets to `sparsebase::sparsebase`:

```cmake
target_link_libraries(your_target sparsebase::sparsebase)
```

## Linking to SparseBase at compile time 
If the library is not built in header-only mode, you must link it to your targets by passing the appropriate flag for your compiler. For example, for `g++`, add the `-lsparsebase` flag:
```bash
g++ source.cpp -lsparsebase
```
If the library was installed to location other than the system-default, say `/custom/location/`, then make sure to guide the compiler to the locations of the headers and the binary:
```bash
g++ source.cpp -I/custom/location/include -L/custom/location/lib -lsparsebase
```

On the other hand, if the library was compiled in header-only mode, then you do not need to provide a linking flag. For instance, instead of the two commands used above for compilation, you can simply use the following commands: 
```bash
g++ source.cpp
```
And if the library is not installed in the system-default location:
```bash
g++ source.cpp -I/custom/location/include -L/custom/location/lib
```

## Tests

Users can run unit tests easily after building the project. To do so, they must configure CMake to compile tests:
```bash
mkdir build # if a build directory doesn't exist
cd build
cmake -DRUN_TESTS=ON ..
make
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

## Including SparseBase

SparseBase can be included using the ``sparsebase.h`` header file.

```cpp
#include "sparsebase.h"
```

If desired users can include individual namespaces using their respective headers. 
This can be useful to reduce compile times if the header only build is being used.

```cpp
#include "sparsebase/utils/io/reader.h"
#include "sparsebase/preprocess/preprocess.h"
```

## Creating a Format Object

Currently two sparse data formats are supported:
- COO (Coordinate List)
- CSR (Compressed Sparse Row)

In the code snippet below you can see the creation of a CSR type object 
which only contains connectivity information. As a result the value parameter is set to nullptr
```cpp
unsigned int row_ptr[4] = {0, 2, 3, 4};
unsigned int col[4] = {1, 2, 0, 0};

// There are 3 template parameters for each sparse data format
// First is IDType which is related to the size of the dimensions
// Second is NumNonZerosType which is related to the number of non-zeros stored
// Third is ValueType which determines the type of the stored values
sparsebase::format::CSR<unsigned int, unsigned int, unsigned int> csr(3, 3, row_ptr, col, nullptr);
```

> In the example, we are not using any values but we still set the ValueType parameter to ``unsigned int``.
> Due to how the internal memory management system works, these types can not be set to ``void``.
> In this case, the actual type used does not matter as long as it is one of the supported types.


In the code snippet below you can see the creation of a COO type object which also contains value information.

```cpp
int row[6] = {0, 0, 1, 1, 2, 2};
int col[6] = {0, 1, 1, 2, 3, 3};
int vals[6] = {10, 20, 30, 40, 50, 60};

// Unlike the previous example we are storing integer type values here
auto coo = new sparsebase::COO<int,int,int>(6, 6, 6, row, col, vals);
```



## Input

Currently, we support two sparse data file formats:
- Matrix Market Files (.mtx)
- Edge List Files

We can perform a read operation on these formats as shown below: 
```cpp
// Reading a mtx file into a COO format
auto reader = new sparsebase::utils::io::MTXReader<vertex_type, edge_type, value_type>(file_name);
auto coo = reader->ReadCOO();

// Reading an edge list file into a CSR format
auto reader2 = new sparsebase::utils::io::EdgeListReader<vertex_type, edge_type, value_type>(file_name);
auto csr = reader2->ReadCSR();
```

## Casting Formats

Many functions in the library return generic format pointers like ``FormatOrderTwo`` or ``FormatOrderOne`` to ensure flexibility. These pointers can easily be converted into concrete versions using the ``Convert<>()`` member function. 

```cpp
// Consider the scenario where you obtained a generic pointer from a function
sparsebase::format::FormatOrderTwo<int, int, int>* format = ...;

// If the type of this pointer is known, then you can simply use the Convert function
sparsebase::format::CSR<int,int,int>* csr = format->Convert<sparsebase::format::CSR>();

```

Alternatively, if the pointer you have is to the abstract ``Format`` class, then you should use the ``As<>()`` casting function. Keep in mind that you must pass, as a template argument to ``As<>()``, the format class you wish to cast to along with its templated types:

```cpp
// Consider the scenario where you obtained a generic pointer from a function
sparsebase::format::Format* format = ...;

// The template argument includes the class name along with its template types (int, int, int)
sparsebase::format::CSR<int,int,int>* csr = format->As<sparsebase::format::CSR<int, int, int>>();

```


## Converting Formats

As explained in the previous section, readers will read to different formats.

However, we can convert the data into the format we desire using the ``Convert`` member funciton:
```cpp
// Consider the scenario where you obtained a COO and want to convert it to a CSR
auto coo = ...; 

// Since we don't want the result to be in a external device such as a GPU, we will use a default CPUContext here
sparsebase::context::CPUContext cpu_context;

// Convert<>() function will convert to the desired format and cast the pointer to the right type
// The final parameter being true indicates a move conversion will be performed
// This will be faster but will invalidate the original coo matrix
// If both the coo and csr are needed you should pass false here
auto csr = coo->Convert<sparsebase::format::CSR>(&cpu_context, true);

```

> If the source and destination formats are the same the converter will simply do nothing.

## Ownership

SparseBase allows users to choose whether they want to take responsibility of managing the memory.
By default the data stored inside ``Format`` instances are considered owned by the instance and as a
result will be deallocated when the instance is deleted. This can be avoided either by creating the
instance with the ``kNotOwned`` parameter or by manually releasing the arrays from the instance.

```cpp
// Arrays owned and deallocated automatically by the Format instance
auto* csr_owned = new sparsebase::format::CSR<int,int,int>(4, 4, row_ptr, col, vals, sparsebase::format::kOwned);

// Users can release the arrays to prevent this deallocation
auto* vals = csr_owned->release_vals();

// Arrays owned and deallocated by the user
auto* csr_not_owned = new sparsebase::format::CSR<int,int,int>(4, 4, row_ptr, col, vals, sparsebase::format::kNotOwned);
```

> Format instances created within the library (for example when a matrix is read from a file using an MTXReader)
> will almost always be owned by the instance. The user can release the arrays manually as discussed 
> above if this is not desired.

## Working with Graphs

Graphs can be created using any SparseFormat as the connectivity information of the graph.

```cpp
auto reader = new sparsebase::utils::io::MTXReader<vertex_type, edge_type, value_type>(file_name);
auto data = reader->ReadCOO();
auto g = sparsebase::object::Graph<vertex_type, edge_type, value_type>(data);
```

Alternatively we can create a graph by directly passing the reader.

```cpp
 sparsebase::object::Graph<vertex_type, edge_type, value_type> g;
 g.read_connectivity_to_coo(sparsebase::MTXReader<vertex_type, edge_type, value_type>(file_name));
```

As of the current version of the library, graphs function as containers of sparse data. However, there are plans to expand this in future releases.

## Ordering

Various orderings can be generated for a graph using the ``ReorderPreprocessType`` classes. 
Currently these include ``RCMReoder`` and ``DegreeReorder``. There is also a ``GenericReorder`` class
allowing the users to define their own custom orderings. For more details, please see the examples
in the Github repository.

Below you can see an example of an RCM reordering of a graph.
```cpp
sparsebase::preprocess::RCMReorder<vertex_type, edge_type, value_type> orderer(1, 4);
sparsebase::format::Format<vertex_type, edge_type, value_type> * con = g.get_connectivity();
sparsebase::context::CPUContext cpu_context;
vertex_type * order = orderer.GetReorder(con, {&cpu_context});
```

Orders are returned as arrays which describe the transformation that needs to take place for the graph to be reordered.
So by default, reordering won't actually mutate the graph. If the user wishes to do so, they can use the `PermuteOrderTwo` class
to mutate the graph.

```cpp
// use `order` to permute both rows and columns
sparsebase::preprocess::PermuteOrderTwo<vertex_type, edge_type, value_type> permute(order, order);
sparsebase::format::Format *result = permute.GetTransformation(con, {&cpu_context});
```