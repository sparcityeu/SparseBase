# Usage
SparseBase can be easily added to your project either through CMake's `find_package()` command, or by directly linking it at compilation time.


## Adding SparseBase through CMake

```{note}
In the commands below replace 0.1.5 with the version of SparseBase you installed.
```

If you installed SparseBase to the default system directory, use the following command in your `CMakeLists.txt` file to add the library to your project:
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
#include "sparsebase/io/io.h"
#include "sparsebase/preprocess/preprocess.h"
```

## Aliasing

SparseBase classes and namespaces are named in a rather verbose way. 
This is done to keep things well-structured during development.
However, users may find it difficult to work with. 
In such cases the namespaces can be aliased using the C++11 `using` keyword.


```cpp
using sbfo = sparsebase::format;
using sbio = sparsebase::io;
using sbco = sparsebase::converter;
using sbfe = spasebase::feature;
using sbob = sparsebase::object;
using sbpe = sparsebase::preprocess;
using sbco = sparsebase::context;
using sbut = sparsebase::utils;
```

## Template Types

To be flexible, efficient and safe, SparseBase classes take most of the types they use as
template parameters. However, users may find this difficult to work with.
In such cases, the commonly used types can be combined in a `#define` statement for ease of use.

```cpp
// Definitions
#define iif int,int,float
#define iid int,int,double
#define iii int,int,int

// Usage
CSR<iif> csr = ...;
```


## Creating a Format Object

Multiple sparse data formats are supported including:
- COO (Coordinate List)
- CSR (Compressed Sparse Row)

In the code snippet below you can see the creation of a CSR type object 
which only contains connectivity information. As a result the value argument is set to `nullptr` and the last template argument (`ValueType`) is set to `void`.

```cpp
unsigned int row_ptr[4] = {0, 2, 3, 4};
unsigned int col[4] = {1, 2, 0, 0};

// There are 3 template parameters for each sparse data format
// First is IDType which is related to the size of the dimensions
// Second is NumNonZerosType which is related to the number of non-zeros stored
// Third is ValueType which determines the type of the stored values
sparsebase::format::CSR<unsigned int, unsigned int, void> csr(3, 3, row_ptr, col, nullptr);
```

In the code snippet below you can see the creation of a COO type object which contains value information.

```cpp
int row[6] = {0, 0, 1, 1, 2, 2};
int col[6] = {0, 1, 1, 2, 3, 3};
int vals[6] = {10, 20, 30, 40, 50, 60};

// Unlike the previous example we are storing integer type values here
auto coo = new sparsebase::format::COO<int,int,int>(6, 6, 6, row, col, vals);
```

```{note}
SparseBase is designed with HPC users in mind, so the underlying arrays of the formats
are always accessible through the various get_... functions.
```


## Casting Formats

Many functions in the library return generic format pointers like ``FormatOrderTwo`` or ``FormatOrderOne`` to ensure flexibility. These pointers can easily be converted into concrete versions using the ``As<>()`` member function. 

```cpp
// Consider the scenario where you obtained a generic pointer from a function
sparsebase::format::FormatOrderTwo<int, int, int>* format = ...;

// If the type of this pointer is known, then you can simply use the As function
sparsebase::format::CSR<int,int,int>* csr = format->As<sparsebase::format::CSR>();

```

Alternatively, if the pointer you have is to the abstract ``Format`` class, then you should use the ``AsAbsolute<>()`` casting function. Keep in mind that you must pass, as a template argument to ``AsAbsolute<>()``, the format class you wish to cast to along with its templated types:

```cpp
// Consider the scenario where you obtained a generic pointer from a function
sparsebase::format::Format* format = ...;

// The template argument includes the class name along with its template types (int, int, int)
sparsebase::format::CSR<int,int,int>* csr = format->AsAbsolute<sparsebase::format::CSR<int, int, int>>();

```


```{warning}
Casting can only be successful if the provided type is a valid type for the given pointer.
The As and AsAbsolute functions will never perform a conversion. They will only cast.
```


## Converting Formats

We can convert between different data formats using the `Convert<>()` function.

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

```{note}
If the format object is already of the desired type, 
the Convert function will not do anything besides type checking.
```

```{warning}
Conversion is only allowed between formats of the same order.
So you can not convert a FormatOrderOne to FormatOrderTwo.
```

## Input

Currently, we support two sparse data file formats:
- Matrix Market Files (.mtx)
- Edge List Files

Reading such files can easily be done using the `IOBase` class.

```cpp
auto coo = sparsebase::io::IOBase::ReadMTXtoCOO<int,int,float>();
auto csr = sparsebase::io::IOBase::ReadEdgeListtoCSR<int,int,float>();
```

```{note}
Alternatively the ReadPigoMTX... and ReadPigoEdgeList... functions can be used.
These use the PIGO library to read the files in a multi-threaded fashion.
However they may not support all the options of our default readers.
```


Users can also use the underlying reader classes directly if need be. 
```cpp
// Reading a mtx file into a COO format
auto reader = new sparsebase::io::MTXReader<int, int, float>(file_name);
auto coo = reader->ReadCOO();

// Reading an edge list file into a CSR format
auto reader2 = new sparsebase::io::EdgeListReader<int, int, float>(file_name);
auto csr = reader2->ReadCSR();
```


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

```{note}
Format instances created within the library (for example when a matrix is read from a file using an MTXReader)
will almost always be owned by the instance. The user can release the arrays manually as discussed 
above if this is not desired.
```

## Working with Graphs

Graphs can be created using any Format as the connectivity information of the graph.

```cpp
auto reader = new sparsebase::io::MTXReader<vertex_type, edge_type, value_type>(file_name);
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

Sparse data formats can be reordered easily using the `ReorderBase` class.


```cpp
sparsebase::preprocess::DegreeReorderParams params(true);
sparsebase::context::CPUContext cpu_context;
IDType* order = ReorderBase::Reorder<DegreeReorder>(params, format, {&cpu_context}, true);
```

Multiple different reordering algorithms are supported including `DegreeReorder`, `RCMReorder` and `GrayReorder`.

Each reordering algorithm supports a set of parameters which is represented by a struct named in 
the `<Algorithm>Params` format. It is also accessible as a static member of each algorithm as `<Class>::ParamsType`.
The user can override these parameters by creating an object of this struct and changing its member variables.
This object can then be passed to the `Reorder` function as shown on the code snippet above.


As an alternative to `ReorderBase`, the user can also directly call the underlying reordering classes.
Below you can see an example of an RCM reordering of a graph using this method.

```cpp
sparsebase::preprocess::RCMReorder<int,int,float> orderer;
sparsebase::context::CPUContext cpu_context;
IDType * order = orderer.GetReorder(format, {&cpu_context});
```

In both cases the returned value is an array describing the reordering.
So if a reordered format is desired, the reordering needs to be applied to the format.
This can be done using `ReorderBase` or again manually as shown below.

```cpp
// Using ReorderBase
auto new_format = ReorderBase.Permute2D(order, format, {&cpu_context}, true);

// Manual Method
preprocess::PermuteOrderTwo<int, int, float> permute(order, order);
auto new_format = permute.GetPermutation(format, {&cpu_context});
```
