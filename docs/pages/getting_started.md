# Getting Started

## Installation & Building

### Requirements

Compilers listed below are the ones we used during development and testing.
Other compilers may work but are not officially supported at the moment.
Lowest versions are included based on the feature sets of the tools/compilers
and not all of them are tested.
We suggest using the most recent versions when possible.

- Supported Compilers:
    - clang++ (version 6 or above)
    - g++ (version 7 or above)
    - MSYS2 g++ ([mingw-w64-x86_64-gcc](https://packages.msys2.org/package/mingw-w64-x86_64-gcc?repo=mingw64))


- Build tools:
    - cmake (version 3.0 or above)
    - GNU make (version 4.1 or above) or ninja (version 1.11 or above)
    

### Compiling

> A version of the header-only release can be obtained in the releases section on Github 
> (no compilation needed)
> along with a documentation pdf and license information.
> We still suggest compiling and using a non-header-only build if possible
> for faster compile times on your projects.

```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCUDA={$CUDA} ..
make
```
Where `${CUDA}` is `ON` if CUDA support is needed and `OFF` otherwise.

This will generate the library as a static library. In addition, the example codes are compiled and their binaries are located in `build/examples`.

Due to optimizations and templates, this process might take several minutes.

### Specifying explicit instantiation types
Most classes in the library are templated over four template parameter types:

1. `IDType`: data type that will contain IDs of element in a format object.
2. `NNZType`: data type that will contain numbers of non-zeros in a format object. 
3. `ValueType`: data type that will contain values stored inside format objects.
4. `FloatType`: data type that will contain floating point values (e.g., degree distributions).

When generating the build system, you can specify a list of data types for each one of these template parameters. Explicit instantiations of all the classes in the library will be created to satisfy the power set of all the types you passed. You can specify the data types using CMake cache variables that you pass as semi-colon-seperated, quotation-enclosed lists. For example, the following command:
```bash
cmake -DID_TYPES="unsigned int; unsigned long long" -DNNZ_TYPES="unsigned long long; short" -DVALUE_TYPES="float" -DFLOAT_TYPES="double; float" ..
```
specifies the data types to be used with each of the four template parameter types stated above. 

Alternatively, you can edit the `CMakeCache.txt` file located in the build directory. Note that this file is generated after creating the build system (after executing `cmake`).

### Header-only


Additionally, the library has a header-only setting, in which none of the classes of the library will be explicitly instantiated at library-build time. Building the library to be header-only can be done as shown:
```
mkdir build && cd build
cmake -D_HEADER_ONLY=ON -DCMAKE_BUILD_TYPE=Release -DCUDA=${CUDA} ..
make
```
Where `${CUDA}` is `ON` if CUDA support is needed and `OFF` otherwise.

> Note: if the library is installed with `${CUDA}=ON`, the user code must be compiled using `nvcc`.

This will prepare the library for installation and compile the example codes located in `build/examples`.


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

### Linking to SparseBase at compile time 
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

### Tests

Users can run unit tests easily after building the project. To do so, they must configure CMake to compile tests:
```bash
mkdir build # if a build directory doesn't exist
cd build
cmake -DRUN_TESTS=ON ..
```
Once its built, while in the build directory, do the following:
```bash
ctest -V
```
### Formatting
Source files can be automatically formatted using `clang-format`. After installing `clang-format`, generate the build system using CMake and build the target `format`. This example shows its usage with `make`:
```bash
mkdir build
cd build
cmake ..
make format
``` 

### Including SparseBase

SparseBase can be included using the ``sparsebase.h`` header file.

```cpp
#include "sparsebase/sparsebase.h"
```

If desired users can include individual namespaces using their respective headers. 
This can be useful to reduce compile times if the header only build is being used.

```cpp
#include "sparsebase/utils/io/reader.h"
#include "sparsebase/preprocess/preprocess.h"
```

### Creating a Format Object

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



### Input

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

### Casting Formats

Many function in the library return generic ``Format`` pointers to ensure flexibility.
These pointers can easily be converted into concrete versions using the ``As<>()`` function of the
``Format`` class.

```cpp
// Consider the scenario where you obtained a generic pointer from a function
sparsebase::format::Format* format = ...;

// If the type of this pointer is known, then you can simply use the As function
// If the type is not known you can use the converters described in the next section
sparsebase::format::CSR<int,int,int>* csr = format->As<sparsebase::format::CSR<int,int,int>>();

```


### Converting Formats

As explained in the previous section, readers will read to different formats.

However, we can convert the data into the format we desire using ``Converter``:
```cpp
// Consider the scenario where you obtained a COO and want to convert it to a CSR
auto coo = ...; 

// Converter instances automatically select the correct conversion function and apply it
auto converter = sparsebase::utils::converter::Converter<vertex_type, edge_type, value_type>();

// Since we don't want the result to be in a device, we will use a default CPUContext here
sparsebase::context::CPUContext cpu_context;

// Convert<>() function will convert to the desired format and cast the pointer to the right type
// The final parameter being true indicates a move conversion will be performed
// This will be faster but will invalidate the original coo matrix
// If both the coo and csr are needed you should pass false here
auto csr = converter.Convert<sparsebase::format::CSR<vertex_type, edge_type, value_type>>(result, &cpu_context, true);

```

> If the source and destination formats are the same the converter will simply do nothing.

### Ownership

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

### Working with Graphs

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

### Ordering

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
So by default, reordering won't actually mutate the graph. If the user wishes to do so, they can use the `Transform` class
to mutate the graph.

```cpp
sparsebase::preprocess::Transform<vertex_type, edge_type, value_type> transformer(order);
sparsebase::format::Format *result = transformer.GetTransformation(con, {&cpu_context});
```