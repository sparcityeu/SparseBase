# SparseBase

## Compiling
```
mkdir build && cd build
cmake ..
make
```

This will generate the library as a static library. In addition, the example codes are compiled and their binaries are located in `build/examples`.

## Installation

To install the library, compile the library as shwon in the previous section. Afterwards, you can install the library files either to the systems global location or to a custom location. To install the library to the default system location:
```
cd build
cmake --install .
```

To install the library to a custom directory, do the following:
```
cd build
cmake --install . --prefix "/custom/location"
```
**Note:** We suggest using **absolute paths** for prefixes as relative paths have been known to cause problems from time to time.

## Usage
SparseBase can be easily added to your project either through CMake's `find_package()` command, or by directly linking it at compilation time.
### Adding SparseBase through CMake
If you installed SparseBase to the default system directory, use the following the command in your `CMakeLists.txt` file to add the library to your project:
```
find_package(sparsebase 0.1 REQUIRED)
```
However, if you installed the library to a different path, say `/custom/location/`, you must specify that path in the command:
```
find_package(sparsebase 0.1 REQUIRED PATHS /custom/location/)
```
After the library is added to your project, you can simply link your targets to `sparsebase::sparsebase`:

```
target_link_libraries(your_target sparsebase::sparsebase)
```

### Linking to SparseBase at compile time 
You can link SparseBase directly to your targets by passing the appropriate flag for your compiler. For example, for `g++`, add the `-lsparsebase` flag:
```
g++ source.cpp -lsparsebase
```
If the library was installed to a different location, say `/custom/location/`, then make sure to guide the compiler to the locations of the headers and the binary:
```
g++ source.cpp -I/custom/location/include -L/custom/location/lib -lsparsebase
```

## Tests

Users can run unit tests easily after building the project. To do so, they must configure CMake to compile tests:
```
mkdir build # if a build directory doesn't exist
cd build
cmake -DRUN_TESTS ..
```
Once its built, while in the build directory, do the following:
``` 
ctest -V
```
## Formatting
Source files can be automatically formatted using `clang-format`. After installing `clang-format`, generate the build system using CMake and build the target `format`. This example shows its usage with `make`:
```
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

CSR<unsigned int, unsigned int, void> csr(3, 3, row_ptr, col, nullptr);
```

In the code snippet below you can see the creation of a COO type object which also contains value information.

```cpp
int row[6] = {0, 0, 1, 1, 2, 2};
int col[6] = {0, 1, 1, 2, 3, 3};
int vals[6] = {10, 20, 30, 40, 50, 60};

COO<int,int,int>* coo = new COO<int,int,int>(6, 6, 6, row, col, vals);
```


## Input

Currently, we support two sparse data file formats:
- Matrix Market Files (.mtx)
- Undirected Edge List Files (.uedgelist)

We can perform a read operation on these formats as shown below: (use `UedgeListReader` for .uedgelist files)
```cpp
auto reader = new MTXReader<vertex_type, edge_type, value_type>(file_name);
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
auto converter = SparseConverter<vertex_type, edge_type, value_type>();
auto converted = converter.convert(result, CSR_f);
auto csr = dynamic_cast<CSR<vertex_type, edge_type, value_type>>(converted);
```

## Working with Graphs

Graphs can be created using any SparseFormat as the connectivity information of the graph.

```cpp
auto reader = new MTXReader<vertex_type, edge_type, value_type>(file_name);
auto data = reader->read();
auto g = Graph<vertex_type, edge_type, value_type>(data);
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
ReorderInstance<vertex_type, edge_type, value_type, RCMReorder> orderer;
SparseFormat<vertex_type, edge_type, value_type> * con = g.get_connectivity();
vertex_type * order = orderer.get_reorder(con);
```

For these operations, we also support an alternative syntax (without the template parameter) using the ``RCMReorderInstance`` and ``DegreeReorderInstance`` wrapper classes.

Below you can see this alternative syntax being used to reorder the same graph.
```cpp
RCMReorderInstance<vertex_type, edge_type, value_type> orderer;
SparseFormat<vertex_type, edge_type, value_type> * con = g.get_connectivity();
vertex_type * order = orderer.get_reorder(con);
```

Orders are returned as arrays which describe the transformation that needs to take place for the graph to be reordered.
So by default, reordering won't actually mutate the graph. If the user wishes to do so, they can use the `TransformInstance` class

```cpp
TransformInstance<vertex_type, edge_type, value_type, Transform> transformer(1);
auto csr = transformer.get_transformation(con, order);
```

# Contribution Guidelines

Contributions preferably start with an issue on the issue tracker of GitHub. In addition, a contribution of any kind must be forked out of `origin/develop` and merged back into it. 

TL;DR: the process for making a contribution is to make a topic branch out of `origin/develop` into your local machine, make your contributions on this topic branch, push your new branch back into `origin`, and create a pull request to pull your new topic branch into `origin/develop`. Please do not merge your changes to `develop` on your local machine and push your changes to `origin/develop` directly. 

More precisely, a typical contribution will follow this pattern:

1. Create an issue on GitHub discussing your contribution. At this point, a discussion may happen where the entire team can get on the same page.
2. Pull `origin/develop` into your local to start developing from the latest state of the project, and create a new branch for your contribution. The naming convention for a contribution branch is `feature/<new_feature>`:
    
    ```bash
    # on your local
    cd sparsebase
    git checkout develop
    git pull origin develop
    git checkout -b feature/<new_feature>
    ```
    
3. After you're done working on your feature, make sure that it can be merged cleanly with `origin/develop` by pulling `origin/develop` back into your local machine and merging it with your feature branch:
    
    ```bash
    git checkout develop
    git pull origin develop
    git checkout feature/<new_feature>
    git merge develop
    # merge conflicts may arise
    ```
    
4. Once your feature branch merges successfully with `develop`, push your branch to `origin`:
    
    ```bash
    git checkout feature/<new_feature>
    git push origin feature/<new_feature>
    ```
    
5. On GitHub, create a pull request to merge your branch with `develop`; the base of the request will be `develop` and the merging branch will be `feature/<new_feature>`.
6.  Once the contribution is reviewed, a maintainer from the team will merge the pull request into `origin/develop`.

Thank you for your efforts!
