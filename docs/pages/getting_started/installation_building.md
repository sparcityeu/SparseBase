# Installation & Building

## Requirements

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
    - cmake (version 3.12 or above) (with CUDA support enabled 3.18 or above)
    - GNU make (version 4.1 or above) or ninja (version 1.11 or above)

    
- Other dependencies:
  - Python (version 3.6 or above)


## Compiling

> A version of the header-only release can be obtained in the releases section on Github 
> (no compilation needed)
> along with a documentation pdf and license information.
> We still suggest compiling and using a non-header-only build if possible
> for faster compile times on your projects.

```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=OFF ..
make
```
This will build the library without CUDA support. If CUDA support is needed, then replace `-DUSE_CUDA=OFF` with `-DUSE_CUDA=ON`.

```{warning}
If the library is installed with cuda support, the user code must be compiled using `nvcc`.
```

This will generate the library as a static library. In addition, the example codes are compiled and their binaries are located in `build/examples`.

Due to optimizations and templates, this process might take several minutes.

## Specifying explicit instantiation types
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

## Header-only


Additionally, the library has a header-only setting, in which none of the classes of the library will be explicitly instantiated at library-build time. Building the library to be header-only can be done as shown:
```
mkdir build && cd build
cmake -D_HEADER_ONLY=ON -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=CUDA ..
make
```
Where `CUDA` is `ON` if CUDA support is needed and `OFF` otherwise.

```{warning}
If the library is installed with cuda support, the user code must be compiled using `nvcc`.
```

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
