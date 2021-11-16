# sparsebase

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

## Usage

When compiling a project that uses SparseBase, simply link the project with the library using the flag `-lsparsebase`.

## Tests

Users can run unit tests easily after building the project. Once its built, do the following:
```
cd build 
ctest -V
```