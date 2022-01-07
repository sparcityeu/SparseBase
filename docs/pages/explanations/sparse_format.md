# Sparse Format Guide

`SparseFormat` is the building block of Sparsebase. A `SparseFormat` is designed to store the connection/relationship information of a Matrix, Graph, Tensor, etc. Thus an object or a function in Sparsebase makes use of a `SparseFormat` in some way or another. All sparse formats in the Sparsebase library implement the `SparseFormat` interface. This makes `SparseFormat` easily extendible and interchangeable.

## Sparse Format Interface

SparseFormat interface have three variable templates; `IDType`, `NNZType`, and `ValueType`. `IDType` is related to the size of the dimensions, where the largest dimension must be in the range of type `IDType`. Likewise, `NNZType` is related to the number of non-zeros stored, where the number of non-zeros must be in the range of type `NNZType`. Lastly, `ValueType` represents the type of the `vals` data member, which stores the weight of the connections. For instance, for a road network, `ValueType` can be a floating-point number representing the distance in between, or, for a social network, it can be of type void since the connections may not have a weight attached. 

```cpp
template <typename IDType, typename NNZType, typename ValueType> class SparseFormat {
public:
  virtual ~SparseFormat(){};
  virtual unsigned int get_order() = 0;
  virtual Format get_format() = 0;
  virtual std::vector<IDType> get_dimensions() = 0;
  virtual NNZType get_num_nnz() = 0;
  virtual NNZType *get_row_ptr() = 0;
  virtual IDType *get_col() = 0;
  virtual IDType *get_row() = 0;
  virtual ValueType *get_vals() = 0;
  virtual IDType **get_ind() = 0;
};
```
All classes that implement the Sparse Format interface store an enum type Format.

```cpp
  //! Enum keeping formats  
  enum Format{
    //! CSR Format
    CSR_f=0, 
    //! COO Format
    COO_f=1 
    //! CSF Format
    CSF_f=2 
  };
```
As you might guess, not all concrete Sparse Formats have the same data members. For instance, compressed sparse rows (CSR) do not have `is` as a data member. So what happens if we call the `get_row()` function from the CSR class? 

```cpp
try{ //if the data member is invalid, sparse format throws an exception
  auto is = csr->get_row();
}
catch(InvalidDataMember ex){
  cout << ex.what() << endl;
}
```
```bash
Format 0 does not have is as a data member.
```
Thankfully such cases are handled by Sparsebase, where `InvalidDataMember` exception is thrown. 

## Usage

Currently Sparsebase supports two ways to create a sparse format:

1. **User initialized.** The user passes the required data members of the respective sparse format through the constructor.
```cpp
#include "sparsebase/sparse_format.h"

unsigned int row_ptr[4] = {0, 2, 3, 4};
unsigned int col[4] = {1,2,0,0};
sparsebase::CSR<unsigned int, unsigned int, void> csr(3, 3, row_ptr, col, nullptr);
```

2. **Read directly from the file.** User first creates a reader by passing the file_name. Then calls the correct read function.
```cpp
#include "sparsebase/sparse_format.h"
#include "sparsebase/sparse_reader.h"

sparsebase::MTXReader<unsigned int, unsigned int, void> reader(file_name);
sparsebase::COO<unsigned int, unsigned int, void> * coo = reader.ReadCOO();
```

## Currently Supported Sparse Formats

| Type | ENUM | ORDER | READERS
| --- | --- | --- | --- |
| COO | kCSRFormat | 2 | MTX |
| CSR | kCOOFormat | 2 | UEDGELIST |

