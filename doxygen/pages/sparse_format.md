# Sparse Format Guide

`SparseFormat` is the building block of Sparsebase. A `SparseFormat` is designed to store the connection/relationship information of a Matrix, Graph, Tensor, etc. Thus an object or a function in Sparsebase makes use of a `SparseFormat` in some way or another. All sparse formats in the Sparsebase library implement the `SparseFormat` interface. This makes `SparseFormat` easily extendible and interchangeable.

## Sparse Format Interface

SparseFormat interface have three variable templates; `ID_t`, `NNZ_t`, and `VAL_t`. `ID_t` is related to the size of the dimensions, where the largest dimension must be in the range of type `ID_t`. Likewise, `NNZ_t` is related to the number of non-zeros stored, where the number of non-zeros must be in the range of type `NNZ_t`. Lastly, `VAL_t` represents the type of the `vals` data member, which stores the weight of the connections. For instance, for a road network, `VAL_t` can be a floating-point number representing the distance in between, or, for a social network, it can be of type void since the connections may not have a weight attached. 

```cpp
template<typename ID_t, typename NNZ_t, typename VAL_t>
class SparseFormat{
    public:
    Format format;
    virtual ~SparseFormat(){};
    virtual unsigned int get_order() = 0;
    virtual Format get_format() = 0;
    virtual std::vector<ID_t> get_dimensions() = 0;
    virtual NNZ_t get_num_nnz() = 0;
    virtual NNZ_t * get_xadj() = 0;
    virtual ID_t * get_adj() = 0;
    virtual ID_t * get_is() = 0;
    virtual VAL_t * get_vals() = 0;
    virtual ID_t ** get_ind() = 0;
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
As you might guess, not all concrete Sparse Formats have the same data members. For instance, compressed sparse rows (CSR) do not have `is` as a data member. So what happens if we call the `get_is()` function from the CSR class? 

```cpp
    try{ //if the data member is invalid, sparse format throws an exception
      auto is = csr->get_is();
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
  unsigned int xadj[4] = {0, 2, 3, 4};
  unsigned int adj[4] = {1,2,0,0};
  CSR<unsigned int, unsigned int, void> csr(3, 3, xadj, adj, nullptr);
```

2. **Read directly from the file.** User first creates a reader by passing the file_name. Then calls the correct read function.
```cpp
  MTXReader<unsigned int, unsigned int, void> reader(file_name);
  COO<unsigned int, unsigned int, void> * coo = reader.read_coo();
```

## Currently Supported Sparse Formats

| Type | ENUM | ORDER | READERS
| --- | --- | --- | --- |
| COO | COO_f | 2 | MTX |
| CSR | CSR_f | 2 | EDGELIST |
