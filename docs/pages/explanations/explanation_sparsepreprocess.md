# Representing operations using SparsePreprocess

## Overview

Preprocessing, as in the name, represents preprocessing operations that involve `SparseFormat` objects. Reordering, partitioning, graph coarsening, matrix factorization, and tensor decomposition are a few examples of such operations. Preprocessing in SparseBase is done using objects. Each object carries out a certain operation and can contain multiple functions that handle different `SparseFormat` formats. Every object also comes with a matching and conversion mechanism for these functions. This mechanism allows the object to take the user input, find a function that matches the input format, and if none are found, convert the input into a different format for which it has a function. This mechanism has two main benefits:

1. **Users need a single representation only.** Users do not need to worry about the format of their data. The library will take care of doing the necessary conversions to carry out their work. Simply load your data and pass it to a `SparsePreprocess` object and it will do the work for you.
2. **Developers don’t need to cover the entire format space.** When a developer wishes to add a `SparsePreprocess` implementation, e.g. a new reordering implementation, they do not need to account for every input format. A single implementation on a common format will be sufficient; any object that is convertible to that format will be able to use the reordering. Of course, if they wish, they could provide additional implementations for different sparse formats.

The rest of this document will describe the hierarchy of `SparsePreprocess`.

## Hierarchy

Since `SparsePreprocess` can be used as an abstraction for many different perprocessing operations. It is important to keep it as generic as possible. We achieve this generality by making `SparsePreprocess` abstract in terms of the *preprocess operation’s function signature*, or what we will refer to moving forward as `PreprocessFunction`. In other words, a preprocess type can be created for any combination of inputs and outputs — the data types of inputs and outputs have no restrictions, and there can be as many inputs as the developer needs. Obviously, inputs should have at least one `SparseFormat` object to justify using the `SparsePreprocess` matching mechanism.  

Using this generalization, we classify `SparsePreprocess` into different *types;* a `PreprocessType` is identified by the signature of its function, i.e., its `PreprocessFunction`. For example, reordering, partitioning, coarsening, and factorization are all different types, and this is the `PreprocessFunction` defining the reorder type:

```cpp
IDType* ReorderFunction(std::vector<SparseFormat<IDType, NumNonZerosType>*>)
```

Concretely, each `PreprocessType` is defined by a class that is templated using the type’s `PreprocessFunction`. 

Each `PreprocessType` can have multiple *implementations.* These are different algorithms or approaches used to carry out this preprocessing, with each being represented by a class. For example, METIS, Kernighan-Lin, and Fiduccia-Mattheyses are all different implementations of the partitioning `PreprocessType`. Every implementation can have multiple functions, each accommodating different `SparseFormat` formats. This is where the function matching and format conversion takes place. For example, the METIS partitioning implementation can have a single function only that takes a `SparseFormat` object with the CSR format as input. However, users will be able to use METIS partitioning even if their data was in a different format — as long as that format is convertible to CSR.  

Finally, each `PreprocessType` can have one or more `PreprocessInstance` classes. These are adapters that take an implementation class and expose an API to the user. Every `PreprocessType` has a `PreprocessInstance` that can be used with all of its implementations. However, one can also build instances for specific implementations as well. For example, the `ReorderInstance` is an instance that can be used with any reordering implementation (RCM, degree, etc.) and it will expose the same API to the user regardless of the implementation its given. On the other hand, the `RCMReorderInstance` is an instance that is specifically made for RCM reordering, and its API can be specialized for it. The following example demonstrates the usage of preprocessing instances:

```cpp
#include "sparsebase/sparse_format.h"
#include "sparsebase/sparse_preprocess.h"

sparsebase::COO<int, int, void> coo;
sparsebase::ReorderInstance<int, int, void, sparsebase::RCMReorder> rcm_orderer;
// Even though there is no function to carry out RCM Reordering 
// on a COO, the data will be converted to CSR behind the scenes
// and reordered.
auto order = rcm_reorderer.GetOrder(csr);
```

## Why use the instance class?

An important design decision was whether or not to merge the `PreprocessType` and `PreprocessInstance` classes into a single class. This would mean that a `PreprocessType` would be defined by its operation’s function signature, as well as its public API. Even though this approach might have resulted in cleaner code, we found it might not be as scalable as the current approach. The current approach allows for multiple APIs for a single preprocess type, which we believe might become necessary for more complex preprocessing types.
