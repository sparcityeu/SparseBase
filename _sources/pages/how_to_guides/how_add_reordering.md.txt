# How to: add a new reordering algorithm

## Objective
This guide demonstrates how to add a new reordering algorithm to the library.
## Overview

Adding new reordering algorithms to SparseBase consists of five steps:

1. Create a class for your ordering. 
2. Create a struct that will contain the hyperparameters needed by your ordering. 
3. Add implementation functions that will carry out the reordering. 
4. Add a constructor to your reordering class.
5. Add explicit template instantiations for your class.

## Steps

In this guide, we will create a new reordering `OptimalReorder`. This reordering has the following properties:

- It requires two floating number hyperparameters for execution, `alpha` and `beta`.
- It has two implementations. One that operates on a `CSR` format, and another that operates on a `CUDACSR` format, i.e., a `CSR` that is stored on a `CUDA` GPU.

### 1. Create a new class for the ordering

The class will be split into a header file and an implementation file. Both files will be stored in the directory `src/sparsebase/reorder` and will have the same name as the class but in [snake case](https://en.wikipedia.org/wiki/Snake_case). For `OptimalReorder`, the files will be `optimal_reorder.h` and `optimal_reorder.cc`. At the top of the header file, include the following headers:
```c++
// Flags containing compilation flags, e.g. USE_CUDA
#include "sparsebase/config.h"
// Definition of base reordering class
#include "sparsebase/reorder/reorderer.h"
// Definition of parameters struct
#include "sparsebase/utils/parameterizable.h"
```

And at the top of the implementation file, include the created header.
```c++
#include "sparasebase/reorder/optimal_reorder.h"
```

Next, add the decleration of your class to the header file. You should add your class under the namespace `sparsebase::reorder`. It must be templated on the three types `IDType`, `NNZType`, and `ValueType` which define the data types of the `Format` objects it will reorder. Also, it must inherit from the class `Reorderer<IDType>` which defines the common API of all reordering classes.

Here is the decleration of `OptimalReorder` in the header file:
```cpp
// File: src/sparsebase/reorder/optimal_reorder.h
namespace sparsebase::reorder {
template <typename IDType, typename NNZType, typename ValueType>
class OptimalReorder : public Reorderer<IDType> {};
} // namespace sparsebase::reorder
```

For now, the definition file will be empty.

Finally, we must include the definition file inside the header file to enable header-only usage of the class. We make this inclusion conditional on the preprocessor directive `_HEADER_ONLY`. We make this addition to `optimal_reorder.h` as follows:
```c++
// File: src/sparsebase/reorder/optimal_reorder.h
namespace sparsebase::reorder {
template <typename IDType, typename NNZType, typename ValueType>
class OptimalReorder : public Reorderer<IDType> {};
} // namespace sparsebase::reorder

#ifdef _HEADER_ONLY
#include "sparsebase/reorder/optimal_reorder.cc"
#endif
```

Notice that the include is added _outside_ the `sparsebase::reorder` namespace.

> **Compiled vs. header-only**. In header-only mode, the user includes the code they want to use in their own code and compiles it as needed. In the compiled mode, the library classes and functions are precompiled into a static library and the user links to them at compile-time.

### 2. Create a struct containing the hyperparameters you need

In the header file created in step 1, create a new struct inheriting from `utils::Parameters`. Its members will be whichever hyperparameters your reordering will require. The naming convention for these structs is the name of the reordering class suffixed with `Params`. For our class, that would be `OptimalReorderParams`. We add `alpha` and `beta` to it. You may also add custom constructors for your parameter struct.

```cpp
// File: src/sparsebase/reorder/optimal_reorder.h
namespace sparsebase::reorder {
struct OptimalReorderParams : utils::Parameters {
    float alpha;
    float beta;
    OptimalReorderParams(float a, float b): alpha(a), beta(b){}
}
} // namespace sparsebase::reorder
```
Note: you still need to create such a struct even if your reordering class does not require any hyperparameters. However, you may leave it as an empty struct.

Additionally, create a `typedef` for your struct as `ParamsType` inside the reordering class you created. This is needed by the `Reorder` function in `ReorderBase`, which is the interface most users will be using to access reordering classes.

```cpp
// File: src/sparsebase/reorder/optimal_reorder.h
namespace sparsebase::reorder {
template <typename IDType, typename NNZType, typename ValueType>
class OptimalReorder : public Reorderer<ValueType> {
    // ...
    typedef OptimalReorderParams ParamsType;
    // ...
};
} // namespace sparsebase::reorder
```

### 3. Add implementation functions

Add to the class the implementation functions that will carry out the reordering. Each function will be specific for an input `Format` format type. These functions should match the following signature:

```cpp
static IDType* FunctionName(std::vector<format::Format*>, utils::Parameters*) 
```
The functions must be *static*. This is required to enable the mechanism of choosing the correct implementation function for the input `Format`'s format type.  

The parameters that your function will take are:

1. A vector of pointers at `Format` objects.
2. A pointer at a `utils::Parameters` struct. This pointer is polymorphic, and when this function is called, it will be pointing at an instance of the parameters struct created for your ordering. In our case, that would be an `OptimalReorderParams` object. 

Generally, all implementation functions will start with the same three steps:
1. Cast the input `Format` objects to the correct concrete type.
2. Cast the input `utils::Parameters` to the params struct created for this class.
3. Fetch the `Context` of the input `Format` object (this step is not needed for reordering on the CPU, but is necessary when using other architectures, e.g. `CUDA`).

For our example, `OptimalReorder` will have two implementation functions, `OptimallyOrderCSR()` and `OptimallyOrderCUDACSR()`. The former will reorder `CSR` objects on the CPU, and the latter will reorder `CUDACSR` objects, i.e., `CSR` objects stored on a `CUDA` GPU. 

#### 3.1 Adding CPU function implementations
You must add the implementation function according to the aforementioned signature and follow the steps mentioned above, namely casting the format and param objects, and fetching the context of the input.

First, add the decleration of the function to the header file.
```cpp
// File: src/sparsebase/reorder/optimal_reorder.h
namespace sparsebase::reorder {
template <typename IDType, typename NNZType, typename ValueType>
class OptimalReorder : public Reorderer<IDType> {
  //.......
  static IDType *OptimallyOrderCSR(
      std::vector<format::Format*> input_sf,
      utils::Parameters *poly_params);
};
} // namespace sparsebase::reorder
```

Then add the definition of the function to the `.cc` file. Don't forget to include the header of the format for which you are writing the implementation. In this case, that would be `csr.h`
```c++
// File: src/sparsebase/reorder/optimal_reorder.cc

// Header of OptimalReorder
#include "sparasebase/reorder/optimal_reorder.h"
// The header of the CSR class
#include "sparsebase/format/csr.h"

namespace sparsebase::reorder {
template <typename IDType, typename NNZType, typename ValueType>
IDType *OptimalReorder<IDType, NNZType, ValueType>::OptimallyOrderCSR(
  std::vector<format::Format*> input_sf,
  utils::Parameters *poly_params) {
    // safely cast the Format pointer to a CSR pointer
    auto csr = input_sf[0]->AsAbsolute<format::CSR<IDType, NNZType, ValueType>>();
    OptimalReorderParams *params =
        static_cast<OptimalReorderParams *>(poly_params);
    context::CPUContext *cpu_context =
        static_cast<context::CPUContext *>(csr->get_context());
    // ... carry out the ordering logic
    return order;
}
} // namespace sparsebase::reorder
```

#### 3.b Adding `CUDA` GPU function implementations
Adding the implementation for `OptimallyOrderCUDACSR()` follows the same process as `OptimallyOrderCSR()` except for a major difference: it will use a `CUDA` kernel during its execution. This poses a problem since `CUDA` kernels need to be compiled by `nvcc`, not by a pure C++ compiler. The solution to this issue is to add `CUDA` kernels to a separate `.cu` file, and to use driver functions to interface with them. The kernel and driver functions will be compiled using `nvcc`, but the driver function's signature will not have any `CUDA` special syntax and will be included in the non-`CUDA` file. This way, the pure `C++` code can make calls to the driver function, and the driver function will be able to call the GPU kernel since it is compiled with `nvcc`. The process for `OptimalReorder` is shown in the following image:
![optimal_reorder_cudacsr](res/optimal_reorder_cudacsr.png)
In the `CUDA` file `src/sparsebase/reorder/optimal_reorder_cuda.cu`, we define the GPU kernel `OptimallyOrderCUDACSRKernel()` (shown in red) and the driver function `OptimallyOrderCUDACSRDriver()` (shown in blue). We add the signature of the driver function to a header file `src/sparsebase/reorder/optimal_reorder_cuda.cuh`. Now, in the pure `C++` file, `src/sparsebase/reorder/optimal_reorder.cc`, we include the header file `sparsebase/reorder/optimal_reorder_cuda.cuh`. This way we can call it inside the implementation function `OptimallyOrderCUDACSR()` (shown in green).

Note that `CUDA` functions related to reordering and processing in general should be added to a file with the same name as the processing file, with the suffix `_cuda`.

Let's add these functions to our code. Add the `CUDA` kernel `OptimallyOrderCUDACSRKernel()` to the file `sparsebase/reorder/optimal_reorder_cuda.cu` under the namespace `sparsebase::reorder`. This function carries out the reordering on the GPU. In the same file and under the same namespace, add the driver function `OptimallyOrderCUDACSRDriver()` that dispatches the kernel. 

```cpp
// File: sparsebase/reorder/optimal_reorder_cuda.cu
namespace sparsebase::reorder {
template <typename IDType, typename NNZType>
__global__ void OptimallyOrderCUDACSRKernel(IDType *order, IDType *row_ptr,
                                           NNZType *col, IDType n) {
  // ... carry out ordering
}

template <typename IDType, typename NNZType, typename ValueType>
IDType *
OptimallyOrderCUDACSRDriver(format::CUDACSR<IDType, NNZType, ValueType> *csr,
                               context::CUDAContext context) {
  // set up context and get pointers from format
  OptimallyOrderCUDACSRKernel<<<...>>>(...);
  // fetch output from GPU
  return order;
}
} // namespace sparsebase::reorder
```

Next, add the signature of the driver function to the file `sparsebase/reorder/optimal_reorder_cuda.cuh` in order for the implementation function to be able to use it.

```cpp
// File: src/sparsebase/reorder/optimal_reorder_cuda.cuh
namespace sparsebase::reorder {
template <typename IDType, typename NNZType, typename ValueType>
IDType *
OptimallyOrderCUDACSRDriver(format::CUDACSR<IDType, NNZType, ValueType> *csr,
                               context::CUDAContext context);
}
```

Finally, add the function `OptimallyReorderCUDACSR()` as an implementation inside the `OptimalReorder` class. Note that the function decleration and definition are enclosed in an `#ifdef USE_CUDA` preprocessor block. This will guarantee that the function is not compiled unless compilation of the library with `CUDA` is enabled.

First, add the function decleration to the header file:
```cpp
// File: src/sparsebase/reorder/optimal_reorder.h
namespace sparsebase::reorder {
template <typename IDType, typename NNZType, typename ValueType>
class OptimalReorder : public Reorderer<IDType> {
// We do not want the function to be compiled if CUDA isn't enabled
#ifdef USE_CUDA
  static IDType *OptimallyOrderCUDACSR(
      std::vector<format::Format*> input_sf,
      utils::Parameters *poly_params);
#endif
  // .......
};
} // namespace sparsebase::reorder
```
Then define it in the implementation file. Don't forget to include the `CUDA` header with the driver decleration we created earlier:
```cpp
// File: src/sparsebase/reorder/optimal_reorder.h
//...
#include <sparsebase/reorder/optimal_reorder_cuda.cuh"
//...
namespace sparsebase::reorder {
// ...
#ifdef USE_CUDA
template <typename IDType, typename NNZType, typename ValueType>
IDType *OptimallyOrderCUDACSR(
      std::vector<format::Format*> input_sf,
      utils::Parameters *poly_params) {
  auto cuda_csr =
      input_sf[0]->AsAbsolute<format::CUDACSR<IDType, NNZType, ValueType>>();
  OptimalReorderParams *params =
      static_cast<OptimalReorderParams *>(poly_params);
  context::CPUContext *cuda_context =
      static_cast<context::CUDAContext *>(cuda_csr->get_context());
  // ...
  // use the driver to call the CUDA kernel
  order = OptimallyOrderCUDACSRDriver(cuda_csr, *cuda_context);
  // ...
  return order;
}
#endif
    // ...
} // namespace sparsebase::reorder
```

### 4. Create a constructor for your reordering class
The constructor of a reordering class (and, in general, of any processing class) does two things:

1. set the hyperparameters of class instance created,
2. register implementation functions to the right format types. 

Reordering classes can have as many constructors as the developer needs. However, at least one of them is required to have the following signature:
```c++
ReorderingClass(ReorderingClassParams);
```
Where `ReorderingClassParams` is the struct representing the hyperparameters of the reordering (the one we created in step 2).

We add the constructor decleration to the header file:

```cpp
// File: src/sparsebase/reorder/optimal_reorder.h
namespace sparsebase::reorder {
template <typename IDType, typename NNZType, typename ValueType>
class OptimalReorder : public Reorderer<IDType> {
    OptimalReorder(OptimalReorderParams params);
};
} // namespace sparsebase::reorder
```

And we add the definition to the implementation file:

```cpp
// File: src/sparsebase/reorder/optimal_reorder.cc
namespace sparsebase::reorder {
template <typename IDType, typename NNZType, typename ValueType>
OptimalReorder(OptimalReorderParams params){
    
}
} // namespace sparsebase::reorder
```

Now, let's populate this constructor.

#### 4.1 Set the hyperparameters of instances of the class

Use the hyperparameter argument from the user and set the data member `params_`, which the class inherited from `Reorderer`, by copying the argument passed in the constructor.

```cpp
// File: src/sparsebase/reorder/optimal_reorder.cc
namespace sparsebase::reorder {
template <typename IDType, typename NNZType, typename ValueType>
OptimalReorder(OptimalReorderParams params) {
  this->params_ =
      make_unique<OptimalReorderParams>(params);
}
} // namespace sparsebase::reorder
```

#### 4.2 Register the implementations you wrote to the correct formats

Register the implementation functions you made in step 3 to the `Format` type they are made for. This is how the library will be able to associate implementation functions with format types. 

Note that registering the `CUDACSR` implementation function is surrounded by an `#ifdef USE_CUDA` block to prevent it from being registered if the library is not compiled with `CUDA` enabled.

```cpp
// File: src/sparsebase/reorder/optimal_reorder.cc
namespace sparsebase::reorder {
template <typename IDType, typename NNZType, typename ValueType>
OptimalReorder(float alpha, float beta) {
    // ...
    this->RegisterFunction(
        {format::CSR<IDType, NNZType, ValueType>::get_id_static()},
        OptimallyOrderCSR);
#ifdef USE_CUDA
    this->RegisterFunction(
        {format::CUDACSR<IDType, NNZType, ValueType>::get_id_static()},
        OptimallyOrderCUDACSR);
#endif
    // ...
  }
} // namespace sparsebase::reorder
```

### 5. Add explicit template instantiations

As mentioned in step 1, SparseBase supports two modes of usage, a compiled mode and a header-only mode. In the compiled mode, classes are pre-compiled using certain data types that the user selects. To add your class to the list of pre-compilable classes, you must add it to the list of classes that will be explicitly instantiated. This list is kept in the JSON file `src/class_instantiation_list.json`.

The aforementioned JSON file consists of an object containing a list called `classes`. To this list, add an object pertaining to your ordering. The object will contain the `OptimalReorder` class definition and the name of the file to which we want the instantiations to be printed. In the case of `OptimalReorder`, we add the following line to the end of the list:


```json
{
  "template": "class OptimalReorder<$id_type, $nnz_type, $value_type>",
  "filename": "optimal_reorder.inc",
  "ifdef": null,
  "folder": null,
  "exceptions": null
}
```
The `"template"` member is the class declaration. The three variables beginning with `$` in the declaration above are placeholders that will be filled with the `IDType`, `NNZType`, and `ValueType` data types the user selects when compiling the library. If a user compiles the library with three `IDType` data types, two `NNZType` data types, and two `ValueType` data types, then the class will be compiled with 3 \* 2 \* 2 = 12 different type combinations.

The `"filename"` member is the name of the file to which these instantiations will be printed, and it matches the name of the header file in which the class is defined, but with the extension `inc`.

For details about the last three parameters, please check the python script `src/generate_explicit_instantiaions.py` which processes the instantiation JSON.

Finally, in the implementation file (`optimal_reorder.cc`), you must include the file which will contain your explicit instantiations. That file will be located in a directory `init` and will have the name given to the JSON object as `filename`. Notice that we only want to use explicit instantiations if the library is not in header-only mode. That is why we must make this `include` statement contingent on `_HEADER_ONLY` not being defined. 
For `OptimalReorder`, we add the following lines:
```c++
// File: src/sparsebase/reorder/optimal_reorder.cc
namespace sparsebase::reorder {
    // ...
}
#ifndef _HEADER_ONLY
#include "init/optimal_reorder.inc"
#endif
```

## Results

Now, you can easily use your reordering as shown in the following example:

```c++
#include "sparsebase/reorder/optimal_reorder.h"
#include "sparsebase/bases/reorder_base.h"
using namespace std;
{
    float alpha = 1.0, beta = 0.5;
    OptimalReorderParams params(alpha, beta);
    unsigned int * order = ReorderBase::Reorder<OptimalReorder>
            (params, some_format_object, {&cpu}, true);
}
```
Or, you could create the parameters object and make the call to `Reorder` in one line by utilizing bracket initialization:
```cpp
#include "sparsebase/reorder/optimal_reorder.h"
#include "sparsebase/bases/reorder_base.h"
using namespace std;
{
    float alpha = 1.0, beta = 0.5;
    unsigned int * order = ReorderBase::Reorder<OptimalReorder>
            ({alpha, beta}, some_format_object, {&cpu}, true);
}
```

Alternatively, you can use the class directly as shown:

```cpp
#include "sparsebase/reorder/optimal_reorder.h"

float alpha = 1.0, beta = 0.5;
sparsebase::reorder::OptimalReorder<int, int, int>
    reorder(alpha, beta);
int *order = reorder.GetOrder(some_format_object, {&cpu}, true);
```

In all the previous examples, if the format type of `some_sparseformat_object` is `CSR`, `CUDACSR`, or any other format that is convertible to either of the two aforementioned formats, then an order will be calculated for it.