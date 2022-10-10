# How to: add a new reordering algorithm

## Objective
This guide demonstrates how to add a new reordering algorithm to the library.
## Overview

Adding new reordering algorithms to SparseBase is very simple. It consists of five steps:

1. create a new class for your ordering. It must inherit from the base `ReorderPreprocessType` class.
2. Inside the class, create a new struct that will contain the hyperparameters needed by your ordering, and initialize 
it in the constructor. 
3. Add implementation functions that will carry out the reordering. 
4. Register the implementation functions in the constructor.
5. Set the converter of the class.
6. Add a constructor for your class a parameters object as parameter.
7. Add explicit template instantiations of your class.

## Steps

In this guide, we will create a new reordering `OptimalReorder`. This reordering has the following properties:

- This reordering requires two float hyperparameters for execution, `alpha` and `beta`.
- It has two implementations. One that operates on a `CSR` format, and another that operates on a `CUDACSR` format, i.e., a `CSR` that is stored on a `CUDA` GPU.

### 1. Create a new class for the ordering

In the header file `src/sparsebase/preprocess/preprocess.h`, add the definition of your class. You should add your class under the namespace `sparsebase::preprocess`. It must be templated on three types `IDType`, `NNZType`, and `ValueType` which define the data types of the `Format` objects it will reorder. Also, it must inherit from the class `ReorderPreprocessType`.


```cpp
// File: src/sparsebase/preprocess/preprocess.h
namespace sparsebase::preprocess {
template <typename IDType, typename NNZType, typename ValueType>
class OptimalReorder : ReorderPreprocessType<IDType, NNZType, ValueType> {};
} // namespace sparsebase::preprocess
```

### 2. Create a struct containing the hyperparameters you need, and initialize them in the constructor

Inside the class, create a new struct inheriting from `PreprocessParams`. Its members will be whichever hyperparameters that your reordering will require. We will call this struct `OptimalReorderParams`. We add `alpha` and `beta` to it.

Additionally, create a `typedef` for your struct as `ParamsType`. This is needed by the `Reorder` function in `ReorderBase`.

Note: if your reordering class does not require any hyperparameters, leave this struct empty.

```cpp
// File: src/sparsebase/preprocess/preprocess.h
namespace sparsebase::preprocess {
template <typename IDType, typename NNZType, typename ValueType>
class OptimalReorder : ReorderPreprocessType<IDType, NNZType, ValueType> {
  struct OptimalReorderParams : PreprocessParams {
    float alpha;
    float beta;
  }
  typedef OptimalReorderParams ParamsType;
};
} // namespace sparsebase::preprocess
```

Inside the constructor of the class, you will take the hyperparameters from the user, add them to an instance of the struct you just created, and set the data member `params_`, which your class inherited from `ReorderPreprocessType`, to the newly added struct.

```cpp
// File: src/sparsebase/preprocess/preprocess.h
namespace sparsebase::preprocess {
template <typename IDType, typename NNZType, typename ValueType>
class OptimalReorder : ReorderPreprocessType<IDType, NNZType, ValueType> {
  // ...
  OptimalReorder(float alpha, float beta) {
    this->params_ =
        unique_ptr<OptimalReorderParams>(new OptimalReorderParams{alpha, beta});
  }
  // ...
};
} // namespace sparsebase::preprocess
```

### 3. Add implementation functions

Add to the class the implementation functions that will carry out the reordering. Each function will be specific for an input `Format` format type. These functions should match the following signature:

```cpp
static IDType* FunctionName(std::vector<format::Format*>, PreprocessParams*) 
```
Not that the functions must also be *static*. This is required to enable the mechanism of choosing the correct implementation function for the input `Format`'s format type.  

The parameters that your function will take are:

1. A vector of pointers at `Format` objects.
2. A pointer at a `PreprocessParams` struct. This pointer is polymorphic, and will be pointing at an instance of the parameters structs created for your ordering. In our case, that would be an `OptimalReorderParams` object. 

Generally, all implementation functions will start with the same three steps:
1. Cast the input `Format` objects to the correct concrete type.
2. Cast the input `PreprocessParams` to the params struct created for this class.
3. Fetch the `Context` of the input `Format` object (this step is not needed for reordering on the CPU, but is necessary when using other architectures, e.g. `CUDA`).

For our example, `OptimalReorder` will have two implementation functions, `OptimallyOrderCSR()` and `OptimallyOrderCUDACSR()`. The former will reorder `CSR` objects on the CPU, and the latter will reorder `CUDACSR` objects, i.e., `CSR` objects stored on a `CUDA` GPU. 

#### 3.a Adding CPU function implementations
We simply add the function according to the aforementiond signature and follow the steps mentioned above, namely casting the format and param objects, and fetching the context of the input.
```cpp
// File: src/sparsebase/preprocess/preprocess.h
namespace sparsebase::preprocess {
template <typename IDType, typename NNZType, typename ValueType>
class OptimalReorder : ReorderPreprocessType<IDType, NNZType, ValueType> {
  //.......
  static IDType *OptimallyOrderCSR(
      std::vector<format::Format<IDType, NNZType, ValueType> *> input_sf,
      PreprocessParams *poly_params) {
    auto csr = input_sf[0]->As<format::CSR<IDType, NNZType, ValueType>>();
    OptimalReorderParams *params =
        static_cast<OptimalReorderParams *>(poly_params);
    context::CPUContext *cpu_context =
        static_cast<context::CPUContext *>(csr->get_context());
    // ... carry out the ordering logic
    return order;
  }
};
} // namespace sparsebase::preprocess
```

#### 3.b Adding `CUDA` GPU function implementations
Adding the implementation for `OptimallyOrderCUDACSR()` follows the same process as `OptimallyReorderCSR()` except for a major difference: it will use a `CUDA` kernel during its execution. This poses a problem since `CUDA` kernels need to be compiled by `nvcc`, not by a pure C++ compiler. The solution to this issue is to add `CUDA` kernels to a seperate `.cu` file, and to use a non-`CUDA` dependent driver functions to interface with them. 

Note that `CUDA` functions related to reordering and preprocessing in general should be added to the file `src/preprocess/cuda/preprocess.cc`. 

For our example, we will add the `CUDA` kernel `OptimalReorderCSROnCUDAGPU()` to the file `src/preprocess/cuda/preprocess.cc` under the namespace `sparsebase::preprocess::cuda`. This function carries out the reordering on the GPU. In addition, we add a driver function that dispatches this kernel under the same namespace.

```cpp
// File: src/sparsebase/preprocess/cuda/preprocess.cc
namespace sparsebase::preprocess::cuda {
template <typename IDType, typename NNZType>
__global__ void OptimalReorderCSROnCUDAGPU(IDType *order, IDType *row_ptr,
                                           NNZType *col, IDType n) {
  // ... carry out ordering
}

template <typename IDType, typename NNZType, typename ValueType>
IDType *
OptimalOrderCSRonCUDAGPUDriver(format::CSR<IDType, NNZType, ValueType> *csr,
                               context::CUDAContext context) {
  // set up context and get pointers from format
  OptimalReorderCSROnCUDAGPU<<<...>>>(...);
  // fetch output from GPU
  return order;
}
} // namespace sparsebase::preprocess::cuda
```

Importantly, we must add the signature of the driver function to the file `src/preprocess/cuda/preprocess.h` in order for the implementation function to be able to use it.

```cpp
// File: src/sparsebase/preprocess/cuda/preprocess.h
namespace sparsebase::preprocess::cuda {
template <typename IDType, typename NNZType, typename ValueType>
IDType *
OptimalOrderCSRonCUDAGPUDriver(format::CUDACSR<IDType, NNZType, ValueType> *csr,
                               context::CUDAContext context);
}
```


Finally, we add the function `OptimallyReorderCUDACSR()` as an implementation inside the `OptimalReorder` class. Note that the function is enclosed in an `#ifdef USE_CUDA` preprocessor block. This will guarantee that it does not get compiled unless compilation of the library with `CUDA` is enabled.

```cpp
// File: src/sparsebase/preprocess/preprocess.h
namespace sparsebase::preprocess {
template <typename IDType, typename NNZType, typename ValueType>
class OptimalReorder : ReorderPreprocessType<IDType, NNZType, ValueType> {
// We do not want the function to be compiled if CUDA isn't enabled
#ifdef USE_CUDA
  static IDType *OptimallyOrderCUDACSR(
      std::vector<format::Format<IDType, NNZType, ValueType> *> input_sf,
      PreprocessParams *poly_params) {
    auto cuda_csr =
        input_sf[0]->As<format::CUDACSR<IDType, NNZType, ValueType>>();
    OptimalReorderParams *params =
        static_cast<OptimalReorderParams *>(poly_params);
    context::CPUContext *cuda_context =
        static_cast<context::CUDAContext *>(cuda_csr->get_context());
    // ...
    // use the driver to call the CUDA kernel
    order = OptimalOrderCSRonCUDAGPUDriver(cuda_csr, *cuda_context);
    // ...
    return order;
  }
#endif
  // .......
};
} // namespace sparsebase::preprocess
```


### 4. Register the implementations you wrote to the correct formats

Inside the constructor, register the functions you made to the correct `Format` type. Note that registering the `CUDACSR` implementation function is surrounded by an `#ifdef USE_CUDA` block to prevent it from being registered if the library is not compiled with `CUDA` enabled.

```cpp
// File: src/sparsebase/preprocess/preprocess.h
namespace sparsebase::preprocess {
template <typename IDType, typename NNZType, typename ValueType>
class OptimalReorder : ReorderPreprocessType<IDType, NNZType, ValueType> {
  // ...
  OptimalReorder(float alpha, float beta) {
    // ...
    this->RegisterFunction(
        {format::CSR<IDType, NNZType, ValueType>::get_format_id_static()},
        OptimallyOrderCSR);
#ifdef USE_CUDA
    this->RegisterFunction(
        {format::CUDACSR<IDType, NNZType, ValueType>::get_format_id_static()},
        OptimallyOrderCUDACSR);
#endif
    // ...
  }
  // ...
};
} // namespace sparsebase::preprocess
```

### 5. Set the converter of the class to the correct type.

While reordering, your class might need to carry out `Format` conversions. For example, if the user tries to use `OptimalReorder` to reorder a `COO`, it needs to be converted to a `CSR` before it can be reordered. 

Each preprocessing class must have an associated converter. The type of the converter depends on the order of the input `Format` objects. In the case of reordering, all the inputs are matrices, therefore, you need to set the converter of your class to the `ConverterOrderTwo` type.

```cpp
// File: src/sparsebase/preprocess/preprocess.h
namespace sparsebase::preprocess {
template <typename IDType, typename NNZType, typename ValueType>
class OptimalReorder : ReorderPreprocessType<IDType, NNZType, ValueType> {
  // ...
  OptimalReorder(float alpha, float beta) {
    // ...
    this->SetConverter(
        utils::converter::ConverterOrderTwo<IDType, NNZType, ValueType>{});
    // ...
  }
  // ...
};
} // namespace sparsebase::preprocess
```

### 6. Add a constructor for your class a parameters object as parameter. 
In order for your class to work with the `Reorder` function in `ReorderBase`, it must have a constructor that takes a 
single parameter of type `ParamsType`, i.e., the type of the hyperparameter struct of your class.

```c++
namespace sparsebase::preprocess {
template <typename IDType, typename NNZType, typename ValueType>
class OptimalReorder : ReorderPreprocessType<IDType, NNZType, ValueType> {
  // ...
  OptimalReorder(OptimalReorderParams params) {
    OptimalReorder(params.alpha, params.beta);
  }
  // ...
};
} // namespace sparsebase::preprocess
```

### 7. Add explicit template instantiations

The functions we have defined so far (with the exception of the `CUDA` kernel and driver functions) have been defined in header files. This means that they will be compiled as they become needed by the user's code, and not at library build-time. However, the library supports a compiled version in which classes are pre-compiled using certain data types that the user selects. To add your class to the list of pre-compilable classes, you must do the following:

1. Move all the implementations from the header file (`src/sparsebase/preprocess/preprocess.h`) to the implementation file (`src/sparsebase/preprocess/preprocess.cc`).
2. Add your class to the list of classes that will be explicitly instantiated by the python script `src/generate_explicit_instantiations.ph`.

Each `.cc` file in the library has a dedicated class inside the python script that handles instantiating its classes. For example, to explicitly instantiate the `OptimalReorder` class, add your class name to the list of classes in the `preprocess.h` file that should be explicitly instantiated. Specifically, in the function `run(self)` inside the class `preprocess_init`, add your class name to the list of classes inside the call to the `print_implementation()` function.

```python
class preprocess_init(explicit_initialization):
  # ...
  def run(self):
    # ...
    print_implementations([..., 'OptimalReorder'], self.out_stream)
```

As for `CUDA` functions, their explicit instantiations should be added to the `run(self)` function of the class `preprocess_cuda_init`. For the exact format, you may follow the existing code in the function.

## Results

Now, you can easily use your reordering as shown in the following example:

```cpp
#include "sparsebase/preprocess/preprocess.h"
using namespace std;
{
    float alpha = 1.0, beta = 0.5;
    unsigned int * order = preprocess::ReorderBase<preprocess::OptimalReorder>
            ({alpha, beta}, some_format_object, {&cpu});
}
```

Or, you can use the class directly as shown:

```cpp
#include "sparsebase/preprocess/preprocess.h"

float alpha = 1.0, beta = 0.5;
sparsebase::preprocess::OptimalReorder<unsigned int, unsigned int, unsigned int>
    reorder(alpha, beta);
unsigned int *order = reorder.GetOrder(some_format_object, {&cpu});
```

If the format type of `some_sparseformat_object` is `CSR`, `CUDACSR`, or any other format that is convertible to the two aforementioned formats, then an order will be calculated for it.