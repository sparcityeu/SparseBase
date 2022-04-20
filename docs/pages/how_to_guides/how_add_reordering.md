# How to: add a new reordering algorithm

## Objective
This guide demonstrates how to add a new reordering algorithm to the library.
## Overview

Adding new reordering algorithms to SparseBase is very simple. It consists of five steps:

1. create a new class for your ordering. It must inherit from the base `ReorderPreprocessType` class.
2. Inside the class, create a new struct that will contain the hyperparameters needed by your ordering, and initialize it in the constructor. Note: This step is only needed if your reordering requires hyperparameters.
3. Add implementation functions that will carry out the reordering. 
4. Register the implementation functions in the constructor.
5. Add explicit template instantiations of your class.

## Steps

In this guide, we will create a new reordering `OptimalReorder`. This reordering has the following properties:

- This reordering requires two float hyperparameters for execution, `alpha` and `beta`.
- It has two implementations. One that operates on a `CSR` format, and another that operates on a `COO` format.

### 1. Create a new class for the ordering

In the header file `src/sparsebase/preprocess/preprocess.h`, add the definition of your class. You should add your class under the namespace `sparsebase::preprocess`. It must be templated on three types `IDType`, `NNZType`, and `ValueType` which define the data types of the `Format` objects it will reorder. Also, it must inherit from the class `ReorderPreprocessType`.

```cpp
template <typename IDType, typename NNZType, typename ValueType>
class OptimalReorder : ReorderPreprocessType<IDType, NNZType, ValueType> {

};
```

### 2. Create a struct containing the hyperparameters you need, and initialize them in the constructor

Inside the class, create a new struct inheriting from `PreprocessParams`. Its members will be whichever hyperparameters that your reordering will require. We will call this struct `OptimalReorderParams`. We add `alpha` and `beta` to it.

```cpp
template <typename IDType, typename NNZType, typename ValueType>
class OptimalReorder : ReorderPreprocessType<IDType, NNZType, ValueType> {
  struct OptimalReorderParams : PreprocessParams {
    float alpha;
    float beta;
  }
};
```

Inside the constructor of the class, you will take the hyperparameters from the user, add them to an instance of the struct you just created, and set the data member `params_`, which your class inherited from `ReorderPreprocessType`, to the newly added struct.

```cpp
template <typename IDType, typename NNZType, typename ValueType>
class OptimalReorder : ReorderPreprocessType<IDType, NNZType, ValueType> {
  // ...
  OptimalReorder(float alpha, float beta){
    this->params_ = unique_ptr<OptimalReorderParams>(new OptimalReorderParams{alpha, beta});
  }
  // ...
};
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

For our example, we add two functions, `OptimallyOrderCSR()` and `OptimallyOrderCOO()`. 

Generally, all implementation functions will start with the same three steps:
1. Cast the input `Format` objects to the correct concrete type.
2. Cast the input `PreprocessParams` to the params struct created for this class.
3. Fetch the `Context` of the input `Format` object (this step is not needed for reordering on the CPU, but is necessary when using other architectures, e.g. `CUDA`).


```cpp
template <typename IDType, typename NNZType, typename ValueType>
class OptimalReorder : ReorderPreprocessType<IDType, NNZType, ValueType> {
  //.......
  static IDType* OptimallyOrderCSR(std::vector<format::Format<IDType, NNZType, ValueType>*> input_sf, PreprocessParams* poly_params){
    auto csr = input_sf[0]->As<format::CSR<IDType, NNZType, ValueType>>();
      OptimalReorderParams* params = static_cast<OptimalReorderParams*>(poly_params);
    context::CPUContext *cpu_context =
      static_cast<context::CPUContext *>(csr->get_context());
    // ... carry out the ordering logic
    return order;
  }
  
  static IDType* OptimallyOrderCOO(std::vector<format::Format<IDType, NNZType, ValueType>*> input_sf, PreprocessParams* poly_params){
    auto coo = input_sf[0]->As<format::COO<IDType, NNZType, ValueType>>();
    OptimalReorderParams* params = static_cast<OptimalReorderParams*>(poly_params);
    context::CPUContext *cpu_context =
      static_cast<context::CPUContext *>(coo->get_context());
    // ... carry out the ordering logic
    return order;
  }
  // .......
};
```

### 4. Register the implementations you wrote to the correct formats

Inside the constructor, register the functions you made to the correct `Format` type. 

```cpp
template <typename IDType, typename NNZType, typename ValueType>
class OptimalReorder : ReorderPreprocessType<IDType, NNZType, ValueType> {
  // ...
  OptimalReorder(float alpha, float beta){
    // ...
    this->RegisterFunction({format::CSR<IDType, NNZType, ValueType>::get_format_id_static()}, optimally_order_csr);
    this->RegisterFunction({format::COO<IDType, NNZType, ValueType>::get_format_id_static()}, optimally_order_coo);
    // ...
  }
  // ...
};
```

### 5. Set the converter of the class to the correct type.

While reordering, your class might need to carry out `Format` conversions. For example, if the user tries to use `OptimalReorder` to reorder a `CUDACSR`, it needs to be converted to a `CSR` before it can be reordered. 

Each preprocessing class must have an associated converter. The type of the converter depends on the order of the input `Format` objects. In the case of reordering, all the inputs are matrices, therefore, you need to set the converter of your class to the `ConverterOrderTwo` type.

```cpp
template <typename IDType, typename NNZType, typename ValueType>
class OptimalReorder : ReorderPreprocessType<IDType, NNZType, ValueType> {
  // ...
  OptimalReorder(float alpha, float beta){
    // ...
    this->SetConverter(
      utils::converter::ConverterOrderTwo<IDType, NNZType, ValueType>{});
    // ...
  }
  // ...
};
```

### 6. Add explicit template instantiations

Since this library can be used as a compiled library, you must add explicit instantiations of your class in order for it to be compiled. This can be done easily using the python script `src/generate_explicit_instantiations.py`. This script automatically generate explicit instantiations of classes in the library at build system-generation time. The types that will be used when generating explicit instantiations are a compilation option that users can specify. 

All you need to do is add your class name to the list of classes in the `preprocess.h` file that should be explicitly instantiated. To do so, in the function `run(self)` inside the class `preprocess_init`, add your class name to the list of classes inside the call to the `print_implementation()` function.

```python
class preprocess_init(explicit_initialization):
  # ...
  def run(self):
    # ...
    print_implementations([..., 'OptimalReorder'], self.out_stream)
```

# Results

Now, you can easily use your reordering like the following example:

```cpp
#include "sparsebase/preprocess/preprocess.h"
 
float alpha= 1.0, beta = 0.5;
sparsebase::preprocess::OptimalReorder<unsigned int, unsigned int, unsigned int> reorder(alpha, beta);
unsigned int * order = reorder.GetOrder(some_format_object);
```

If the format type of `some_sparseformat_object` is `CSR`, `COO`, or any other format that is convertible to the two aforementioned formats, then an order will be calculated for it.