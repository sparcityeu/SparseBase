# How to: add a new feature

## Overview

Adding new features to SparseBase is simple. It consists of six steps:

1. Create a new class for your feature. It must inherit from the base `FeaturePreprocessType` class.
2. `FeaturePreprocessType` implements the `ExtractableType` interface thus all virtual functions defined by `ExtractableType` must be implemented. `ExtractableType` provides a template for all the functionality needed for feature extraction.
3. Create a new struct that will contain the parameters needed by your feature, and initialize it in the constructor. Note: This step is only needed if your feature requires parameters to work with.
4. Add implementation functions that will carry out the feature calculation.
5. Register the implementation functions in the constructor.
6. Add explicit template instantiations of your class to the JSON file.

## Example

The following example demonstrates the process of creating a new feature `FeatureX`.

- This feature requires two float parameters for execution, `alpha` and `beta`.
- It has two implementations. One that operates on a `CSR` format, and another that operates on a `COO` format.

### 1. Create a new class for the feature

The class will be split into a header file and an implementation file. Both files will be stored in the directory `src/sparsebase/feature` and will have the same name as the class but in [snake case](https://en.wikipedia.org/wiki/Snake_case). For `FeatureX`, the files will be `feature_x.h` and `feature_x.cc`. At the top of the header file, include the following headers:
```c++
// Flags containing compilation flags, e.g. USE_CUDA
#include "sparsebase/config.h"
// Definition of base feature extraction class
#include "sparsebase/feature/feature_preprocess_type.h"
// Definition of parameters struct
#include "sparsebase/utils/parameterizable.h"
```

And at the top of the implementation file, include the created header.
```c++
#include "sparasebase/reorder/feature_x.h"
```
In the header file, add the definition of your class. It must be templated on four types `IDType`, `NNZType`, `ValueType`, and `FeatureType` which define the data types of the `Format` and the return type of the feature class. Finally, as mentioned above, `FeatureX` must inherit from the class `FeaturePreprocessType`.

```cpp
namespace sparsebase::feature{
template <typename IDType, typename NNZType, typename ValueType, typename FeatureType>
class FeatureX : FeaturePreprocessType<FeatureType> {

};
}
```
For now, the definition file will be empty.

Finally, we must include the definition file inside the header file to enable header-only usage of the class. We make this inclusion conditional on the preprocessor directive `_HEADER_ONLY`. We make this addition to `feature.h` as follows:
```c++
// File: src/sparsebase/feature/feature_x.h
namespace sparsebase::feature {
template <typename IDType, typename NNZType, typename ValueType, typename FeatureType>
class FeatureX : FeaturePreprocessType<FeatureType> {
} // namespace sparsebase::feature

#ifdef _HEADER_ONLY
#include "sparsebase/feature/feature.cc"
#endif
```

Notice that the `include` statement is added _outside_ the `sparsebase::feature` namespace.

> **Compiled vs. header-only**. In header-only mode, the user includes the code they want to use in their own code and compiles it as needed. In the compiled mode, the library classes and functions are precompiled into a static library and the user links to them at compile-time.

### 2. Create a struct containing the parameters you need, and initialize them in the constructor

In the header file created in step 1, create a new struct inheriting from `utils::Parameters`. Its members will be whichever hyperparameters your feature will require. The naming convention for these structs is the name of the reordering class suffixed with `Params`. For our class, that would be `FeatureXParams`. We add `alpha` and `beta` to it. You may also add custom constructors for your parameter struct. If your feature do not require additional parameters you can skip this step.

Furthermore, create an instance of the struct you just defined and also create a `std::unordered_map<std::type_index, utils::Parameters>` that holds the parameters of features separately (only applicable if the class implements more than one feature simultaneously). This is especially important for the functionalities provided by the `feature` namespace.

```cpp
struct FeatureXParams : utils::Parameters {
    float alpha;
    float beta;
}

template <typename IDType, typename NNZType, typename ValueType, typename FeatureType>
class FeatureX : FeaturePreprocessType<FeatureType> {
    
};
```

Inside the constructor, you will take the parameters from the user, add them to an instance of the struct you just created, and set the data member `params_`, which your class inherited from `ExtractableType`, to the newly added struct. 
If your feature do not require additional parameters you can always use `utils::Parameters` to initialize `params_`.
Furthermore, fill the unordered_map `pmap_` which is also inherited from `ExtractableType`. 

```cpp
template <typename IDType, typename NNZType, typename ValueType, typename FeatureType>
class FeatureX : FeaturePreprocessType<FeatureType> {
	// ...
	FeatureX(float alpha, float beta){
		this->params_ = std::make_shared<FeatureXParams>(alpha, beta);
        pmap_.insert(get_id_static(), this->params_);
	// ...
};
```

### 3. Implement all virtual functions that are defined by ExtractableType interface

Some of the `virtual` functions are implemented in `FeaturePreprocessType`, however not all. Some need to be implemented by the developer:

```cpp
template <typename IDType, typename NNZType, typename ValueType, typename FeatureType>
class FeatureX : FeaturePreprocessType<FeatureType> {
	// ...
    virtual std::unordered_map<std::type_index, std::any> Extract(format::Format * format);
    virtual std::vector<std::type_index> get_sub_ids();
    virtual std::vector<ExtractableType*> get_subs();
	// ...
};
```
The Extract function is used by the `feature::Extractor` to call the implementation functions that are registered in the constructor. `feature::Extractor::Extract` function will first try to merge feature implementations, using the `ClassMatcherMixin` and then call the `Extract` function for each chosen class. Therefore, in this function the user must either call `FunctionMatcherMixin::Execute` directly or make a call to such a function. 

`get_subs()` and `get_sub_ids()` functions are used by the `feature::Extractor` to break apart and merge feature computations. The latter is used to return instances of all the features that are implemented by this class and the former return the ids of those features. For the classes that implement a single feature implementing the above functions is straightforward since the `get_subs()` and `get_sub_ids()` functions return info related to the class itself. For classes that implement multiple features simultaneously, make sure that these functions return the correct information in the correct order. The `type_index` vectors must be sorted for the `feature::Extractor` to work properly.

### 4. Add implementations for the feature class

Add implementation functions that will carry out the computations of the feature to the `FeatureX` class. Each function will be specific for an input `Format`. These functions should match the function signature provided in `FunctionMatcherMixin`:

```cpp
using PreprocessFunction = ReturnType (*)(std::vector<format::Format *>, utils::Parameters *);
```
Not that the functions must also be *static*. This is required to enable the mechanism of choosing the correct implementation function for the input `Format`.

Your function takes the following parameters:

1. A vector of pointers at `Format` objects.
2. A pointer at a `utils::Parameters` struct. This pointer is polymorphic, and will be pointing at an instance of the parameters structs created for your feature.

For our example, we add two functions, `FeatureCSR()` and `FeatureCOO()`:

```cpp
#include "sparsebase/format/csr.h"
#include "sparsebase/format/coo.h"

template <typename IDType, typename NNZType, typename ValueType, typename FeatureType>
class FeatureX : FeaturePreprocessType<FeatureType> {
	//.......
	static FeatureType* FeatureCSR(std::vector<Format*> input_sf, utils::Parameters* params){
		auto csr = static_cast<sparsebase::CSR<IDType, NNZType, ValueType>(input_sf[0]);
		FeatureXParams* params = static_cast<FeatureXParams*>(params);
		// ... carry out feature extraction
		return feature;
	}

	static FeatureType* FeatureCOO(std::vector<Format*> input_sf, utils::Parameters* params){
		auto coo = static_cast<sparsebase::COO<IDType, NNZType, ValueType>(input_sf[0]);
		FeatureXParams* params = static_cast<FeatureXParams*>(params);
		// ... carry out feature extraction
		return feature;
	}
	// .......
};
```

### 5. Register the implementations you wrote to the correct formats

Inside the constructor, register the functions you made to the correct `Format`.

```cpp
template <typename IDType, typename NNZType, typename ValueType, typename FeatureType>
class FeatureX : FeaturePreprocessType<FeatureType> {
	// ...
	FeatureX(float alpha, float beta){
		// ...
		this->RegisterFunction({format::CSR<IDType, NNZType, ValueType>::get_id_static()}, FeatureCSR);
		this->RegisterFunction({format::COO<IDType, NNZType, ValueType>::get_id_static()}, FeatureCOO);
		// ...
	// ...
};
```

### 6. Add explicit template instantiations

The functions we have defined so far have been created in header files. This means that they will be compiled as they become needed by the user's code, and not at library build-time. However, sparsebase supports a compiled version in which classes are pre-compiled using certain data types that the user selects. To add your class to the list of pre-compilable classes, you must do the following:

1. Move all the implementations from the header file (`src/sparsebase/feature/feature_x.h`) to the implementation file (`src/sparsebase/feature/feature_x.cc`).
2. Add your class to the list of classes that will be explicitly instantiated by the python script `src/generate_explicit_instantiations.py`.

Step two is much simpler than it sounds. To the file `src/class_instantiation_list.json`, add a single entry containing the `FeatureX` class definition and the name of the file to which the instantiations are to be printed. In the case of `FeatureX`, add the following entry to the aformentioned file:

```json
{
  "template": "class FeatureX<$id_type, $nnz_type, $value_type, $float_type>",
  "filename": "feature_x.inc",
  "ifdef": null,
  "folder": null,
  "exceptions": null
}
```
The `template` field is the class declaration. The four variables beginning with `$` in the declaration above are placeholders that will be filled with the `IDType`, `NNZType`, `ValueType`, and `FeatureType` data types the user selects when compiling the library. If a user compiles the library with three `IDType` data types, two `NNZType` data types, two `ValueType` data types, and a single `FeatureType` then the class will be compiled with 3 \* 2 \* 2 \* 1 = 12 different type combinations.

The `filename` field is the name of the file to which these instantiations will be printed, and it matches the name of the header file in which the class is defined.

Finally, in the implementation file (`feature_x.cc`), you must include the file which will contain your explicit instantiations. That file will be located in a directory `init` and will have the name given to the JSON object as `filename`. Notice that we only want to use explicit instantiations if the library is not in header-only mode. That is why we must make this `include` statement contingent on `_HEADER_ONLY` not being defined. 
For `FeatureX`, we add the following lines:
```c++
// File: src/sparsebase/feature/feature_x.cc
namespace sparsebase::feature {
    // ...
}
#ifndef _HEADER_ONLY
#include "init/feature_x.inc"
#endif
```

### Result

Now, you can easily extract your feature as the following:

```cpp
#include "sparsebase/feature/feature_extractor.h"
#include "sparsebase/feature/feature_x.h"
 
float alpha= 1.0, beta = 0.5;
sparsebase::feature::Extractor<vertex_type, edge_type, value_type, feature_type> engine;
engine.Add(feature::FeatureX(sparsebase::preprocess::FeatureX{alpha, beta}));
auto raws = engine.Extract(coo);
```