# How to: add a new feature

## Overview

Adding new features to SparseBase is simple. It consists of six steps:

1. Create a new class for your feature. It must inherit from the base `FeaturePreprocessType` class.
2. `FeaturePreprocessType` implements the `ExtractableType` interface thus all virtual functions defined by `ExtractableType` must be implemented. `ExtractableType` provides a template for all the functionality needed for feature extraction.
3. Inside the class, create a new struct that will contain the parameters needed by your feature, and initialize it in the constructor. Note: This step is only needed if your feature requires parameters to work with.
4. Add implementation functions that will carry out the feature calculation.
5. Register the implementation functions in the constructor.
6. Add explicit template instantiations of your class to the python script.

## Example

The following example demonstrates the process of creating a new feature `Feature`.

- This feature requires two float parameters for execution, `alpha` and `beta`.
- It has two implementations. One that operates on a `CSR` format, and another that operates on a `COO` format.

### 1. Create a new class for the feature

In the header file `sparsebase/include/sparse_preprocess.hpp`, add the definition of your class. It must be templated on four types `IDType`, `NNZType`, `ValueType`, and `FeatureType` which define the data types of the `Format` and the return type of the feature class.  Finally, as mentioned above, `Feature` must inherit from the class `FeaturePreprocessType`.

```cpp
template <typename IDType, typename NNZType, typename ValueType, typename FeatureType>
class Feature : FeaturePreprocessType<FeatureType> {

};
```

### 2. Create a struct containing the parameters you need, and initialize them in the constructor

Inside the class, create a new struct inheriting from `PreprocessParams`. Its members will be whichever parameters that your feature will require. We will call this struct `FeatureParams`. We add `alpha` and `beta` to it. If your feature do not require additional parameters you can skip this step.
Furthermore, create an instance of the struct you just defined and also create a `std::unordered_map<std::type_index, PreprocessParams>` that holds the parameters of features separately (only applicable if the class implements more than one feature simultaneously). This is especially important for the functionalities provided by the `feature` namespace.

```cpp
struct FeatureParams : Preprocess {
    float alpha;
    float beta;
}

template <typename IDType, typename NNZType, typename ValueType, typename FeatureType>
class Feature : FeaturePreprocessType<FeatureType> {
    
};
```

Inside the constructor, you will take the parameters from the user, add them to an instance of the struct you just created, and set the data member `params_`, which your class inherited from `ExtractableType`, to the newly added struct. 
If your feature do not require additional parameters you can always use `PreprocessParams` to initialize `params_`.
Furthermore, fill the unordered_map `pmap_` which is also inherited from `ExtractableType`. 

```cpp
template <typename IDType, typename NNZType, typename ValueType, typename FeatureType>
class Feature : FeaturePreprocessType<FeatureType> {
	// ...
	Feature(float alpha, float beta){
		this->params_ = std::make_shared<FeatureParams>(alpha, beta);
        pmap_.insert(get_id_static(), this->params_);
	// ...
};
```

### 3. Implement all virtual functions that are defined by ExtractableType interface

Some of the `virtual` functions are implemented in `FeaturePreprocessType`, however not all. Some need to be implemented by the developer:

```cpp
template <typename IDType, typename NNZType, typename ValueType, typename FeatureType>
class Feature : FeaturePreprocessType<FeatureType> {
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

Add implementation functions that will carry out the computations for the feature to the file `sparsebase/src/sparse_preprocess.cc`. Each function will be specific for an input `Format`. These functions should match the function signature provided in `FunctionMatcherMixin`:

```cpp
using PreprocessFunction = ReturnType (*)(std::vector<format::Format *>, PreprocessParams *);
```
Not that the functions must also be *static*. This is required to enable the mechanism of choosing the correct implementation function for the input `Format`.

Your function takes the following parameters:

1. A vector of pointers at `Format` objects.
2. A pointer at a `PreprocessParams` struct. This pointer is polymorphic, and will be pointing at an instance of the parameters structs created for your feature.

For our example, we add two functions, `FeatureCSR()` and `FeatureCOO()`:

```cpp
template <typename IDType, typename NNZType, typename ValueType, typename FeatureType>
class Feature : FeaturePreprocessType<FeatureType> {
	//.......
	static FeatureType* FeatureCSR(std::vector<Format<IDType, NNZType, ValueType>*> input_sf, PreprocessParams* params){
		auto csr = static_cast<sparsebase::CSR<IDType, NumNonZerosType, ValueType>(input_sf[0]);
		FeatureParams* params = static_cast<OptimalReorderParams*>(params);
		// ... carry out feature extraction
		return feature;
	}

	static FeatureType* FeatureCOO(std::vector<Format<IDType, NNZType, ValueType>*> input_sf, PreprocessParams* params){
		auto coo = static_cast<sparsebase::COO<IDType, NNZType, ValueType>(input_sf[0]);
		FeatureParams* params = static_cast<FeatureParams*>(params);
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
class Feature : FeaturePreprocessType<FeatureType> {
	// ...
	Feature(float alpha, float beta){
		// ...
		this->RegisterFunction({format::CSR<IDType, NNZType, ValueType>::get_id_static()}, FeatureCSR);
		this->RegisterFunction({format::COO<IDType, NNZType, ValueType>::get_id_static()}, FeatureCOO);
		// ...
	// ...
};
```

### 6. Add explicit template instantiations

The functions we have defined so far have been created in header files. This means that they will be compiled as they become needed by the user's code, and not at library build-time. However, sparsebase supports a compiled version in which classes are pre-compiled using certain data types that the user selects. To add your class to the list of pre-compilable classes, you must do the following:

1. Move all the implementations from the header file (`src/sparsebase/preprocess/preprocess.h`) to the implementation file (`src/sparsebase/preprocess/preprocess.cc`).
2. Add your class to the list of classes that will be explicitly instantiated by the python script `src/generate_explicit_instantiations.py`.

Step two is much simpler than it sounds. To the file `src/class_instantiation_list.json`, add a single entry containing the `Feature` class definition and the name of the file to which the instantiations are to be printed. In the case of `Feature`, add the following entry to the aformentioned file:

```json
{
  "template": "class Feature<$id_type, $nnz_type, $value_type, $feature_type>",
  "filename": "feature.inc",
  "ifdef": null,
  "folder": null,
  "exceptions": null
}
```
The `template` field is the class declaration. The four variables beginning with `$` in the declaration above are placeholders that will be filled with the `IDType`, `NNZType`, `ValueType`, and `FeatureType` data types the user selects when compiling the library. If a user compiles the library with three `IDType` data types, two `NNZType` data types, two `ValueType` data types, and a single `FeatureType` then the class will be compiled with 3 * 2 * 2 * 1 = 12 different type combinations.

The `filename` field is the name of the file to which these instantiations will be printed, and it matches the name of the header file in which the class is defined.

### Result

Now, you can easily extract your feature as the following:

```cpp
#include "sparsebase/sparse_preprocess.h"
#include "sparsebase/sparse_feature.h"
 
float alpha= 1.0, beta = 0.5;
sparsebase::feature::Extractor<vertex_type, edge_type, value_type, feature_type> engine;
engine.Add(feature::Feature(sparsebase::preprocess::Feature{alpha, beta}));
auto raws = engine.Extract(coo);
```