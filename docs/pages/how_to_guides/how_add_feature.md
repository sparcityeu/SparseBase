# How to: add a new feature

## Overview

Adding new features to SparseBase is very simple. It consists of six steps:

1. create a new class for your feature. It must inherit from the base `FeaturePreprocessType` class.
2. `FeaturePreprocessType` implements the `ExtractableType` interface thus all virtual functions defined by `ExtractableType` must be implemented. `ExtractableType` provides a seamless interface for feature extraction.
3. Inside the class, create a new struct that will contain the parameters needed by your feature, and initialize it in the constructor. Note: This step is only needed if your feature requires parameters to work with.
4. Add implementation functions that will carry out the feature.
5. Register the implementation functions in the constructor.
6. Add explicit template instantiations of your class to the python script.

## Example

The following example demonstrates the process of creating a new feature `Fearture`.

- This feature requires two float parameters for execution, `alpha` and `beta`.
- It has two implementations. One that operates on a `CSR` format, and another that operates on a `COO` format.

### 1. Create a new class for the feature

In the header file `sparsebase/include/sparse_preprocess.hpp`, add the definition of your class. It must be templated on four types `IDType`, `NNZType`, `ValueType`, and `FeatureType` which define the data types of the `Format` and the return type of the feature class. `Feature` must inherit from the class `FeaturePreprocessType`.

```cpp
template <typename IDType, typename NNZType, typename ValueType, typename FeatureType>
class Feature : FeaturePreprocessType<IDType, NNZType, ValueType, FeatureType> {

};
```

### 2. Create a struct containing the parameters you need, and initialize them in the constructor

Inside the class, create a new struct inheriting from `PreprocessParams`. Its members will be whichever parameters that your feature will require. We will call this struct `FeatureParams`. We add `alpha` and `beta` to it.
Furthermore, create an instance of the struct you just defined and also create a `std::unordered_map<std::type_index, PreprocessParams>` that holds the parameters of features separately (only applicable if the class implements more than one feature simultaneously). This is especially important for the functionalities provided by the `feature` namespace.

```cpp
template <typename IDType, typename NNZType, typename ValueType, typename FeatureType>
class Feature : FeaturePreprocessType<IDType, NNZType, ValueType, FeatureType> {
	struct FeatureParams : Preprocess {
		float alpha;
		float beta;
	}
};
```

Inside the constructor, you will take the parameters from the user, add them to an instance of the struct you just created, and set the data member `params_`, which your class inherited from `ExtractableType`, to the newly added struct. 
Furthermore, fill the unordered_map `pmap_` which is also inherited from `ExtractableType`. 

```cpp
template <typename IDType, typename NNZType, typename ValueType, typename FeatureType>
class Feature : FeaturePreprocessType<IDType, NNZType, ValueType, FeatureType> {
	// ...
	Feature(float alpha, float beta){
		this->params_ = shared_ptr<FeatureParams>(new FeatureParams{alpha, beta});
        pmap_.insert(get_feature_id_static(), this->params_);
	// ...
};
```

### 3. Implement all virtual functions that are implemented by ExtractableType interface

Some of the `virtual` functions are implemented in `FeaturePreprocessType`, however not all. Some need to be implemented by the developer:

```cpp
template <typename IDType, typename NNZType, typename ValueType, typename FeatureType>
class Feature : FeaturePreprocessType<IDType, NNZType, ValueType, FeatureType> {
	// ...
    virtual std::unordered_map<std::type_index, std::any> Extract(format::Format * format);
    virtual std::vector<std::type_index> get_sub_ids();
    virtual std::vector<ExtractableType*> get_subs();
	// ...
};
```
The Extract function is used by the `feature::Extractor` to call the implementation functions that are registered in the constructor. `feature::Extractor::Extract` function will first try to merge feature implementations, using the `ClassMatcherMixin` and then call the `Extract` function for each chosen class. Therefore, in this function the user must either call `FunctionMatcherMixin::Execute` directly or make a call to such a function. 

`get_subs()` and `get_sub_ids()` functions are used by the `feature::Extractor` to break apart and merge feature computations. The latter is used to return instances of all the features that are implemented by this class and the latter return the ids of those features. These are needed to find the most optimal implementation for the features registered by the user for extraction. For the classes that implement a single feature implementing the above functions is very straightforward since the `get_subs()` and `get_sub_ids()` functions return info related to the class itself. For classes that implement multiple features simultaneously make sure that these functions return the correct information in the correct order. The `type_index` vectors must be sorted for the `feature::Extractor` to work properly.

### 4. Add implementations for the feature class

Add implementation functions that will carry out the computations for the feature to the file `sparsebase/src/sparse_preprocess.cc`. Each function will be specific for an input `Format`. These functions should match the function signature provided in `FunctionMatcherMixin`:

```cpp
using PreprocessFunction = ReturnType (*)(std::vector<format::Format *>, PreprocessParams *);
```
Not that the functions must also be *static*. This is required to enable the mechanism of choosing the correct implementation function for the input `Format`.

The parameters that your function will take are:

1. A vector of pointers at `Format` objects.
2. A pointer at a `PreprocessParams` struct. This pointer is polymorphic, and will be pointing at an instance of the parameters structs created for your feature.

For our example, we add two functions, `FeatureCSR()` and `FeatureCOO()`:

```cpp
template <typename IDType, typename NNZType, typename ValueType, typename FeatureType>
class Feature : FeaturePreprocessType<IDType, NNZType, ValueType, FeatureType> {
	//.......
	static FeatureType* FeatureCSR(std::vector<SparseFormat<IDType, NNZType, ValueType>*> input_sf, PreprocessParams* params){
		auto csr = static_cast<sparsebase::CSR<IDType, NumNonZerosType, ValueType>(input_sf[0]);
		FeatureParams* params = static_cast<OptimalReorderParams*>(params);
		// ... carry out feature extraction
		return feature;
	}

	static FeatureType* FeatureCOO(std::vector<SparseFormat<IDType, NNZType, ValueType>*> input_sf, PreprocessParams* params){
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
class Feature : FeaturePreprocessType<IDType, NNZType, ValueType, FeatureType> {
	// ...
	Feature(float alpha, float beta){
		// ...
		this->RegisterFunction({kCSRFormat}, FeatureCSR);
		this->RegisterFunction({kCOOFormat}, FeatureCOO);
		// ...
	// ...
};
```

### 6. Add explicit template instantiations

For the compiled version of the library, you must add explicit instantiations of your class depending on the data types you are going to use for `IDType`, `NNZType`, `ValueType`, and `FeatureType`.

```cpp
template class Feature<unsigned int, unsigned int, float, double>;
```

For this purpose you must edit `generate_explicit_instantiations.py` accordingly.

Now, you can easily extract your feature as following:

```cpp
#include "sparsebase/sparse_preprocess.h"
#include "sparsebase/sparse_feature.h"
 
float alpha= 1.0, beta = 0.5;
sparsebase::feature::Extractor<vertex_type, edge_type, value_type, feature_type> engine;
engine.Add(feature::Feature(sparsebase::preprocess::Feature{alpha, beta}));
auto raws = engine.Extract(coo);
```