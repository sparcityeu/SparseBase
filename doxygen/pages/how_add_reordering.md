# How to: add a new reordering algorithm

# Overview

Adding new reordering algorithms to SparseBase is very simple. It consists of five steps:

1. create a new class for your ordering. It must inherit from the base `ReorderPreprocessType` class.
2. Inside the class, create a new struct that will contain the hyperparameters needed by your ordering, and initialize it in the constructor. Note: This step is only needed if your reordering requires hyperparameters.
3. Add implementation functions that will carry out the reordering. 
4. Register the implementation functions in the constructor.
5. Add explicit template instantiations of your class.

# Example

The following example demonstrates the process of creating a new reordering `OptimalReorder`.

- This reordering requires two float hyperparameters for execution, `alpha` and `beta`.
- It has two implementations. One that operates on a `CSR` format, and another that operates on a `COO` format.

### 1. Create a new class for the ordering

In the header file `sparsebase/include/sparse_preprocess.hpp`, add the definition of your class. It must be templated on three types `ID`, `NumNonZeros`, `Value` which define the data types of the `SparseFormat` objects it will reorder. Also, it must inherit from the class `ReorderPreprocessType`.

```cpp
template <typename ID, typename NumNonZeros, typename Value>
class OptimalReorder : ReorderPreprocessType<ID, NumNonZeros, Value> {

};
```

### 2. Create a struct containing the hyperparameters you need, and initialize them in the constructor

Inside the class, create a new struct inheriting from `ReorderParams`. Its members will be whichever hyperparameters that your reordering will require. We will call this struct `OptimalReorderParams`. We add `alpha` and `beta` to it.

```cpp
template <typename ID, typename NumNonZeros, typename Value>
class OptimalReorder : ReorderPreprocessType<ID, NumNonZeros, Value> {
	struct OptimalReorderParams : ReorderParams {
		float alpha;
		float beta;
	}
};
```

Inside the constructor, you will take the hyperparameters from the user, add them to an instance of the struct you just created, and set the data member `_params`, which your class inherited from `ReorderPreprocessType`, to the newly added struct.

```cpp
template <typename ID, typename NumNonZeros, typename Value>
class OptimalReorder : ReorderPreprocessType<ID, NumNonZeros, Value> {
	// ...
	OptimalReorder(float alpha, float beta){
		this->_params = unique_ptr<OptimalReorderParams>(new OptimalReorderParams{alpha, beta});

	// ...
};
```

### 3. Add implementations for optimal reordering

Add implementation functions that will carry out the reordering. Each function will be specific for an input `SparseFormat` Format. These functions should match the `ReorderFunction` signature:

```cpp
static ID* function_name(std::vector<SparseFormat<ID, NumNonZeros, Value>*>, ReorderParams*) 
```
Not that the functions must also be *static*. This is required to enable the mechanism of choosing the correct implementation function for the input `SparseFormat` object's Format.  

The parameters that your function will take are:

1. A vector of pointers at `SparseFormat` objects.
2. A pointer at a `ReorderParams` struct. This pointer is polymorphic, and will be pointing at an instance of the parameters structs created for your ordering. 

For our example, we add two functions, `optimally_order_csr()` and `optimally_order_coo()`. Notice how we use the inputs to extract the `SparseFormat` object we will reorder, and the `OptimalReorderParams` object storing the user's parameters:

```cpp
template <typename ID, typename NumNonZeros, typename Value>
class OptimalReorder : ReorderPreprocessType<ID, NumNonZeros, Value> {
	//.......
	static ID* optimally_order_csr(std::vector<SparseFormat<ID, NumNonZeros, Value>*> input_sf, ReorderParams* poly_params){
		CSR<ID, NumNonZeros, Value> csr = static_cast<CSR<ID, NumNonZeros, Value>(input_sf[0]);
		OptimalReorderParams* params = static_cast<OptimalReorderParams*>(poly_params);
		// ... carry out the ordering logic
		return order;
	}

	static ID* optimally_order_coo(std::vector<SparseFormat<ID, NumNonZeros, Value>*> input_sf, ReorderParams* poly_params){
		COO<ID, NumNonZeros, Value> coo = static_cast<COO<ID, NumNonZeros, Value>(input_sf[0]);
		OptimalReorderParams* params = static_cast<OptimalReorderParams*>(poly_params);
		// ... carry out the ordering logic
		return order;
	}
	// .......
};
```

### 4. Register the implementations you wrote to the correct formats

Inside the constructor, register the functions you made to the correct `Format`. 

```cpp
template <typename ID, typename NumNonZeros, typename Value>
class OptimalReorder : ReorderPreprocessType<ID, NumNonZeros, Value> {
	// ...
	OptimalReorder(float alpha, float beta){
		// ...
		this->register_function({CSR_f}, optimally_order_csr);
		this->register_function({COO_f}, optimally_order_coo);
		// ...
	// ...
};
```

### 5. Add explicit template instantiations

Since this library is compiled, you must add explicit instantiations of your class depending on the data types you are going to use for `ID`, `NumNonZeros`, and `Value`. 

```cpp
template class OptimalReorder<unsigned int, unsigned int, void>;
```

In addition, you must add explicit instantiations of the `ReorderInstance` class with your class as the template argument. This class is the access point of users to use reordering.

```cpp
template class ReorderInstance<unsigned int, unsigned int, void, OptimalReorder<unsigned int, unsigned int, void>>
```

Now, you can easily use your reordering like the following example:

```cpp
 
float alpha= 1.0, beta = 0.5;
ReorderInstance<unsigned int, unsigned int, void, OptimalReorder<unsigned int, unsigned int, void>> reorder(alpha, beta);
unsigned int * order = reorder.get_order(some_sparseformat_object);
```

If the format of `some_sparseformat_object` is `CSR_f`, `COO_f`, or any other format that is convertible to the two aforementioned formats, then an order will be calculated for it.