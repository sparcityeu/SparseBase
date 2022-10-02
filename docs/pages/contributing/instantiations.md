# Instantiating Classes in SparseBase

While SparseBase supports a header-only mode, our main focus is in the compiled version of the library. 
This is mainly due to the large impact of header-only mode on the build times of user's projects.

As a result of this, we try to keep the header files as light as possible by avoiding putting implementations there.
There are a few exceptions to this (like the `As<>()` function of `Format`) 
but this is only done in very rare situations.

Hence, any templated class or function you create in SparseBase will require instantiation as it needs to be compiled
with the library with the types known at compile time. This is achieved with the script 
`src/generate_explicit_instantiations.py`.

```{note}
This script is not intended to be called manually besides debugging. It will be called automatically
by CMake during the configuration step.
```

To add a new instantiation you need to call the following function. Each parameter is explained in its own section.

```python
def gen_inst(template, filename, ifdef=None, folder=output_folder):
```

## Template

**Example:** `CSR<$id_type, $nnz_type, $value_type>`

The following operations will be performed on the template you provide:

1. Template will be placed in `template class <your_template>;\n`
2. The identifiers starting with `$` will be replaced with their counterparts in the CMake variables. Currently, we have
`$id_type`, `$nnz_type`, `$value_type`, `$float_type`. Every possible combination will be covered.


## Filename

The filename to be used for the output. As per [Google's Style Guide](https://google.github.io/styleguide/cppguide.html),
the filenames we use end with the `.inc` extension. 

```{warning}
If the file is being used for the first time you should also call the `reset_file()` function with it.
Otherwise you may encounter duplicate instantiation errors
```

## Ifdef

This is an optional parameter. If used, the instantiations will be surrounded with an `#ifdef ... #endif` using the 
string provided in this parameter as the definition name.

**Example:**

```cpp
// Generated when the function is called with ifdef="USE_METIS" 
#ifdef USE_METIS
template class MetisPartition<int,int,double>;
template class MetisPartition<int,int,float>;
#endif
```

## Folder

By default, the script will put everything in the `output_folder` argument passed by CMake. 
This optional parameter can be used to override that.
Generally you will want to use this to open a new folder under `output_folder`.

```{warning}
Files under different folders are different from each other. And as a result the `reset_file()`
function must be called on each file separately. Note that `reset_file()` also takes an optional folder 
parameter for this purpose.
```
