# Adding New Dependencies

SparseBase contains two different types of dependencies:
- Required dependencies.
- Optional dependencies.

## Required Dependencies

A dependency is required if the library loses a substantial amount of functionality without it.
To keep the library simple and cross-platform, we currently avoid all required non-header-only dependencies.

Currently, for choosing required dependencies we suggest the following options in order of preference:
1. Look for header-only libraries
2. Check if it can be achieved by calling a Python script (we already depend on Python in our build system)
3. Look for UNIX commandline tools
4. Open a [Github](https://github.com/sparcityeu/sparsebase) issue for discussion

```{note}
SparseBase is designed as an HPC library, so you shouldn't be forced to use a header-only solution
if it lacks the performance necessary. If this is the case please open an issue on 
[Github](https://github.com/sparcityeu/sparsebase) so that we can discuss.
```

### Header Only Dependencies

All header-only dependencies must be located in their own directory located at `src/sparsebase/external`
This directory should contain the header files to be included as well as a LICENSE file containing 
the license for the dependency.

When choosing a header-only dependency please make sure it uses a permissive license (like MIT,Apache,BSD).
Copy-left licences (like GPL) can cause legal issues since the SparseBase itself uses a permissive license.
If you are not sure, please open an issue on [Github](https://github.com/sparcityeu/sparsebase).


### Calling Python and UNIX Tools

This can be achieved easily by using [std::system](https://en.cppreference.com/w/cpp/utility/program/system)


## Optional Dependencies

Optional dependencies in the context of SparseBase are libraries that for one reason or 
another can not be directly bundled with SparseBase. Some examples:
- [METIS](https://github.com/KarypisLab/METIS) (Has some compilation options that are best chosen by the user)
- [PATOH](https://faculty.cc.gatech.edu/~umit/software.html) (Not open-source)

```{note}
If the dependency is header-only we suggest treating it as a required dependency 
even if it doesn't have substantial effect on the library.
```

Optional dependencies must be build separately by users.

Here are the steps to add an optional dependency:

1. In the `src/CMakeLists.txt` file add a new option in the form `USE_<dependency>`. 
This will be set by the user to turn the dependency ON or OFF. It should always default to OFF.
    ```cmake
    option(USE_METIS "Enable METIS integration" OFF)
    ```

2. In the same file call the `add_opt_library` macro with the name of the library if it is ON.
This name should match the library's built file (ie, `lib<name>.so` or `lib<name>.a` etc.)
    ```cmake
    if(USE_METIS)
        add_opt_library("metis")
    endif()
    ```

3. In the `src/config.h.in` file add `#cmakedefine USE_<dependency>`
    ```cpp
    #cmakedefine USE_METIS
    ```

4. Now the dependency should be accessible. The included files will be in `sparsebase/external/<dependency>`. 
You should surround every usage with `#ifdef USE_<dependency> ... #endif`.
    ```cpp
    #ifdef USE_METIS

    #include "sparsebase/external/metis/metis.h"

    template <typename IDType, typename NNZType, typename ValueType>
    MetisPartition<IDType, NNZType, ValueType>::MetisPartition(){...}

    #endif
    ```