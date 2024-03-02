cmake_minimum_required(VERSION 3.16)

if(APPLE)
  find_package(OpenMP)
  # From https://github.com/microsoft/LightGBM/pull/5563/files
  if(NOT OpenMP_FOUND)
    # libomp 15.0+ from brew is keg-only, so have to search in other locations.
    # See https://github.com/Homebrew/homebrew-core/issues/112107#issuecomment-1278042927.
    execute_process(COMMAND brew --prefix libomp
            OUTPUT_VARIABLE HOMEBREW_LIBOMP_PREFIX
            OUTPUT_STRIP_TRAILING_WHITESPACE)
    set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I${HOMEBREW_LIBOMP_PREFIX}/include")
    set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${HOMEBREW_LIBOMP_PREFIX}/include")
    set(OpenMP_C_LIB_NAMES omp)
    set(OpenMP_CXX_LIB_NAMES omp)
    set(OpenMP_omp_LIBRARY ${HOMEBREW_LIBOMP_PREFIX}/lib/libomp.dylib)
    find_package(OpenMP REQUIRED)
  endif()
else()
  find_package(OpenMP REQUIRED)
endif()

set (CMAKE_CXX_STANDARD 17)
# Creating the sparsebase library executable
if(NOT ${_HEADER_ONLY})
  include(cmake/generate_instantiations.cmake)
  generate_instantiations()

  # Find the C++ source files
  file(GLOB_RECURSE CC_FILES *.cc)

  # Find the CUDA source files
  file(GLOB_RECURSE CU_FILES *.cu)

  # Set library source files
  set(LIB_FILES ${CC_FILES})
  if (${USE_CUDA})
    set(LIB_FILES ${CC_FILES} ${CU_FILES})
  endif()

  add_library(sparsebase ${LIB_FILES})
  # defining the include paths of the library
  target_include_directories(sparsebase PUBLIC
          # Location of the headers at build time
          $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
          $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
          $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
          # Location of the headers at install time
          $<INSTALL_INTERFACE:include>)
  target_compile_features(sparsebase PUBLIC cxx_std_17)
  target_link_libraries(sparsebase PUBLIC OpenMP::OpenMP_CXX)
else()
  add_library(sparsebase INTERFACE)
  target_compile_features(sparsebase INTERFACE cxx_std_17)
  target_link_libraries(sparsebase INTERFACE OpenMP::OpenMP_CXX )
  # defining the include paths of the library
  target_include_directories(sparsebase INTERFACE
          # Location of the headers at build time
          $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
          $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
          # Location of the headers at install time
          $<INSTALL_INTERFACE:include>)
endif()

# Contains macro for adding optional libraries
include(cmake/add_opt_library.cmake)

# External Optional Libraries
option(USE_METIS "Enable METIS integration" OFF)
if(USE_METIS)
  add_opt_library("metis" "OFF")
endif()

# External Optional Libraries q qq
option(USE_MTKAHYPAR "Enable MTKAHYPAR integration" OFF)
if(USE_MTKAHYPAR)
  add_opt_library("mtkahypar" "OFF")
endif()

option(USE_AMD_ORDER "Enable AMD reorder integration" OFF)
if(USE_AMD_ORDER)
  add_opt_library("amd" "OFF")
endif()

option(USE_RABBIT_ORDER "Enable Rabbit order integration" OFF)
if(USE_RABBIT_ORDER)
  find_package(Boost)
  if (NOT Boost_FOUND)
    message(FATAL_ERROR "Boost is required to use Rabbit order")
  endif()
  if (NOT ${_HEADER_ONLY})
    target_include_directories(sparsebase PRIVATE ${Boost_INCLUDE_DIRS} )
  else()
    target_include_directories(sparsebase INTERFACE ${Boost_INCLUDE_DIRS} )
  endif()
  find_library(NUMA_LIBRARIES NAMES numa)
  if (NOT NUMA_LIBRARIES)
    message(FATAL_ERROR "Numa is required to use Rabbit order")
  endif()
  if (NOT ${_HEADER_ONLY})
    target_link_libraries(sparsebase PUBLIC numa)
  else()
    target_link_libraries(sparsebase INTERFACE numa)
  endif()
  add_opt_library("rabbit_order" "ON")
endif()

option(USE_PULP "Enable PULP integration" OFF)
if(USE_PULP)
  add_opt_library("pulp" "OFF")
endif()

option(USE_PATOH "Enable PATOH integration" OFF)
if(USE_PATOH)
  add_opt_library("patoh" "OFF")
endif()

### Create configuration file for install
configure_file(
        "${PROJECT_SOURCE_DIR}/src/sparsebase/config.h.in"
        "${PROJECT_BINARY_DIR}/src/sparsebase/config.h"
)

# Defines some paths used for installation
include(GNUInstallDirs)

# Install the files
# If header_only excludes all ".cc" files
if(${_HEADER_ONLY})
  install(DIRECTORY sparsebase
          DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
          PATTERN "*.py" EXCLUDE
          PATTERN "*.in" EXCLUDE
          )
else()
  install(DIRECTORY sparsebase
          DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
          PATTERN "*.py" EXCLUDE
          PATTERN "*.in" EXCLUDE
          PATTERN "*.cc" EXCLUDE
          PATTERN "*.cu" EXCLUDE
          )
endif()


# Install the created config file
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/sparsebase/config.h DESTINATION include/sparsebase)

# Creating an alias for the library when used through "find_package()"
add_library(sparsebase::sparsebase ALIAS sparsebase)

# Installing the library, and creating the export set "sparsebaseTargets"
install(TARGETS sparsebase EXPORT sparsebaseTargets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        # telling the export set where the headers of the library are located
        INCLUDES DESTINATION include
        )
