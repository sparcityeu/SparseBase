find_package(OpenMP REQUIRED)

# Requires C++17 support in the compiler
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
  if (${CUDA})
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
add_opt_library("metis")
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
