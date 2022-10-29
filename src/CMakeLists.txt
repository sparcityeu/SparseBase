find_package(OpenMP REQUIRED)
set (CMAKE_CXX_STANDARD 17)

file(GLOB_RECURSE CC_FILES *.cc)
file(GLOB_RECURSE H_FILES *.h)
file(GLOB_RECURSE CU_FILES *cu)
file(GLOB_RECURSE CUH_FILES *cuh)

# Creating the sparsebase library executable
if(NOT ${_HEADER_ONLY})

  include(cmake/generate_explicit_instantiations.cmake)
  generate_explicit_instantiations()

  set(LIB_FILES ${CC_FILES})
  if (${USE_CUDA})
    set(LIB_FILES ${LIB_FILES} ${CU_FILES})
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

include(cmake/add_opt_library.cmake)

### External Optional Libraries
option(USE_METIS "Enable METIS integration" OFF)
if(USE_METIS)
  add_opt_library("metis" "OFF")
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

### Create configuration file for install
configure_file(
  "${PROJECT_SOURCE_DIR}/src/sparsebase/config.h.in"
  "${PROJECT_BINARY_DIR}/src/sparsebase/config.h"
)

# Creating an alias for the library when used through "find_package()"
add_library(sparsebase::sparsebase ALIAS sparsebase)
# Installing the headers to the include directory
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/sparsebase/config.h DESTINATION include/sparsebase)
#install(DIRECTORY ${CMAKE_BINARY_DIR}/init DESTINATION include/sparsebase)
include(GNUInstallDirs)

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

# Installing the library, and creating the export set "sparsebaseTargets" 
install(TARGETS sparsebase EXPORT sparsebaseTargets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        # telling the export set where the headers of the library are located
        INCLUDES DESTINATION include
        )
