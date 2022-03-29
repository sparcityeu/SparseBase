cmake_minimum_required( VERSION 3.12 )

if (${CMAKE_VERSION} VERSION_LESS 3.18)
  cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
else()
  cmake_policy(VERSION 3.18)
endif()

project( SparseBase_project VERSION 0.1.5 )
option(RUN_TESTS "Enable running tests" OFF)
option(_HEADER_ONLY "Use the library as a header only library?" OFF)
option(CUDA "Enable CUDA" OFF)
option(USE_PIGO "Use PIGO for parallel I/O" ON)
set(ID_TYPES "unsigned int" "int" CACHE STRING "C++ data types used for variables storing IDs")
set(NNZ_TYPES "unsigned int" "int" CACHE STRING "C++ data types used for variables storing numbers of non-zeros")
set(VALUE_TYPES "unsigned int" "int" "float" CACHE STRING "C++ data types used for variables storing values inside formats")
set(FLOAT_TYPES "float" "double" CACHE STRING "C++ data types used for variables storing floating point numbers")

if(${CUDA})
  if (${CMAKE_VERSION} VERSION_LESS "3.18")
    message(FATAL_ERROR "Using CUDA requires CMake version 3.18 or higher")
  endif()
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
  set(CMAKE_CUDA_ARCHITECTURES 60 62 70 75)
endif()
if (WIN32)
    message(STATUS "PIGO is not supported on Windows.")
    set(USE_PIGO OFF)
endif()

configure_file(
    "${PROJECT_SOURCE_DIR}/src/sparsebase/config.h.in"
    "${PROJECT_BINARY_DIR}/src/sparsebase/config.h"
)

include( CheckCXXCompilerFlag )
check_cxx_compiler_flag( "-std=c++17"   COMPILER_SUPPORTS_CXX17 )
check_cxx_compiler_flag( "-std=c++11"   COMPILER_SUPPORTS_CXX11 )
check_cxx_compiler_flag( "-std=c++0x"   COMPILER_SUPPORTS_CXX0X )
if( COMPILER_SUPPORTS_CXX17 )
    if( CMAKE_COMPILER_IS_GNUCXX )
        set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++17" )
    else()
        set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17" )
    endif()
elseif( COMPILER_SUPPORTS_CXX11 )
    if( CMAKE_COMPILER_IS_GNUCXX )
        set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11" )
    else()
        set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11" )
    endif()
elseif( COMPILER_SUPPORTS_CXX0X)
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x" )
else()
    # MSVC, On by default (if available)
endif()
# Got this from https://github.com/district10/cmake-templates/blob/master/cpp11_vs2010/CMakeLists.txt
if (MSVC)
    add_compile_options("/bigobj")
endif()
add_subdirectory(src)
add_subdirectory(examples)

add_custom_target(format
    COMMAND bash ${CMAKE_SOURCE_DIR}/.format.bash ${CMAKE_SOURCE_DIR}
    COMMENT "Formatting source files"
)

if ( RUN_TESTS )
    execute_process(COMMAND git submodule update --recursive --init
    ${CMAKE_CURRENT_SOURCE_DIR})
    add_subdirectory(tests)
    enable_testing()
    add_test(NAME sparseformat_tests COMMAND tests/sparseformat_tests.test)
endif()

# Create the sparseprocess export set
install(EXPORT sparsebaseTargets
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/sparsebase
        NAMESPACE sparsebase::
        FILE sparsebaseTargets.cmake
        )

# configure the package configuration file
include(CMakePackageConfigHelpers)
configure_package_config_file(
    "Config.cmake.in"
    "sparsebaseConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/sparsebase
    PATH_VARS
        CMAKE_INSTALL_LIBDIR
)

# create the package version file
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/sparsebaseConfigVersion.cmake
    VERSION ${sparsebase_VERSION}
    COMPATIBILITY SameMajorVersion
)

# install the package
install(
    FILES   "${CMAKE_CURRENT_BINARY_DIR}/sparsebaseConfig.cmake"
            "${CMAKE_CURRENT_BINARY_DIR}/sparsebaseConfigVersion.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/sparsebase"
)
