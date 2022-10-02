cmake_minimum_required( VERSION 3.12 )

if (${CMAKE_VERSION} VERSION_LESS 3.18)
  cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
else()
  cmake_policy(VERSION 3.18)
endif()

project( SparseBase_project VERSION 0.2.4 )
option(RUN_TESTS "Enable running tests" OFF)
option(_HEADER_ONLY "Use the library as a header only library?" OFF)
option(USE_CUDA "Enable CUDA" OFF)
option(USE_PIGO "Use PIGO for parallel I/O" ON)
option(BUILD_EXAMPLES "Build example codes" ON)
set(ID_TYPES "unsigned int" "int" CACHE STRING "C++ data types used for variables storing IDs")
set(NNZ_TYPES "unsigned int" "int" CACHE STRING "C++ data types used for variables storing numbers of non-zeros")
set(VALUE_TYPES "unsigned int" "int" "float" "double" CACHE STRING "C++ data types used for variables storing values inside formats")
set(FLOAT_TYPES "float" "double" CACHE STRING "C++ data types used for variables storing floating point numbers")


if(${USE_CUDA})
  if (${CMAKE_VERSION} VERSION_LESS "3.18")
    message(FATAL_ERROR "Using CUDA requires CMake version 3.18 or higher")
  endif()
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)

  include(FindCUDA/select_compute_arch)
  cuda_select_nvcc_arch_flags(CUDA_ARCH_FLAGS_RAW Auto)
  string(REGEX MATCHALL "[0-9][0-9]" CUDA_ARCHS ${CUDA_ARCH_FLAGS_RAW})
  set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCHS})
  # The four lines above figure out the architectures of CUDA for the GPUs in
  # this system automatically. Technically, using FindCuda is deprecated.
  # However, there is no modern CMAKE alternative until CMake 3.24:
  # https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html#prop_tgt:CUDA_ARCHITECTURES
  # If CMake is upgraded to 3.24, the above four lines will be replaced by
  # set(CMAKE_CUDA_ARCHITECTURES native)
endif()
if (WIN32 OR APPLE)
  message(STATUS "PIGO is not supported on Windows and MacOS at the moment.")
  set(USE_PIGO OFF)
endif()



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
if (${BUILD_EXAMPLES})
  add_subdirectory(examples)
endif()

add_custom_target(format
  COMMAND bash ${CMAKE_SOURCE_DIR}/.format.bash ${CMAKE_SOURCE_DIR}
  COMMENT "Formatting source files"
)

if ( RUN_TESTS )
  execute_process(
    COMMAND git submodule update --recursive --init
    ${CMAKE_CURRENT_SOURCE_DIR}
  )
  enable_testing()
  add_subdirectory(tests)
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

# install license
install(
    FILES LICENSE
    DESTINATION include/sparsebase
)
install(
    FILES LICENSE
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/sparsebase"
)
