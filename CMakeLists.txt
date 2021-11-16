cmake_minimum_required( VERSION 3.0 )

project( SparseBase_project VERSION 0.1 )

include( CheckCXXCompilerFlag )
check_cxx_compiler_flag( "-std=c++11"   COMPILER_SUPPORTS_CXX11 )
check_cxx_compiler_flag( "-std=c++0x"   COMPILER_SUPPORTS_CXX0X )
if( COMPILER_SUPPORTS_CXX11 )
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

add_subdirectory(sparsebase)
add_subdirectory(examples)
add_subdirectory(tests)

enable_testing()

add_test(NAME basic_format COMMAND tests/run_test)