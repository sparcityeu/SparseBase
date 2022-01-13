cmake_minimum_required( VERSION 3.0 )

project( SparseBase_project VERSION 0.1.3 )
option(RUN_TESTS "Enable running tests" OFF)

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
if (MSVC)
    add_compile_options("/bigobj")
endif()
add_subdirectory(sparsebase)
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