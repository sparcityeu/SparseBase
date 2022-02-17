# Creating the sparsebase library executable
if(NOT ${_HEADER_ONLY})
add_library(sparsebase
        ${CMAKE_CURRENT_SOURCE_DIR}/include/sparsebase/sparse_format.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/include/sparsebase/sparse_reader.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/include/sparsebase/sparse_object.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/include/sparsebase/sparse_converter.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/include/sparsebase/sparse_preprocess.cc
        )
# defining the include paths of the library
target_include_directories(sparsebase PUBLIC
        # Location of the headers at build time
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include/sparsebase>
        # Location of the headers at install time
        $<INSTALL_INTERFACE:include>)
target_compile_features(sparsebase PRIVATE cxx_std_17)
else()
add_library(sparsebase INTERFACE)
target_compile_features(sparsebase INTERFACE cxx_std_17)
# defining the include paths of the library
target_include_directories(sparsebase INTERFACE
        # Location of the headers at build time
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include/sparsebase>
        # Location of the headers at install time
        $<INSTALL_INTERFACE:include>)
install(FILES 
          include/sparsebase/sparse_format.cc 
          include/sparsebase/sparse_object.cc 
          include/sparsebase/sparse_preprocess.cc 
          include/sparsebase/sparse_converter.cc 
          include/sparsebase/sparse_reader.cc 
          DESTINATION include/sparsebase)
endif()

# Creating an alias for the library when used through "find_package()"
add_library(sparsebase::sparsebase ALIAS sparsebase)
# Installing the headers to the include directory
install(FILES 
          include/sparsebase/sparse_format.h 
          include/sparsebase/sparse_exception.h 
          include/sparsebase/sparse_object.h 
          include/sparsebase/sparse_preprocess.h 
          include/sparsebase/sparse_converter.h 
          include/sparsebase/sparse_reader.h 
          DESTINATION include/sparsebase)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/include/sparsebase/config.h DESTINATION include/sparsebase)
include(GNUInstallDirs)

# Installing the library, and creating the export set "sparsebaseTargets" 
install(TARGETS sparsebase EXPORT sparsebaseTargets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        # telling the export set where the headers of the library are located
        INCLUDES DESTINATION include
        )
