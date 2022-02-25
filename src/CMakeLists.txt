# Creating the sparsebase library executable
if(NOT ${_HEADER_ONLY})
find_package (Python COMPONENTS Interpreter)
list(JOIN ID_TYPES "," ID_TYPES_JOINED)
list(JOIN NNZ_TYPES "," NNZ_TYPES_JOINED)
list(JOIN VALUE_TYPES "," VALUE_TYPES_JOINED)
list(JOIN FLOAT_TYPES "," FLOAT_TYPES_JOINED)
execute_process(COMMAND ${Python_EXECUTABLE}
                ${CMAKE_SOURCE_DIR}/src/generate_explicit_instantiations.py
                --id-types ${ID_TYPES_JOINED} 
                --nnz-types ${NNZ_TYPES_JOINED} 
                --value-types ${VALUE_TYPES_JOINED} 
                --float-types ${FLOAT_TYPES_JOINED}
                --output-folder ${PROJECT_BINARY_DIR}/init)
add_library(sparsebase
        ${CMAKE_CURRENT_SOURCE_DIR}/sparsebase/sparse_format.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/sparsebase/sparse_reader.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/sparsebase/sparse_object.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/sparsebase/sparse_converter.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/sparsebase/sparse_preprocess.cc
        )
# defining the include paths of the library
target_include_directories(sparsebase PUBLIC
        # Location of the headers at build time
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/sparsebase>
        $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
        # Location of the headers at install time
        $<INSTALL_INTERFACE:include>)
target_compile_features(sparsebase PRIVATE cxx_std_17)
else()
add_library(sparsebase INTERFACE)
target_compile_features(sparsebase INTERFACE cxx_std_17)
# defining the include paths of the library
target_include_directories(sparsebase INTERFACE
        # Location of the headers at build time
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/sparsebase>
        # Location of the headers at install time
        $<INSTALL_INTERFACE:include>)
install(FILES 
          sparsebase/sparse_format.cc 
          sparsebase/sparse_object.cc 
          sparsebase/sparse_preprocess.cc 
          sparsebase/sparse_converter.cc 
          sparsebase/sparse_reader.cc 
          DESTINATION include/sparsebase)
endif()

# Creating an alias for the library when used through "find_package()"
add_library(sparsebase::sparsebase ALIAS sparsebase)
# Installing the headers to the include directory
install(FILES 
          sparsebase/sparse_format.h 
          sparsebase/sparse_exception.h 
          sparsebase/sparse_object.h 
          sparsebase/sparse_preprocess.h 
          sparsebase/sparse_converter.h 
          sparsebase/sparse_reader.h 
          DESTINATION include/sparsebase)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/sparsebase/config.h DESTINATION include/sparsebase)
#install(DIRECTORY ${CMAKE_BINARY_DIR}/init DESTINATION include/sparsebase)
include(GNUInstallDirs)

# Installing the library, and creating the export set "sparsebaseTargets" 
install(TARGETS sparsebase EXPORT sparsebaseTargets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        # telling the export set where the headers of the library are located
        INCLUDES DESTINATION include
        )
