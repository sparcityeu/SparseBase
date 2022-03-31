find_package(OpenMP REQUIRED)
set (CMAKE_CXX_STANDARD 17)
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
                --pigo ${USE_PIGO}
                --cuda ${CUDA}
                --output-folder ${PROJECT_BINARY_DIR}/init)
set(LIB_FILES 
        ${CMAKE_CURRENT_SOURCE_DIR}/sparsebase/sparse_format.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/sparsebase/sparse_reader.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/sparsebase/sparse_object.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/sparsebase/sparse_converter.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/sparsebase/sparse_preprocess.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/sparsebase/sparse_feature.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/sparsebase/sparse_writer.cc
)
if (${CUDA})

set(LIB_FILES ${LIB_FILES}
        ${CMAKE_CURRENT_SOURCE_DIR}/sparsebase/cuda/format.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/sparsebase/cuda/converter.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/sparsebase/cuda/preprocess.cu
)
endif()
add_library(sparsebase ${LIB_FILES})
# defining the include paths of the library
target_include_directories(sparsebase PUBLIC
        # Location of the headers at build time
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/sparsebase>
        $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
        # Location of the headers at install time
        $<INSTALL_INTERFACE:include>)
target_compile_features(sparsebase PRIVATE cxx_std_17)
target_link_libraries(sparsebase OpenMP::OpenMP_CXX)
else()
add_library(sparsebase INTERFACE)
target_compile_features(sparsebase INTERFACE cxx_std_17)
target_link_libraries(sparsebase INTERFACE OpenMP::OpenMP_CXX )
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
          sparsebase/sparse_feature.cc
          sparsebase/sparse_reader.cc
          sparsebase/sparse_writer.cc
          DESTINATION include/sparsebase)
if(${CUDA})
        install(FILES
                sparsebase/cuda/format.cu
                sparsebase/cuda/converter.cu
                sparsebase/cuda/preprocess.cu
                DESTINATION include/sparsebase/cuda
        )
endif()
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
          sparsebase/sparse_feature.h
          sparsebase/sparse_reader.h
          sparsebase/sparse_writer.h
          sparsebase/sparse_file_format.h
        DESTINATION include/sparsebase)
install(FILES 
          sparsebase/external/json/json.hpp
        DESTINATION include/sparsebase/external/json/)
if(${CUDA})
        install(FILES
                sparsebase/cuda/format.cuh
                sparsebase/cuda/converter.cuh
                sparsebase/cuda/preprocess.cuh
                DESTINATION include/sparsebase/cuda
        )
endif()
if(USE_PIGO)
    install(FILES sparsebase/external/pigo/pigo.hpp
            DESTINATION include/sparsebase/external/pigo)
endif()
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/sparsebase/config.h DESTINATION include/sparsebase)
#install(DIRECTORY ${CMAKE_BINARY_DIR}/init DESTINATION include/sparsebase)
include(GNUInstallDirs)

# Installing the library, and creating the export set "sparsebaseTargets" 
install(TARGETS sparsebase EXPORT sparsebaseTargets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        # telling the export set where the headers of the library are located
        INCLUDES DESTINATION include
        )
