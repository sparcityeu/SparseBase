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
        sparsebase/format/format.cc
        sparsebase/context/context.cc
        sparsebase/utils/io/reader.cc
        sparsebase/object/object.cc
        sparsebase/utils/converter/converter.cc
        sparsebase/preprocess/preprocess.cc
        sparsebase/feature/feature.cc
        sparsebase/utils/io/writer.cc
)
if (${CUDA})

set(LIB_FILES ${LIB_FILES}
        sparsebase/format/cuda/format.cu
        sparsebase/utils/converter/cuda/converter.cu
        sparsebase/preprocess/cuda/preprocess.cu
        sparsebase/context/cuda/context.cu
)
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
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
        # Location of the headers at install time
        $<INSTALL_INTERFACE:include>)
endif()
### Installation plan
set(SINGLE_HEADER_FILE 
        sparsebase/sparsebase.h
)
set(CONTEXT_FILES 
        sparsebase/context/context.h
)
set(FORMAT_FILES 
        sparsebase/format/format.h
)
set(FEATURE_FILES 
        sparsebase/feature/feature.h
)
set(PREPROCESS_FILES 
        sparsebase/preprocess/preprocess.h
)
set(OBJECT_FILES 
        sparsebase/object/object.h
)
set(IO_FILES 
        sparsebase/utils/io/reader.h
        sparsebase/utils/io/sparse_file_format.h
        sparsebase/utils/io/writer.h
)
set(CONVERTER_FILES 
        sparsebase/utils/converter/converter.h
)
set(EXCEPTION_FILES 
        sparsebase/utils/exception.h
)
if(${_HEADER_ONLY})
set(CONTEXT_FILES ${CONTEXT_FILES}
        sparsebase/context/context.cc
)
set(FORMAT_FILES ${FORMAT_FILES}
        sparsebase/format/format.cc
)
set(FEATURE_FILES ${FEATURE_FILES}
        sparsebase/feature/feature.cc
)
set(PREPROCESS_FILES ${PREPROCESS_FILES} 
        sparsebase/preprocess/preprocess.cc
)
set(OBJECT_FILES ${OBJECT_FILES}
        sparsebase/object/object.cc
)
set(IO_FILES ${IO_FILES}
        sparsebase/utils/io/reader.cc
        sparsebase/utils/io/writer.cc
)
set(CONVERTER_FILES ${CONVERTER_FILES}
        sparsebase/utils/converter/converter.cc
)
endif()
install(FILES ${SINGLE_HEADER_FILE}
        DESTINATION include)
install(FILES ${CONTEXT_FILES}
        DESTINATION include/sparsebase/context/)
install(FILES ${FORMAT_FILES}
        DESTINATION include/sparsebase/format/)
install(FILES ${FEATURE_FILES}
        DESTINATION include/sparsebase/feature/)
install(FILES ${PREPROCESS_FILES}
        DESTINATION include/sparsebase/preprocess/)
install(FILES ${OBJECT_FILES}
        DESTINATION include/sparsebase/object/)
install(FILES ${IO_FILES}
        DESTINATION include/sparsebase/utils/io/)
install(FILES ${CONVERTER_FILES}
        DESTINATION include/sparsebase/utils/converter/)
install(FILES ${EXCEPTION_FILES}
        DESTINATION include/sparsebase/utils/)

if(USE_PIGO)
    install(FILES sparsebase/external/pigo/pigo.hpp
            DESTINATION include/sparsebase/external/pigo)
endif()

install(FILES 
          sparsebase/external/json/json.hpp
        DESTINATION include/sparsebase/external/json/)

if(${CUDA})
        set(CUDA_FORMAT_FILES
        sparsebase/format/cuda/format.cuh)
        set(CUDA_CONVERTER_FILES
        sparsebase/utils/converter/cuda/converter.cuh)
        set(CUDA_PREPROCESS_FILES
        sparsebase/preprocess/cuda/preprocess.cuh)
        set(CUDA_CONTEXT_FILES
        sparsebase/context/cuda/context.cuh)
        if(${_HEADER_ONLY})
        set(CUDA_FORMAT_FILES ${CUDA_FORMAT_FILES}
        sparsebase/format/cuda/format.cu)
        set(CUDA_CONVERTER_FILES ${CUDA_CONVERTER_FILES}
        sparsebase/utils/converter/cuda/converter.cu)
        set(CUDA_PREPROCESS_FILES ${CUDA_PREPROCESS_FILES}
        sparsebase/preprocess/cuda/preprocess.cu)
        set(CUDA_CONTEXT_FILES ${CUDA_CONTEXT_FILES}
        sparsebase/context/cuda/context.cu)
        endif()
        install(FILES ${CUDA_FORMAT_FILES}
        DESTINATION include/sparsebase/format/cuda/)
        install(FILES ${CUDA_CONVERTER_FILES}
        DESTINATION include/sparsebase/utils/converter/cuda/)
        install(FILES ${CUDA_PREPROCESS_FILES}
        DESTINATION include/sparsebase/preprocess/cuda/)
        install(FILES ${CUDA_CONTEXT_FILES}
        DESTINATION include/sparsebase/context/cuda/)
endif()

# Creating an alias for the library when used through "find_package()"
add_library(sparsebase::sparsebase ALIAS sparsebase)
# Installing the headers to the include directory
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/sparsebase/config.h DESTINATION include/sparsebase)
#install(DIRECTORY ${CMAKE_BINARY_DIR}/init DESTINATION include/sparsebase)
include(GNUInstallDirs)

# Installing the library, and creating the export set "sparsebaseTargets" 
install(TARGETS sparsebase EXPORT sparsebaseTargets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        # telling the export set where the headers of the library are located
        INCLUDES DESTINATION include
        )
