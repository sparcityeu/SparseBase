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
    --cuda ${USE_CUDA}
    --output-folder ${PROJECT_BINARY_DIR}/init
    --class-list ${CMAKE_SOURCE_DIR}/src/class_instantiation_list.json)

  message(STATUS "Generated explicit instantiations")
  set(LIB_FILES
          sparsebase/format/format_order_one.cc
          sparsebase/format/array.cc
          sparsebase/format/format_order_two.cc
          sparsebase/format/csr.cc
          sparsebase/format/csc.cc
          sparsebase/format/coo.cc
          sparsebase/context/cpu_context.cc
          sparsebase/io/mtx_reader.cc
          sparsebase/io/pigo_mtx_reader.cc
          sparsebase/io/edge_list_reader.cc
          sparsebase/io/pigo_edge_list_reader.cc
          sparsebase/io/binary_reader_order_one.cc
          sparsebase/io/binary_reader_order_two.cc
          sparsebase/io/binary_writer_order_one.cc
          sparsebase/io/binary_writer_order_two.cc
          sparsebase/object/object.cc
          sparsebase/converter/converter.cc
          sparsebase/converter/converter_order_two.cc
          sparsebase/converter/converter_order_one.cc
          sparsebase/converter/converter_store.cc
          sparsebase/preprocess/preprocess.cc
          sparsebase/reorder/reorderer.cc
          sparsebase/reorder/rcm_reorder.cc
          sparsebase/reorder/degree_reorder.cc
          sparsebase/reorder/gray_reorder.cc
          sparsebase/reorder/generic_reorder.cc
          sparsebase/reorder/amd_reorder.cc
          sparsebase/reorder/rabbit_reorder.cc
          sparsebase/reorder/metis_reorder.cc
          sparsebase/reorder/reorder_heatmap.cc
          sparsebase/permute/permuter.cc
          sparsebase/permute/permute_order_one.cc
          sparsebase/permute/permute_order_two.cc
          sparsebase/partition/partitioner.cc
          sparsebase/partition/metis_partition.cc
          sparsebase/partition/patoh_partition.cc
          sparsebase/partition/pulp_partition.cc
          sparsebase/feature/extractor.cc
          sparsebase/feature/feature_extractor.cc
          sparsebase/experiment/concrete_experiment.cc
          sparsebase/utils/utils.cc
          sparsebase/utils/logger.cc
          )
  if (${USE_CUDA})

    set(LIB_FILES ${LIB_FILES}
            sparsebase/format/cuda_csr_cuda.cu
            sparsebase/format/cuda_array_cuda.cu
            sparsebase/converter/converter_cuda.cu
            sparsebase/preprocess/cuda/preprocess.cu
            sparsebase/context/cuda_context_cuda.cu
            sparsebase/converter/converter_order_two_cuda.cu
            sparsebase/converter/converter_order_one_cuda.cu
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

# Function for adding optional libraries to sparsebase
# Example : add_opt_library("metis" ${METIS_LIB_DIR} ${METIS_INC_DIR})
macro(add_opt_library name LOCAL_HEADER_ONLY)

  string(TOUPPER ${name} OPT_NAME_UPPER)
  string(TOLOWER ${name} OPT_NAME_LOWER)

  if (NOT "${LOCAL_HEADER_ONLY}")
    set(OPT_LIB "LIB-NOTFOUND")
  endif()
  set(OPT_INC "INC-NOTFOUND")
  set(OPT_DEF USE_${OPT_NAME_UPPER})

  set(${OPT_NAME_UPPER}_INC_DIR "" CACHE STRING "Include directory for ${OPT_NAME_UPPER}")
  if (NOT "${LOCAL_HEADER_ONLY}")
    set(${OPT_NAME_UPPER}_LIB_DIR "" CACHE STRING "Library directory for ${OPT_NAME_UPPER}")
  endif()

  if (NOT "${LOCAL_HEADER_ONLY}")
    find_library(
            OPT_LIB
            NAMES ${OPT_NAME_LOWER} lib${OPT_NAME_LOWER}
            PATH_SUFFIXES lib lib32 lib64 lib{OPT_NAME_LOWER}
            PATHS
            ${${OPT_NAME_UPPER}_LIB_DIR}
            $ENV{${OPT_NAME_UPPER}_LIB_DIR}
            ${CMAKE_PREFIX_PATH}
            ${CMAKE_SYSTEM_PREFIX_PATH}
            NO_DEFAULT_PATH

    )

    if(NOT OPT_LIB)
      message(FATAL_ERROR "${OPT_NAME_UPPER} library file was not found. "
              "Try setting the ${OPT_NAME_UPPER}_LIB_DIR environment variable")
    endif()
  endif()


  find_path(
          OPT_INC
          NAMES ${OPT_NAME_LOWER}.h ${OPT_NAME_LOWER}.hpp ${OPT_NAME_LOWER}.hxx
          PATH_SUFFIXES include
          PATHS
          ${${OPT_NAME_UPPER}_INC_DIR}
          $ENV{${OPT_NAME_UPPER}_INC_DIR}
          ${CMAKE_PREFIX_PATH}
          ${CMAKE_SYSTEM_PREFIX_PATH}
          NO_DEFAULT_PATH
  )

  if(NOT OPT_INC)
    message(FATAL_ERROR "${OPT_NAME_UPPER} include file was not found. "
            "Try setting the ${OPT_NAME_UPPER}_INC_DIR environment variable")
  endif()


  message(STATUS "${OPT_NAME_UPPER} was found")
  message(STATUS "${OPT_NAME_UPPER} include dir: ${OPT_INC}")
  if (NOT "${LOCAL_HEADER_ONLY}")
    message(STATUS "${OPT_NAME_UPPER} library dir: ${OPT_LIB}")
  else()
    message(STATUS "${OPT_NAME_UPPER} library added as a header-only library")
  endif()
  if (NOT "${LOCAL_HEADER_ONLY}")
    if (NOT "${_HEADER_ONLY}")
      target_link_libraries(sparsebase PUBLIC ${OPT_LIB})
    else()
      target_link_libraries(sparsebase INTERFACE ${OPT_LIB})
    endif()
  endif()

  if (NOT "${_HEADER_ONLY}")
    target_include_directories(sparsebase PUBLIC ${OPT_INC})
  else()
    target_include_directories(sparsebase INTERFACE ${OPT_INC})
  endif()
  set(${OPT_DEF} ON)
endmacro()

### External Optional Libraries
option(USE_METIS "Enable METIS integration" OFF)
if(USE_METIS)
  add_opt_library("metis" "OFF")
endif()

option(USE_AMD_ORDER "Enable AMD reorder integration" OFF)
if(USE_AMD_ORDER)
  add_opt_library("amd" "OFF")
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

option(USE_PULP "Enable PULP integration" OFF)
if(USE_PULP)
  add_opt_library("pulp" "OFF")
endif()

option(USE_PATOH "Enable PATOH integration" OFF)
if(USE_PATOH)
  add_opt_library("patoh" "OFF")
endif()

### Create configuration file for install
configure_file(
        "${PROJECT_SOURCE_DIR}/src/sparsebase/config.h.in"
        "${PROJECT_BINARY_DIR}/src/sparsebase/config.h"
)

### Installation plan
set(SINGLE_HEADER_FILE
        sparsebase/sparsebase.h
        )
set(CONTEXT_FILES
        sparsebase/context/context.h
        sparsebase/context/cpu_context.h
        )
set(FORMAT_FILES
        sparsebase/format/format.h
        sparsebase/format/format_implementation.h
        sparsebase/format/format_order_one.h
        sparsebase/format/array.h
        sparsebase/format/format_order_two.h
        sparsebase/format/csr.h
        sparsebase/format/csc.h
        sparsebase/format/coo.h
        )
set(FEATURE_FILES
        sparsebase/feature/extractor.h
        sparsebase/feature/feature_extractor.h
        )
set(PREPROCESS_FILES
        sparsebase/preprocess/preprocess.h
        )
set(REORDERING_FILES
        sparsebase/reorder/reorderer.h
        sparsebase/reorder/rcm_reorder.h
        sparsebase/reorder/degree_reorder.h
        sparsebase/reorder/gray_reorder.h
        sparsebase/reorder/generic_reorder.h
        sparsebase/reorder/amd_reorder.h
        sparsebase/reorder/rabbit_reorder.h
        sparsebase/reorder/metis_reorder.h
        sparsebase/reorder/reorder_heatmap.h
        )
set(PERMUTER_FILES
        sparsebase/permute/permuter.h
        sparsebase/permute/permute_order_one.h
        sparsebase/permute/permute_order_two.h
        )
set(PARTITIONER_FILES
        sparsebase/partition/partitioner.h
        sparsebase/partition/metis_partition.h
        sparsebase/partition/patoh_partition.h
        sparsebase/partition/pulp_partition.h
        )
set(OBJECT_FILES
        sparsebase/object/object.h
        )
set(IO_FILES
        sparsebase/bases/iobase.h
        sparsebase/bases/reorder_base.h
        sparsebase/io/reader.h
        sparsebase/io/mtx_reader.h
        sparsebase/io/pigo_mtx_reader.h
        sparsebase/io/edge_list_reader.h
        sparsebase/io/pigo_edge_list_reader.h
        sparsebase/io/binary_reader_order_one.h
        sparsebase/io/binary_reader_order_two.h
        sparsebase/io/sparse_file_format.h
        sparsebase/io/writer.h
        sparsebase/io/binary_writer_order_one.h
        sparsebase/io/binary_writer_order_two.h
        )
set(CONVERTER_FILES
        sparsebase/converter/converter.h
        sparsebase/converter/converter_order_two.h
        sparsebase/converter/converter_order_one.h
        sparsebase/converter/converter_order_two_cuda.cuh
        sparsebase/converter/converter_order_one_cuda.cuh
        sparsebase/converter/converter_store.h
        )
set(UTILS_FILES
        sparsebase/utils/exception.h
        sparsebase/utils/utils.h
        sparsebase/utils/logger.h
        sparsebase/utils/parameterizable.h
        sparsebase/utils/function_matcher_mixin.h
        sparsebase/utils/class_matcher_mixin.h
        )
set(EXPERIMENT_FILES
        sparsebase/experiment/experiment.h
        sparsebase/experiment/concrete_experiment.h
        sparsebase/experiment/experiment_type.h
        sparsebase/experiment/experiment_helper.h
        )
if(${_HEADER_ONLY})
  set(CONTEXT_FILES ${CONTEXT_FILES}
          sparsebase/context/cpu_context.cc
          )
  set(FORMAT_FILES ${FORMAT_FILES}
          sparsebase/format/format_order_one.cc
          sparsebase/format/array.cc
          sparsebase/format/format_order_two.cc
          sparsebase/format/csr.cc
          sparsebase/format/csc.cc
          sparsebase/format/coo.cc
          )
  set(FEATURE_FILES ${FEATURE_FILES}
          sparsebase/feature/extractor.cc
          sparsebase/feature/feature_extractor.cc
          )
  set(PREPROCESS_FILES ${PREPROCESS_FILES}
          sparsebase/preprocess/preprocess.cc
          )
  set(REORDERING_FILES ${REORDERING_FILES}
          sparsebase/reorder/reorderer.cc
          sparsebase/reorder/rcm_reorder.cc
          sparsebase/reorder/degree_reorder.cc
          sparsebase/reorder/gray_reorder.cc
          sparsebase/reorder/generic_reorder.cc
          sparsebase/reorder/amd_reorder.cc
          sparsebase/reorder/rabbit_reorder.cc
          sparsebase/reorder/metis_reorder.cc
          sparsebase/reorder/reorder_heatmap.cc
          )
  set(PERMUTER_FILES ${PERMUTER_FILES}
          sparsebase/permute/permuter.cc
          sparsebase/permute/permute_order_one.cc
          sparsebase/permute/permute_order_two.cc
          )
  set(PARTITIONER_FILES ${PARTITIONER_FILES}
          sparsebase/partition/partitioner.cc
          sparsebase/partition/metis_partition.cc
          sparsebase/partition/patoh_partition.cc
          sparsebase/partition/pulp_partition.cc
          )

  set(OBJECT_FILES ${OBJECT_FILES}
          sparsebase/object/object.cc
          )
  set(IO_FILES ${IO_FILES}
          sparsebase/io/mtx_reader.cc
          sparsebase/io/pigo_mtx_reader.cc
          sparsebase/io/edge_list_reader.cc
          sparsebase/io/pigo_edge_list_reader.cc
          sparsebase/io/binary_reader_order_one.cc
          sparsebase/io/binary_reader_order_two.cc
          sparsebase/io/binary_writer_order_one.cc
          sparsebase/io/binary_writer_order_two.cc
          )
  set(CONVERTER_FILES ${CONVERTER_FILES}
          sparsebase/converter/converter.cc
          sparsebase/converter/converter_order_two.cc
          sparsebase/converter/converter_order_one.cc
          sparsebase/converter/converter_order_two_cuda.cu
          sparsebase/converter/converter_order_one_cuda.cu
          sparsebase/converter/converter_store.cc
          )
  set(EXPERIMENT_FILES ${EXPERIMENT_FILES}
          sparsebase/experiment/concrete_experiment.cc
          )
  set(UTILS_FILES ${UTILS_FILES}
          sparsebase/utils/utils.cc
          sparsebase/utils/logger.cc
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
install(FILES  ${REORDERING_FILES}
        DESTINATION include/sparsebase/reorder/)
install(FILES  ${PERMUTER_FILES}
        DESTINATION include/sparsebase/permute/)
install(FILES  ${PARTITIONER_FILES}
        DESTINATION include/sparsebase/partition/)
install(FILES ${OBJECT_FILES}
        DESTINATION include/sparsebase/object/)
install(FILES ${IO_FILES}
        DESTINATION include/sparsebase/io/)
install(FILES ${CONVERTER_FILES}
        DESTINATION include/sparsebase/converter/)
install(FILES ${EXPERIMENT_FILES}
        DESTINATION include/sparsebase/experiment/)
install(FILES ${UTILS_FILES}
        DESTINATION include/sparsebase/utils/)

if(USE_PIGO)
  install(FILES sparsebase/external/pigo/pigo.hpp
          DESTINATION include/sparsebase/external/pigo)
endif()

install(FILES
        sparsebase/external/json/json.hpp
        DESTINATION include/sparsebase/external/json/)

if(${USE_CUDA})
  set(CUDA_FORMAT_FILES
          sparsebase/format/cuda_csr_cuda.cuh
          sparsebase/format/cuda_array_cuda.cuh)
  set(CUDA_UTILS_FILES
          sparsebase/utils/utils_cuda.cuh)
  set(CUDA_CONVERTER_FILES
          sparsebase/converter/converter_cuda.cuh)
  set(CUDA_PREPROCESS_FILES
          sparsebase/preprocess/cuda/preprocess.cuh)
  set(CUDA_CONTEXT_FILES
          sparsebase/context/cuda_context_cuda.cuh)
  if(${_HEADER_ONLY})
    set(CUDA_FORMAT_FILES ${CUDA_FORMAT_FILES}
            sparsebase/format/cuda_csr_cuda.cu
            sparsebase/format/cuda_array_cuda.cu)
    set(CUDA_CONVERTER_FILES ${CUDA_CONVERTER_FILES}
            sparsebase/converter/converter_cuda.cu
            sparsebase/converter/converter_order_two_cuda.cu
            sparsebase/converter/converter_order_one_cuda.cu
            )
    set(CUDA_PREPROCESS_FILES ${CUDA_PREPROCESS_FILES}
            sparsebase/preprocess/cuda/preprocess.cu)
    set(CUDA_CONTEXT_FILES ${CUDA_CONTEXT_FILES}
            sparsebase/context/cuda_context_cuda.cu)
  endif()
  install(FILES ${CUDA_FORMAT_FILES}
          DESTINATION include/sparsebase/format/)
  install(FILES ${CUDA_UTILS_FILES}
          DESTINATION include/sparsebase/utils/)
  install(FILES ${CUDA_CONVERTER_FILES}
          DESTINATION include/sparsebase/converter/)
  install(FILES ${CUDA_PREPROCESS_FILES}
          DESTINATION include/sparsebase/preprocess/cuda/)
  install(FILES ${CUDA_CONTEXT_FILES}
          DESTINATION include/sparsebase/context/)
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
