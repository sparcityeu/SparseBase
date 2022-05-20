configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/run_all_examples.py
  ${CMAKE_CURRENT_BINARY_DIR}/run_all_examples.py
  COPYONLY
)

file(COPY ${CMAKE_SOURCE_DIR}/examples/data DESTINATION ${CMAKE_BINARY_DIR}/examples/)

if (${CUDA})
  add_subdirectory(cuda_example)
  add_subdirectory(array_example)
endif()
if(NOT ${_HEADER_ONLY} OR NOT ${CUDA})
  add_subdirectory(csr_coo)
  add_subdirectory(degree_order)
  add_subdirectory(custom_order)
  add_subdirectory(format_conversion)
  add_subdirectory(rcm_order)
  add_subdirectory(sparse_feature)
  add_subdirectory(sparse_reader)
  add_subdirectory(custom_format)
  add_subdirectory(custom_converter)
endif()
