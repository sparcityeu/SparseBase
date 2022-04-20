add_executable(cuda_example ${CMAKE_CURRENT_SOURCE_DIR}/cuda_example.cu)
target_link_libraries(cuda_example sparsebase)
