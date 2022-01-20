add_executable(csr_coo ${CMAKE_CURRENT_SOURCE_DIR}/csr_coo.cpp)
target_link_libraries(csr_coo sparsebase)
target_compile_features(csr_coo PRIVATE cxx_std_17)
