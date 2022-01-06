add_executable(degree_order ${CMAKE_CURRENT_SOURCE_DIR}/degree_order.cpp)
target_link_libraries(degree_order sparsebase)
target_compile_features(degree_order PRIVATE cxx_std_17)
