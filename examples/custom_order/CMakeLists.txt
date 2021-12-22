add_executable(custom_order ${CMAKE_CURRENT_SOURCE_DIR}/custom_order.cpp)
target_link_libraries(custom_order sparsebase)
target_compile_features(custom_order PRIVATE cxx_std_17)
