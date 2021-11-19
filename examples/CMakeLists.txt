add_executable(example_1 ${CMAKE_CURRENT_SOURCE_DIR}/example_1.cpp)
target_link_libraries(example_1 sparsebase)

add_executable(example_conversion ${CMAKE_CURRENT_SOURCE_DIR}/example_conversion.cpp)
target_link_libraries(example_conversion sparsebase)