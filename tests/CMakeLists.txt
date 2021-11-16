project(Tests)

add_subdirectory(lib)

include_directories(${gtest_SOURCE_DIR}/include ${gtest_source_dir})

add_executable(run_test tests.cpp)

target_link_libraries(run_test sparsebase)
target_link_libraries(run_test gtest gtest_main)