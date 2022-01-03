project(Tests)

add_subdirectory(lib EXCLUDE_FROM_ALL)

include_directories(${gtest_SOURCE_DIR}/include ${gtest_source_dir})

add_executable(sparseformat_tests.test suites/SparseFormat_tests.cpp)
target_link_libraries(sparseformat_tests.test sparsebase)
target_link_libraries(sparseformat_tests.test gtest gtest_main)