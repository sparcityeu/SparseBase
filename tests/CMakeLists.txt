project(Tests)

add_subdirectory(lib EXCLUDE_FROM_ALL)

include_directories(${gtest_SOURCE_DIR}/include ${gtest_source_dir})

add_executable(sparseformat_tests.test suites/SparseFormat_tests.cpp)
target_link_libraries(sparseformat_tests.test sparsebase)
target_link_libraries(sparseformat_tests.test gtest gtest_main)

add_executable(feature_tests.test suites/feature_tests.cpp)
target_compile_features(feature_tests.test PRIVATE cxx_std_17)
target_link_libraries(feature_tests.test sparsebase)
target_link_libraries(feature_tests.test gtest gtest_main)
