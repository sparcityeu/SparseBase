add_executable(feature_tests.test feature_tests.cc)
target_link_libraries(feature_tests.test sparsebase)
target_link_libraries(feature_tests.test gtest gtest_main)

add_test(NAME feature_tests.test COMMAND feature_tests.test)