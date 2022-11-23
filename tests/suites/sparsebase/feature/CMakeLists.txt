if(${USE_CUDA})
set_source_files_properties(feature_tests.cc PROPERTIES LANGUAGE CUDA)
endif()
add_executable(sparsebase_feature_feature_tests.test feature_tests.cc)
target_link_libraries(sparsebase_feature_feature_tests.test sparsebase)
target_link_libraries(sparsebase_feature_feature_tests.test gtest gtest_main)

add_test(NAME sparsebase_feature_feature_tests.test COMMAND sparsebase_feature_feature_tests.test)

if(${USE_CUDA})
    set_source_files_properties(extractor_tests.cc PROPERTIES LANGUAGE CUDA)
endif()
add_executable(sparsebase_feature_extractor_tests.test extractor_tests.cc)
target_link_libraries(sparsebase_feature_extractor_tests.test sparsebase)
target_link_libraries(sparsebase_feature_extractor_tests.test gtest gtest_main)

add_test(NAME sparsebase_feature_extractor_tests.test COMMAND sparsebase_feature_extractor_tests.test)
