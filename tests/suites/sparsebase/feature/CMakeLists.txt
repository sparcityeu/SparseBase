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

if(${USE_CUDA})
    set_source_files_properties(degrees_tests.cc PROPERTIES LANGUAGE CUDA)
endif()
add_executable(sparsebase_feature_degrees_tests.test degrees_tests.cc)
target_link_libraries(sparsebase_feature_degrees_tests.test sparsebase)
target_link_libraries(sparsebase_feature_degrees_tests.test gtest gtest_main)

add_test(NAME sparsebase_feature_degrees_tests.test COMMAND sparsebase_feature_degrees_tests.test)

if(${USE_CUDA})
    set_source_files_properties(degree_distribution_tests.cc PROPERTIES LANGUAGE CUDA)
endif()
add_executable(sparsebase_feature_degree_distribution_tests.test degree_distribution_tests.cc)
target_link_libraries(sparsebase_feature_degree_distribution_tests.test sparsebase)
target_link_libraries(sparsebase_feature_degree_distribution_tests.test gtest gtest_main)

add_test(NAME sparsebase_feature_degree_distribution_tests.test COMMAND sparsebase_feature_degree_distribution_tests.test)

if(${USE_CUDA})
    set_source_files_properties(degrees_degree_distribution_tests.cc PROPERTIES LANGUAGE CUDA)
endif()
add_executable(sparsebase_feature_degrees_degree_distribution_tests.test degrees_degree_distribution_tests.cc)
target_link_libraries(sparsebase_feature_degrees_degree_distribution_tests.test sparsebase)
target_link_libraries(sparsebase_feature_degrees_degree_distribution_tests.test gtest gtest_main)

add_test(NAME sparsebase_feature_degrees_degree_distribution_tests.test COMMAND sparsebase_feature_degrees_degree_distribution_tests.test)

if(${USE_CUDA})
    set_source_files_properties(jaccard_weights_tests.cc PROPERTIES LANGUAGE CUDA)
endif()
add_executable(sparsebase_feature_jaccard_weights_tests.test jaccard_weights_tests.cc)
target_link_libraries(sparsebase_feature_jaccard_weights_tests.test sparsebase)
target_link_libraries(sparsebase_feature_jaccard_weights_tests.test gtest gtest_main)

add_test(NAME sparsebase_feature_jaccard_weights_tests.test COMMAND sparsebase_feature_jaccard_weights_tests.test)

if(${USE_CUDA})
    set_source_files_properties(min_degree_tests.cc PROPERTIES LANGUAGE CUDA)
endif()
add_executable(sparsebase_feature_min_degree_tests.test min_degree_tests.cc)
target_link_libraries(sparsebase_feature_min_degree_tests.test sparsebase)
target_link_libraries(sparsebase_feature_min_degree_tests.test gtest gtest_main)

add_test(NAME sparsebase_feature_min_degree_tests.test COMMAND sparsebase_feature_min_degree_tests.test)

if(${USE_CUDA})
    set_source_files_properties(max_degree_tests.cc PROPERTIES LANGUAGE CUDA)
endif()
add_executable(sparsebase_feature_max_degree_tests.test max_degree_tests.cc)
target_link_libraries(sparsebase_feature_max_degree_tests.test sparsebase)
target_link_libraries(sparsebase_feature_max_degree_tests.test gtest gtest_main)

add_test(NAME sparsebase_feature_max_degree_tests.test COMMAND sparsebase_feature_max_degree_tests.test)

if(${USE_CUDA})
    set_source_files_properties(avg_degree_tests.cc PROPERTIES LANGUAGE CUDA)
endif()
add_executable(sparsebase_feature_avg_degree_tests.test avg_degree_tests.cc)
target_link_libraries(sparsebase_feature_avg_degree_tests.test sparsebase)
target_link_libraries(sparsebase_feature_avg_degree_tests.test gtest gtest_main)

add_test(NAME sparsebase_feature_avg_degree_tests.test COMMAND sparsebase_feature_avg_degree_tests.test)

if(${USE_CUDA})
    set_source_files_properties(min_max_avg_degree_tests.cc PROPERTIES LANGUAGE CUDA)
endif()
add_executable(sparsebase_feature_min_max_avg_degree_tests.test min_max_avg_degree_tests.cc)
target_link_libraries(sparsebase_feature_min_max_avg_degree_tests.test sparsebase)
target_link_libraries(sparsebase_feature_min_max_avg_degree_tests.test gtest gtest_main)

add_test(NAME sparsebase_feature_min_max_avg_degree_tests.test COMMAND sparsebase_feature_min_max_avg_degree_tests.test)