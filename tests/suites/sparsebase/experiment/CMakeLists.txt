if(${USE_CUDA})
set_source_files_properties(experiment_tests.cc PROPERTIES LANGUAGE CUDA)
endif()
add_executable(sparsebase_experiment_tests.test experiment_tests.cc)
target_link_libraries(sparsebase_experiment_tests.test sparsebase)
target_link_libraries(sparsebase_experiment_tests.test gtest gtest_main)

add_test(NAME sparsebase_experiment_tests COMMAND sparsebase_experiment_tests.test)
