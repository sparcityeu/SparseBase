if(${USE_CUDA})
    add_executable(sparsebase_preprocess_cuda_preprocess_tests.test preprocess_tests.cu)
    target_link_libraries(sparsebase_preprocess_cuda_preprocess_tests.test sparsebase)
    target_link_libraries(sparsebase_preprocess_cuda_preprocess_tests.test gtest gtest_main)

    add_test(NAME sparsebase_preprocess_cuda_preprocess_tests.test COMMAND sparsebase_preprocess_cuda_preprocess_tests.test)
endif()