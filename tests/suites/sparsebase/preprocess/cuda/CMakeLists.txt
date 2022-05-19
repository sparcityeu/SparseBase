if(${CUDA})
    add_executable(preprocess_cuda_preprocess_tests.test preprocess_tests.cu)
    target_link_libraries(preprocess_cuda_preprocess_tests.test sparsebase)
    target_link_libraries(preprocess_cuda_preprocess_tests.test gtest gtest_main)

    add_test(NAME preprocess_preprocess_cuda_tests.test COMMAND preprocess_cuda_preprocess_tests.test)
endif()