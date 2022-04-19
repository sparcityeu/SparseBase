if(${CUDA})
    add_executable(preprocess_cuda_preprocess_test.test preprocess_test.cu)
    target_link_libraries(preprocess_cuda_preprocess_test.test sparsebase)
    target_link_libraries(preprocess_cuda_preprocess_test.test gtest gtest_main)

    add_test(NAME preprocess_preprocess_cuda_test.test COMMAND preprocess_cuda_preprocess_test.test)
endif()