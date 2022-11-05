if(${USE_CUDA})
    add_executable(sparsebase_utils_converter_cuda_tests.test converter_tests.cu)
    target_link_libraries(sparsebase_utils_converter_cuda_tests.test sparsebase)
    target_link_libraries(sparsebase_utils_converter_cuda_tests.test gtest gtest_main)

    add_test(NAME sparsebase_utils_converter_cuda_tests.test COMMAND sparsebase_utils_converter_cuda_tests.test)
endif()
