
if(${USE_CUDA})
set_source_files_properties(format_tests.cc PROPERTIES LANGUAGE CUDA)
endif()
add_executable(sparsebase_format_format_tests.test format_tests.cc)
target_link_libraries(sparsebase_format_format_tests.test sparsebase)
target_link_libraries(sparsebase_format_format_tests.test gtest gtest_main)

add_test(NAME sparsebase_format_format_tests.test COMMAND sparsebase_format_format_tests.test)

if(${USE_CUDA})
    set_source_files_properties(format_implementation_tests.cc PROPERTIES LANGUAGE CUDA)
endif()
add_executable(sparsebase_format_format_implementation_tests.test format_implementation_tests.cc)
target_link_libraries(sparsebase_format_format_implementation_tests.test sparsebase)
target_link_libraries(sparsebase_format_format_implementation_tests.test gtest gtest_main)

add_test(NAME sparsebase_format_format_implementation_tests.test COMMAND sparsebase_format_format_implementation_tests.test)


if(${USE_CUDA})
    set_source_files_properties(format_order_one_tests.cc PROPERTIES LANGUAGE CUDA)
endif()
add_executable(sparsebase_format_format_order_one_tests.test format_order_one_tests.cc)
target_link_libraries(sparsebase_format_format_order_one_tests.test sparsebase)
target_link_libraries(sparsebase_format_format_order_one_tests.test gtest gtest_main)

add_test(NAME sparsebase_format_format_order_one_tests.test COMMAND sparsebase_format_format_order_one_tests.test)

if(${USE_CUDA})
    set_source_files_properties(format_order_two_tests.cc PROPERTIES LANGUAGE CUDA)
endif()
add_executable(sparsebase_format_format_order_two_tests.test format_order_two_tests.cc)
target_link_libraries(sparsebase_format_format_order_two_tests.test sparsebase)
target_link_libraries(sparsebase_format_format_order_two_tests.test gtest gtest_main)

add_test(NAME sparsebase_format_format_order_two_tests.test COMMAND sparsebase_format_format_order_two_tests.test)

if(${USE_CUDA})
    set_source_files_properties(csr_tests.cc PROPERTIES LANGUAGE CUDA)
endif()
add_executable(sparsebase_format_csr_tests.test csr_tests.cc)
target_link_libraries(sparsebase_format_csr_tests.test sparsebase)
target_link_libraries(sparsebase_format_csr_tests.test gtest gtest_main)

add_test(NAME sparsebase_format_csr_tests.test COMMAND sparsebase_format_csr_tests.test)

if(${USE_CUDA})
    set_source_files_properties(coo_tests.cc PROPERTIES LANGUAGE CUDA)
endif()
add_executable(sparsebase_format_coo_tests.test coo_tests.cc)
target_link_libraries(sparsebase_format_coo_tests.test sparsebase)
target_link_libraries(sparsebase_format_coo_tests.test gtest gtest_main)

add_test(NAME sparsebase_format_coo_tests.test COMMAND sparsebase_format_coo_tests.test)

if(${USE_CUDA})
    set_source_files_properties(csc_tests.cc PROPERTIES LANGUAGE CUDA)
endif()
add_executable(sparsebase_format_csc_tests.test csc_tests.cc)
target_link_libraries(sparsebase_format_csc_tests.test sparsebase)
target_link_libraries(sparsebase_format_csc_tests.test gtest gtest_main)

add_test(NAME sparsebase_format_csc_tests.test COMMAND sparsebase_format_csc_tests.test)

if(${USE_CUDA})
    set_source_files_properties(array_tests.cc PROPERTIES LANGUAGE CUDA)
endif()
add_executable(sparsebase_format_array_tests.test array_tests.cc)
target_link_libraries(sparsebase_format_array_tests.test sparsebase)
target_link_libraries(sparsebase_format_array_tests.test gtest gtest_main)

add_test(NAME sparsebase_format_array_tests.test COMMAND sparsebase_format_array_tests.test)
