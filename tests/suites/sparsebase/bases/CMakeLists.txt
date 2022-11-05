if(${USE_CUDA})
    set_source_files_properties(iobase_tests.cc PROPERTIES LANGUAGE CUDA)
endif()
add_executable(sparsebase_bases_iobase_tests.test iobase_tests.cc)
target_link_libraries(sparsebase_bases_iobase_tests.test  sparsebase)
target_link_libraries(sparsebase_bases_iobase_tests.test  gtest gtest_main)
add_test(NAME sparsebase_bases_iobase_tests.test COMMAND sparsebase_bases_iobase_tests.test)

