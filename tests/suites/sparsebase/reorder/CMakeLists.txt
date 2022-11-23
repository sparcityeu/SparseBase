if(${USE_CUDA})
    set_source_files_properties(reorder_tests.cc PROPERTIES LANGUAGE CUDA)
endif()
add_executable(sparsebase_reorder_reorder_tests.test reorder_tests.cc)
target_link_libraries(sparsebase_reorder_reorder_tests.test sparsebase)
target_link_libraries(sparsebase_reorder_reorder_tests.test gtest gtest_main)

add_test(NAME sparsebase_reorder_reorder_tests COMMAND sparsebase_reorder_reorder_tests.test)

if(${USE_CUDA})
    set_source_files_properties(degree_reorder_tests.cc PROPERTIES LANGUAGE CUDA)
endif()
add_executable(sparsebase_reorder_degree_reorder_tests.test degree_reorder_tests.cc)
target_link_libraries(sparsebase_reorder_degree_reorder_tests.test sparsebase)
target_link_libraries(sparsebase_reorder_degree_reorder_tests.test gtest gtest_main)

add_test(NAME sparsebase_reorder_degree_reorder_tests COMMAND sparsebase_reorder_degree_reorder_tests.test)

if(${USE_AMD_ORDER})
    if(${USE_CUDA})
        set_source_files_properties(amd_reorder_tests.cc PROPERTIES LANGUAGE CUDA)
    endif()
    add_executable(sparsebase_reorder_amd_reorder_tests.test amd_reorder_tests.cc)
    target_link_libraries(sparsebase_reorder_amd_reorder_tests.test sparsebase)
    target_link_libraries(sparsebase_reorder_amd_reorder_tests.test gtest gtest_main)

    add_test(NAME sparsebase_reorder_amd_reorder_tests COMMAND sparsebase_reorder_amd_reorder_tests.test)
endif()


if(${USE_RABBIT_ORDER})
    if(${USE_CUDA})
        set_source_files_properties(rabbit_reorder_tests.cc PROPERTIES LANGUAGE CUDA)
    endif()
    add_executable(sparsebase_reorder_rabbit_reorder_tests.test rabbit_reorder_tests.cc)
    target_link_libraries(sparsebase_reorder_rabbit_reorder_tests.test sparsebase)
    target_link_libraries(sparsebase_reorder_rabbit_reorder_tests.test gtest gtest_main)

    add_test(NAME sparsebase_reorder_rabbit_reorder_tests COMMAND sparsebase_reorder_rabbit_reorder_tests.test)
endif()

if(${USE_METIS})
    if(${USE_CUDA})
        set_source_files_properties(metis_reorder_tests.cc PROPERTIES LANGUAGE CUDA)
    endif()
    add_executable(sparsebase_reorder_metis_reorder_tests.test metis_reorder_tests.cc)
    target_link_libraries(sparsebase_reorder_metis_reorder_tests.test sparsebase)
    target_link_libraries(sparsebase_reorder_metis_reorder_tests.test gtest gtest_main)

    add_test(NAME sparsebase_reorder_metis_reorder_tests COMMAND sparsebase_reorder_metis_reorder_tests.test)
endif()

if(${USE_CUDA})
    set_source_files_properties(reorder_heatmap_tests.cc PROPERTIES LANGUAGE CUDA)
endif()
add_executable(sparsebase_reorder_reorder_heatmap_tests.test reorder_heatmap_tests.cc)
target_link_libraries(sparsebase_reorder_reorder_heatmap_tests.test sparsebase)
target_link_libraries(sparsebase_reorder_reorder_heatmap_tests.test gtest gtest_main)

add_test(NAME sparsebase_reorder_reorder_heatmap_tests COMMAND sparsebase_reorder_reorder_heatmap_tests.test)

if(${USE_CUDA})
    set_source_files_properties(rcm_reorder_tests.cc PROPERTIES LANGUAGE CUDA)
endif()
add_executable(sparsebase_reorder_rcm_reorder_tests.test rcm_reorder_tests.cc)
target_link_libraries(sparsebase_reorder_rcm_reorder_tests.test sparsebase)
target_link_libraries(sparsebase_reorder_rcm_reorder_tests.test gtest gtest_main)

add_test(NAME sparsebase_reorder_rcm_reorder_tests COMMAND sparsebase_reorder_rcm_reorder_tests.test)
