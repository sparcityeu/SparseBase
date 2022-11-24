project(Tests)

add_subdirectory(lib EXCLUDE_FROM_ALL)

include_directories(${gtest_SOURCE_DIR}/include ${gtest_source_dir})

add_subdirectory(suites/sparsebase/bases)
add_subdirectory(suites/sparsebase/reorder)
add_subdirectory(suites/sparsebase/permute)
add_subdirectory(suites/sparsebase/partition)
add_subdirectory(suites/sparsebase/feature)
add_subdirectory(suites/sparsebase/io)
add_subdirectory(suites/sparsebase/converter)
add_subdirectory(suites/sparsebase/utils/)
add_subdirectory(suites/sparsebase/format)
add_subdirectory(suites/sparsebase/experiment)
