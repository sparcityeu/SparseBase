project(Tests)

add_subdirectory(lib EXCLUDE_FROM_ALL)

include_directories(${gtest_SOURCE_DIR}/include ${gtest_source_dir})

add_subdirectory(suites/sparsebase/preprocess)
add_subdirectory(suites/sparsebase/preprocess/cuda)