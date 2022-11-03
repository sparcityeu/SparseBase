project(Tests)

add_subdirectory(lib EXCLUDE_FROM_ALL)

include_directories(${gtest_SOURCE_DIR}/include ${gtest_source_dir})

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/data DESTINATION ${CMAKE_BINARY_DIR}/tests/)

add_subdirectory(suites/sparsebase/preprocess)
add_subdirectory(suites/sparsebase/preprocess/cuda)
add_subdirectory(suites/sparsebase/feature)
add_subdirectory(suites/sparsebase/utils/io)
add_subdirectory(suites/sparsebase/utils/converter)
add_subdirectory(suites/sparsebase/utils/converter/cuda)
add_subdirectory(suites/sparsebase/format)
add_subdirectory(suites/sparsebase/experiment)
