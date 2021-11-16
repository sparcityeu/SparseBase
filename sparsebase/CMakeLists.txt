add_library(sparsebase 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/SparseFormat.cpp 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/SparseReader.cpp 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/SparseObject.cpp 
)
target_include_directories(sparsebase PUBLIC include/)
install(TARGETS sparsebase DESTINATION lib)
install(DIRECTORY include/sparsebase DESTINATION include)