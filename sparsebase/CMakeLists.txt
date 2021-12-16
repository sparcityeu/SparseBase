add_library(sparsebase 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/SparseFormat.cpp 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/SparseReader.cpp 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/SparseObject.cpp 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/SparseConverter.cpp 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/SparseProcess.cpp 
)
target_include_directories(sparsebase PUBLIC include/)
install(TARGETS sparsebase DESTINATION lib)
target_compile_features(sparsebase PRIVATE cxx_std_17)
install(DIRECTORY include/sparsebase DESTINATION include)
