# Creating the sparsebase library executable
add_library(sparsebase 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sparse_format.cpp 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sparse_reader.cpp 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sparse_object.cpp 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sparse_converter.cpp 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sparse_preprocess.cpp 
)
target_compile_features(sparsebase PRIVATE cxx_std_17)

# defining the include paths of the library
target_include_directories(sparsebase PUBLIC 
    # Location of the headers at build time 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    # Location of the headers at install time 
    $<INSTALL_INTERFACE:include>)

# Creating an alias for the library when used through "find_package()"
add_library(sparsebase::sparsebase ALIAS sparsebase)
# Installing the headers to the include directory
install(DIRECTORY include/sparsebase DESTINATION include)
include(GNUInstallDirs)

# Installing the library, and creating the export set "sparsebaseTargets" 
install(TARGETS sparsebase EXPORT sparsebaseTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    # telling the export set where the headers of the library are located
    INCLUDES DESTINATION include
    )
