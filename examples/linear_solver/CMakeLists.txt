add_executable(linear_solver ${CMAKE_CURRENT_SOURCE_DIR}/linear_solver.cc)
target_link_libraries(linear_solver sparsebase)
#add_library(armadillo SHARED IMPORTED)
#set_target_properties(armadillo PROPERTIES
#        IMPORTED_LOCATION "/home/users/amroa/lib/lib/libarmadillo.so"
#        INTERFACE_INCLUDE_DIRECTORIES "/home/users/amroa/lib/include/")
#target_link_libraries(linear_solver armadillo)