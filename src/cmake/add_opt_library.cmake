cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

# Macro for adding optional libraries to sparsebase
# Example : add_opt_library("metis" ${METIS_LIB_DIR} ${METIS_INC_DIR})
macro(add_opt_library name)

  string(TOUPPER ${name} OPT_NAME_UPPER)
  string(TOLOWER ${name} OPT_NAME_LOWER)

  set(OPT_LIB "LIB-NOTFOUND")
  set(OPT_INC "INC-NOTFOUND")
  set(OPT_DEF USE_${OPT_NAME_UPPER})

  set(${OPT_NAME_UPPER}_INC_DIR "" CACHE STRING "Include directory for ${OPT_NAME_UPPER}")
  set(${OPT_NAME_UPPER}_LIB_DIR "" CACHE STRING "Library directory for ${OPT_NAME_UPPER}")

  find_library(
          OPT_LIB
          NAMES ${OPT_NAME_LOWER} lib${OPT_NAME_LOWER}
          PATH_SUFFIXES lib lib32 lib64 lib{OPT_NAME_LOWER}
          PATHS
            ${${OPT_NAME_UPPER}_LIB_DIR}
            $ENV{${OPT_NAME_UPPER}_LIB_DIR}
            ${CMAKE_PREFIX_PATH}
            ${CMAKE_SYSTEM_PREFIX_PATH}
          NO_DEFAULT_PATH

  )

  if(NOT OPT_LIB)
    message(FATAL_ERROR "${OPT_NAME_UPPER} library file was not found. "
            "Try setting the ${OPT_NAME_UPPER}_LIB_DIR environment variable")
  endif()


  find_path(
          OPT_INC
          NAMES ${OPT_NAME_LOWER}.h ${OPT_NAME_LOWER}.hpp ${OPT_NAME_LOWER}.hxx
          PATH_SUFFIXES include
          PATHS
            ${${OPT_NAME_UPPER}_INC_DIR}
            $ENV{${OPT_NAME_UPPER}_INC_DIR}
            ${CMAKE_PREFIX_PATH}
            ${CMAKE_SYSTEM_PREFIX_PATH}
          NO_DEFAULT_PATH
  )

  if(NOT OPT_INC)
    message(FATAL_ERROR "${OPT_NAME_UPPER} include file was not found. "
            "Try setting the ${OPT_NAME_UPPER}_INC_DIR environment variable")
  endif()


  message(STATUS "${OPT_NAME_UPPER} was found")
  message(STATUS "${OPT_NAME_UPPER} include dir: ${OPT_INC}")
  message(STATUS "${OPT_NAME_UPPER} library dir: ${OPT_LIB}")
  target_link_libraries(sparsebase PUBLIC ${OPT_LIB})
  target_include_directories(sparsebase PUBLIC ${OPT_INC})
  file(GLOB OPT_INC_FILES ${inc_dir}/*.h ${inc_dir}/*.hpp ${inc_dir}/*.hxx)
  install(FILES ${OPT_INC_FILES} DESTINATION include/sparsebase/external/${lib_name})
  set(${OPT_DEF} ON)
endmacro()