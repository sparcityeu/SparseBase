macro(generate_explicit_instantiations)
  find_package (Python COMPONENTS Interpreter)
  list(JOIN ID_TYPES "," ID_TYPES_JOINED)
  list(JOIN NNZ_TYPES "," NNZ_TYPES_JOINED)
  list(JOIN VALUE_TYPES "," VALUE_TYPES_JOINED)
  list(JOIN FLOAT_TYPES "," FLOAT_TYPES_JOINED)
  execute_process(COMMAND ${Python_EXECUTABLE}
    ${CMAKE_SOURCE_DIR}/src/generate_explicit_instantiations.py
    --id-types ${ID_TYPES_JOINED}
    --nnz-types ${NNZ_TYPES_JOINED}
    --value-types ${VALUE_TYPES_JOINED}
    --float-types ${FLOAT_TYPES_JOINED}
    --pigo ${USE_PIGO}
    --cuda ${USE_CUDA}
    --output-folder ${PROJECT_BINARY_DIR}/init)
  message(STATUS "Generated explicit instantiations")
endmacro()