macro(generate_instantiations)
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
            --output-folder ${PROJECT_BINARY_DIR}/init
            --class-list ${CMAKE_SOURCE_DIR}/src/class_instantiation_list.json)

    message(STATUS "Generated explicit instantiations")

endmacro()