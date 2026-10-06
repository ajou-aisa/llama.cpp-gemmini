# Gemmini include path (needed for gemmini_params.h in ggml-gemmini-args.h)
if (NOT DEFINED GEMMINI_SW_PATH OR GEMMINI_SW_PATH STREQUAL "")
    get_filename_component(GEMMINI_SW_PATH "${_GGML_GEMMINI_SOURCE_DIR}/../RISC-V-DynDNN-gemmini-include" ABSOLUTE)
elseif(GGML_GEMMINI AND NOT EXISTS "${GEMMINI_SW_PATH}/gemmini.h" AND
       NOT EXISTS "${GEMMINI_SW_PATH}/include/gemmini.h")
    message(FATAL_ERROR "Invalid explicit GEMMINI_SW_PATH='${GEMMINI_SW_PATH}': Gemmini include tree not found")
endif()
if (EXISTS "${GEMMINI_SW_PATH}/gemmini.h" OR EXISTS "${GEMMINI_SW_PATH}/include/gemmini.h")
    get_filename_component(GEMMINI_SW_PATH "${GEMMINI_SW_PATH}" REALPATH)
    include_directories(${GEMMINI_SW_PATH} ${GEMMINI_SW_PATH}/include)
else()
    unset(GEMMINI_SW_PATH)
endif()

set(GGML_GEMMINI_GENERATED_CONFIG_DIR "${_GGML_GEMMINI_BINARY_DIR}/generated")
file(MAKE_DIRECTORY "${GGML_GEMMINI_GENERATED_CONFIG_DIR}")
if(GGML_GEMMINI)
    if(NOT GEMMINI_SW_PATH)
        message(FATAL_ERROR "GGML_GEMMINI requires the Gemmini include tree")
    endif()
    set(_GGML_GEMMINI_PARAMS_SOURCE "${GEMMINI_SW_PATH}/gemmini_params.h")
    if(NOT EXISTS "${_GGML_GEMMINI_PARAMS_SOURCE}")
        message(FATAL_ERROR
            "Missing Gemmini parameter header: ${_GGML_GEMMINI_PARAMS_SOURCE}")
    endif()
    file(READ "${_GGML_GEMMINI_PARAMS_SOURCE}" _GGML_GEMMINI_PARAMS_CONTENT)
    string(REGEX MATCH "#define[ \t]+DIM[ \t]+([0-9]+)"
        _GGML_GEMMINI_PHYSICAL_DIM_MATCH "${_GGML_GEMMINI_PARAMS_CONTENT}")
    if(NOT _GGML_GEMMINI_PHYSICAL_DIM_MATCH)
        message(FATAL_ERROR
            "Gemmini parameter header does not define a numeric DIM")
    endif()
    set(_GGML_GEMMINI_PHYSICAL_DIM "${CMAKE_MATCH_1}")
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "riscv" AND NOT CYCLE_SIM)
        if(NOT _GGML_GEMMINI_PHYSICAL_DIM STREQUAL "${GGML_GEMMINI_DIM}")
            message(FATAL_ERROR
                "Physical Gemmini DIM mismatch: hardware header reports ${_GGML_GEMMINI_PHYSICAL_DIM}, build requests ${GGML_GEMMINI_DIM}")
        endif()
    else()
        string(REGEX REPLACE "#define[ \t]+DIM[ \t]+[0-9]+"
            "#define DIM ${GGML_GEMMINI_DIM}" _GGML_GEMMINI_PARAMS_CONTENT
            "${_GGML_GEMMINI_PARAMS_CONTENT}")
        if(GGML_GEMMINI_ACTIVATION_BITS EQUAL 16)
            string(REGEX REPLACE
                "#define[ \t]+MAX_BLOCK_LEN[ \t]+\\(MAX_BYTES/\\(DIM\\*1\\)\\)"
                "#define MAX_BLOCK_LEN (MAX_BYTES/(DIM*2))"
                _GGML_GEMMINI_PARAMS_CONTENT
                "${_GGML_GEMMINI_PARAMS_CONTENT}")
            string(REGEX REPLACE "typedef[ \t]+int8_t[ \t]+elem_t;"
                "typedef int16_t elem_t;"
                _GGML_GEMMINI_PARAMS_CONTENT
                "${_GGML_GEMMINI_PARAMS_CONTENT}")
            string(REGEX REPLACE
                "static const elem_t elem_t_max = 127;"
                "static const elem_t elem_t_max = 32767;"
                _GGML_GEMMINI_PARAMS_CONTENT
                "${_GGML_GEMMINI_PARAMS_CONTENT}")
            string(REGEX REPLACE
                "static const elem_t elem_t_min = -128;"
                "static const elem_t elem_t_min = -32768;"
                _GGML_GEMMINI_PARAMS_CONTENT
                "${_GGML_GEMMINI_PARAMS_CONTENT}")
            string(REPLACE "INT8_MAX" "INT16_MAX"
                _GGML_GEMMINI_PARAMS_CONTENT
                "${_GGML_GEMMINI_PARAMS_CONTENT}")
            string(REPLACE "INT8_MIN" "INT16_MIN"
                _GGML_GEMMINI_PARAMS_CONTENT
                "${_GGML_GEMMINI_PARAMS_CONTENT}")
        endif()
    endif()
    file(WRITE "${GGML_GEMMINI_GENERATED_CONFIG_DIR}/gemmini_params.h"
        "${_GGML_GEMMINI_PARAMS_CONTENT}")
    include_directories(BEFORE "${GGML_GEMMINI_GENERATED_CONFIG_DIR}")
endif()
