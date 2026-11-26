# Gemmini parameter header. cmake/npu-config.cmake generates gemmini_params.h
# from the NPU_* build parameters into ${NPU_GENERATED_DIR}; no external
# Gemmini SDK is read. Only the FP32 overlay still takes an
# explicit -DGEMMINI_SW_PATH (its own SDK overlay).
set(GGML_GEMMINI_GENERATED_CONFIG_DIR "${_GGML_GEMMINI_BINARY_DIR}/generated")
file(MAKE_DIRECTORY "${GGML_GEMMINI_GENERATED_CONFIG_DIR}")
if (DEFINED GEMMINI_SW_PATH AND NOT GEMMINI_SW_PATH STREQUAL "")
    if (NOT EXISTS "${GEMMINI_SW_PATH}/gemmini.h" AND NOT EXISTS "${GEMMINI_SW_PATH}/include/gemmini.h")
        message(FATAL_ERROR "Invalid explicit GEMMINI_SW_PATH='${GEMMINI_SW_PATH}': Gemmini include tree not found")
    endif()
    get_filename_component(GEMMINI_SW_PATH "${GEMMINI_SW_PATH}" REALPATH)
    # Experiment lanes only (npu-config.cmake rejects the path elsewhere).
    include_directories(${GEMMINI_SW_PATH} ${GEMMINI_SW_PATH}/include)
else()
    unset(GEMMINI_SW_PATH)
endif()
if (GGML_GEMMINI)
    if (GGML_GEMMINI_FP32)
        if (NOT GEMMINI_SW_PATH)
            message(FATAL_ERROR "GGML_GEMMINI_FP32 requires -DGEMMINI_SW_PATH=<FP32 DIM8 SDK overlay>")
        endif()
    elseif (NOT GGML_NPU STREQUAL "GEMMINI" OR NOT EXISTS "${GGML_GEMMINI_GENERATED_CONFIG_DIR}/gemmini_params.h")
        message(FATAL_ERROR "GGML_GEMMINI=ON needs the parameter header from cmake/npu-config.cmake")
    endif()
    # Every target of the build sees the generated header first, as before.
    include_directories(BEFORE "${GGML_GEMMINI_GENERATED_CONFIG_DIR}")
endif()
