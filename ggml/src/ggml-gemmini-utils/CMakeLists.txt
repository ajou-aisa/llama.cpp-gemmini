# Follow the product linkage topology: shared builds need one process-wide
# logger/counter state image used by both executables and backend libraries.
add_library(ggml-gemmini-utils
    src/debug.cpp
    src/cycle.cpp
    src/optrace.cpp
    src/log-capi.cpp
    src/cycle_reader-capi.cpp
)
if (CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
    target_sources(ggml-gemmini-utils PRIVATE src/cycle_reader_aarch64.cpp)
endif()
set_target_properties(ggml-gemmini-utils PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    WINDOWS_EXPORT_ALL_SYMBOLS ON)
install(TARGETS ggml-gemmini-utils LIBRARY ARCHIVE RUNTIME)

target_include_directories(ggml-gemmini-utils
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(ggml-gemmini-utils PUBLIC cxx_std_17)

# A production trace is bound to the actual producer build, including dirty
# source contents. Other backends retain the disabled-by-default writer API but
# cannot falsely identify themselves as the certified HP1 producer.
if (GGML_GEMMINI_EXECUTION_BACKEND STREQUAL "IM2P_SIM" AND
    IM2P_SIM_IMPLEMENTATION STREQUAL "GEMMINI_HP1")
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    get_filename_component(_OPTRACE_LLAMA_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
    set(_OPTRACE_GENERATED "${CMAKE_CURRENT_BINARY_DIR}/optrace-generated")
    set(_OPTRACE_GENERATE
        "${Python3_EXECUTABLE}" "${_OPTRACE_LLAMA_ROOT}/scripts/optrace-build-info.py"
        --llama "${_OPTRACE_LLAMA_ROOT}" --sim "${IM2P_SIM_ROOT}"
        --headers "${_OPTRACE_LLAMA_ROOT}/../RISC-V-DynDNN-gemmini-include"
        --out "${_OPTRACE_GENERATED}"
        --bits "${GGML_GEMMINI_ACTIVATION_BITS}" --dim "${GGML_GEMMINI_DIM}"
        --mode "${GGML_GEMMINI_DEFAULT_MATMUL_MODE}"
        --rmd "${GGML_GEMMINI_ENABLE_RMD}"
        --rmd-backend "${GGML_GEMMINI_DEFAULT_RMD_BACKEND}")
    execute_process(COMMAND ${_OPTRACE_GENERATE} RESULT_VARIABLE _OPTRACE_RESULT
        ERROR_VARIABLE _OPTRACE_IDENTITY_ERROR)
    if (_OPTRACE_RESULT EQUAL 0)
    add_custom_target(ggml-gemmini-optrace-identities
        COMMAND ${_OPTRACE_GENERATE}
        BYPRODUCTS "${_OPTRACE_GENERATED}/optrace-build-config.hpp"
                   "${_OPTRACE_GENERATED}/source-identities.json"
        VERBATIM)
    add_dependencies(ggml-gemmini-utils ggml-gemmini-optrace-identities)
    target_include_directories(ggml-gemmini-utils PRIVATE "${_OPTRACE_GENERATED}")
    target_compile_definitions(ggml-gemmini-utils PRIVATE GEMMINI_OPTRACE_BUILD_CONFIG=1)
    else()
        # Archive distributions and build-contract fixtures may have no Git
        # source identity. Preserve the ordinary trace-off build; explicit
        # tracing fails closed in compiled_run_info() rather than inventing it.
        message(STATUS "Production op-trace source identities unavailable; explicit tracing will fail")
    endif()
endif()

if (DEFINED GGML_GEMMINI_UTILS_COMPILE_DEFS AND NOT "${GGML_GEMMINI_UTILS_COMPILE_DEFS}" STREQUAL "")
    target_compile_definitions(ggml-gemmini-utils PRIVATE ${GGML_GEMMINI_UTILS_COMPILE_DEFS})
else()
    target_compile_definitions(ggml-gemmini-utils
        PRIVATE
            LOG_DEBUG=1
            LOG_CYCLE=1
            LOG_DUMP=0
            LOG_DUMP_SCALE=0
            GEMMINI_VERIFY_DEQ=1
    )
endif()
