# Follow the product linkage topology: shared builds need one process-wide
# logger/counter state image used by both executables and backend libraries.
if(NOT DEFINED CYCLE_SIM)
    set(CYCLE_SIM 0)
endif()
add_library(ggml-gemmini-utils
    src/debug.cpp
    src/cycle.cpp
    src/optrace.cpp
    src/performance.cpp
    src/trace-context.cpp
    src/log-capi.cpp
    src/cycle_reader-capi.cpp
    src/semantic.cpp
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
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../common
)

target_compile_features(ggml-gemmini-utils PUBLIC cxx_std_17)
target_compile_definitions(ggml-gemmini-utils PRIVATE CYCLE_SIM=${CYCLE_SIM})
set(_GEMMINI_SEMANTIC_CPU_ONLY_BUILD 0)
if(GGML_GEMMINI_EXECUTION_BACKEND STREQUAL "HARDWARE" AND
   (NOT GGML_GEMMINI OR GGML_GEMMINI_OPTION STREQUAL "CPU"))
    set(_GEMMINI_SEMANTIC_CPU_ONLY_BUILD 1)
    if(GGML_BACKEND_DL)
        set(_GEMMINI_SEMANTIC_CPU_ONLY_BUILD 0)
    endif()
    if(GGML_GEMMINI AND (GGML_GEMMINI_ALLOW_RUNTIME_MATMUL_OVERRIDE OR
       (GGML_GEMMINI_ENABLE_RMD AND NOT GGML_GEMMINI_DEFAULT_RMD_BACKEND STREQUAL "CPU")))
        set(_GEMMINI_SEMANTIC_CPU_ONLY_BUILD 0)
    endif()
    foreach(_backend IN ITEMS CANN CUDA HIP KOMPUTE METAL MUSA RPC SYCL VULKAN OPENCL)
        if(GGML_${_backend})
            set(_GEMMINI_SEMANTIC_CPU_ONLY_BUILD 0)
        endif()
    endforeach()
endif()
target_compile_definitions(ggml-gemmini-utils PRIVATE
    GEMMINI_SEMANTIC_CPU_ONLY_BUILD=${_GEMMINI_SEMANTIC_CPU_ONLY_BUILD})

if(CYCLE_SIM)
    if(NOT DEFINED _CYCLE_SIM_GENERATE)
        message(FATAL_ERROR "CYCLE_SIM=1 PoTal collection requires the complete llama root build; exported standalone host builds support CYCLE_SIM=0 only")
    endif()
    target_sources(ggml-gemmini-utils PRIVATE
        src/cycle_sim_log.cpp src/cycle_sim_records.cpp src/cycle_sim_stages.cpp)
    add_custom_target(ggml-gemmini-cycle-sim-identities
        COMMAND ${_CYCLE_SIM_GENERATE}
        BYPRODUCTS "${_CYCLE_SIM_GENERATED}/cycle-sim-build-config.hpp"
        VERBATIM)
    add_dependencies(ggml-gemmini-utils ggml-gemmini-cycle-sim-identities)
    target_include_directories(ggml-gemmini-utils PRIVATE
        "${_CYCLE_SIM_GENERATED}")
    target_include_directories(ggml-gemmini-utils PUBLIC "${IM2P_SIM_ROOT}/sim/include")
    target_compile_definitions(ggml-gemmini-utils PRIVATE CYCLE_SIM_BUILD_CONFIG=1)
endif()

# A production trace is bound to the actual producer build, including dirty
# source contents. Other backends retain the disabled-by-default writer API but
# cannot falsely identify themselves as the certified HP1 producer.
if (NOT CYCLE_SIM AND GGML_GEMMINI_EXECUTION_BACKEND STREQUAL "IM2P_SIM" AND
    IM2P_SIM_IMPLEMENTATION STREQUAL "GEMMINI_HP1")
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    get_filename_component(_OPTRACE_LLAMA_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
    set(_OPTRACE_GENERATED "${CMAKE_CURRENT_BINARY_DIR}/optrace-generated")
    set(_OPTRACE_GENERATE
        "${Python3_EXECUTABLE}" "${_OPTRACE_LLAMA_ROOT}/scripts/optrace-build-info.py"
        --llama "${_OPTRACE_LLAMA_ROOT}" --sim "${IM2P_SIM_ROOT}"
        --headers "${_OPTRACE_LLAMA_ROOT}/../RISC-V-DynDNN-gemmini-include"
        --out "${_OPTRACE_GENERATED}"
        --runtime-manifest "${GGML_GEMMINI_IM2P_MANIFEST}"
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
