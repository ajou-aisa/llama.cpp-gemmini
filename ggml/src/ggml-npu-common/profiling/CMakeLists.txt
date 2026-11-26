# Follow the product linkage topology: shared builds need one process-wide
# logger/counter state image used by both executables and backend libraries.
add_library(ggml-npu-profiling
    src/evaluation_metrics.cpp
    src/debug.cpp
    src/cycle.cpp
    src/performance.cpp
    src/trace-context.cpp
    src/log-capi.cpp
    src/cycle_reader-capi.cpp
    src/optrace.cpp
    src/semantic.cpp
)
if (CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
    target_sources(ggml-npu-profiling PRIVATE src/cycle_reader_aarch64.cpp)
endif()
set_target_properties(ggml-npu-profiling PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    WINDOWS_EXPORT_ALL_SYMBOLS ON)
install(TARGETS ggml-npu-profiling LIBRARY ARCHIVE RUNTIME)

target_include_directories(ggml-npu-profiling
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../../common
)

target_compile_features(ggml-npu-profiling PUBLIC cxx_std_17)
if(GGML_GEMMINI_ACT_QUANT_METRICS OR GGML_GEMMINI_RESIDUAL_METRICS OR GGML_GEMMINI_SCALE_METRICS)
    target_compile_definitions(ggml-npu-profiling PRIVATE
        GGML_GEMMINI_ACTIVATION_BITS=${GGML_GEMMINI_ACTIVATION_BITS}
        GGML_GEMMINI_WEIGHT_BITS=${GGML_GEMMINI_WEIGHT_BITS}
        GGML_GEMMINI_DIM=${GGML_GEMMINI_DIM})
endif()
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
    # POTAL executes MUL_MAT on the PoTal accelerator: not a CPU-only build.
    foreach(_backend IN ITEMS CANN CUDA HIP KOMPUTE METAL MUSA RPC SYCL VULKAN OPENCL POTAL)
        if(GGML_${_backend})
            set(_GEMMINI_SEMANTIC_CPU_ONLY_BUILD 0)
        endif()
    endforeach()
endif()
target_compile_definitions(ggml-npu-profiling PRIVATE
    GEMMINI_SEMANTIC_CPU_ONLY_BUILD=${_GEMMINI_SEMANTIC_CPU_ONLY_BUILD})


# Trace source identities are generated independently of metric collection.

if (DEFINED GGML_GEMMINI_UTILS_COMPILE_DEFS AND NOT "${GGML_GEMMINI_UTILS_COMPILE_DEFS}" STREQUAL "")
    target_compile_definitions(ggml-npu-profiling PRIVATE ${GGML_GEMMINI_UTILS_COMPILE_DEFS})
else()
    target_compile_definitions(ggml-npu-profiling
        PRIVATE
            LOG_DEBUG=1
            LOG_CYCLE=1
            LOG_DUMP=0
            LOG_DUMP_SCALE=0
            GEMMINI_VERIFY_DEQ=1
    )
endif()

# Dispatch provenance is independent of cycle telemetry and simulator providers.
find_package(Python3 QUIET COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND AND DEFINED GGML_GEMMINI_ACTIVATION_BITS AND DEFINED GGML_GEMMINI_DIM)
    get_filename_component(_OPTRACE_SOURCE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../.." ABSOLUTE)
    set(_OPTRACE_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/optrace")
    set(_OPTRACE_BACKEND "${GGML_NPU}")
    if(NOT NPU_TRANSPORT STREQUAL "")
        string(APPEND _OPTRACE_BACKEND "/${NPU_TRANSPORT}")
    endif()
    add_custom_target(gemmini-optrace-build-info
        COMMAND "${Python3_EXECUTABLE}" -B "${_OPTRACE_SOURCE_ROOT}/scripts/optrace-build-info.py"
            --llama "${_OPTRACE_SOURCE_ROOT}" --headers "${_OPTRACE_SOURCE_ROOT}/ggml/src/ggml-gemmini"
            --out "${_OPTRACE_BUILD_DIR}" --cache "${CMAKE_BINARY_DIR}/CMakeCache.txt"
            --backend "${_OPTRACE_BACKEND}" --bits "${GGML_GEMMINI_ACTIVATION_BITS}"
            --dim "${GGML_GEMMINI_DIM}" --mode "${GGML_GEMMINI_DEFAULT_MATMUL_MODE}"
            --rmd "$<IF:$<BOOL:${GGML_GEMMINI_ENABLE_RMD}>,ON,OFF>"
        BYPRODUCTS "${_OPTRACE_BUILD_DIR}/optrace-build-config.hpp" "${_OPTRACE_BUILD_DIR}/source-identities.json"
        VERBATIM)
    add_dependencies(ggml-npu-profiling gemmini-optrace-build-info)
    target_include_directories(ggml-npu-profiling PRIVATE "${_OPTRACE_BUILD_DIR}")
    target_compile_definitions(ggml-npu-profiling PRIVATE GEMMINI_OPTRACE_BUILD_CONFIG=1)
endif()
