# Follow the product linkage topology: shared builds need one process-wide
# logger/counter state image used by both executables and backend libraries.
add_library(ggml-gemmini-utils
    src/debug.cpp
    src/cycle.cpp
    src/cycle-serialization.cpp
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
