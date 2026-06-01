add_library(ggml-gemmini-utils STATIC
    src/debug.cpp
    src/cycle.cpp
    src/log-capi.cpp
    src/cycle_reader-capi.cpp
)

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
            ERROR_COMPENSATION=0
            GEMMINI_VERIFY_DEQ=1
    )
endif()
