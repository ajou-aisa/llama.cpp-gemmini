if (NOT DEFINED TEST_SOURCE_DIR)
    message(FATAL_ERROR "TEST_SOURCE_DIR is required")
endif()

set(aggregate_sources
    "ggml/src/ggml-gemmini/residual/residual-capture.hpp"
    "ggml/src/ggml-gemmini/ggml-gemmini.cpp"
    "ggml/src/ggml-gemmini/ggml-gemmini-matmul.cpp"
    "ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp"
    "ggml/src/ggml-gemmini/ggml-gemmini-telemetry.cpp"
    "ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp"
    "tools/main/main.cpp")
foreach(relative_path IN LISTS aggregate_sources)
    file(READ "${TEST_SOURCE_DIR}/${relative_path}" source)
    if (source MATCHES "(steady_clock|high_resolution_clock|system_clock)::now[ \t\r\n]*\\(" OR
       source MATCHES "mach_absolute_time[ \t\r\n]*\\(")
        message(FATAL_ERROR
            "Aggregate profiling source ${relative_path} reads a clock directly; route reads through the LOG_CYCLE timer seam")
    endif()
endforeach()

set(timer_seam_sources
    "ggml/src/ggml-gemmini/residual/residual-capture.hpp"
    "ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp")
foreach(relative_path IN LISTS timer_seam_sources)
    file(READ "${TEST_SOURCE_DIR}/${relative_path}" source)
    if (NOT source MATCHES "cycle::timestamp_ns[ \t\r\n]*\\(")
        message(FATAL_ERROR
            "Aggregate profiling source ${relative_path} must use the instrumented cycle::timestamp_ns seam")
    endif()
endforeach()
