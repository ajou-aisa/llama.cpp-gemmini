if (NOT DEFINED TEST_SOURCE_DIR)
    message(FATAL_ERROR "TEST_SOURCE_DIR is required")
endif()

set(aggregate_sources
    "ggml/src/ggml-npu-common/residual/residual-capture.hpp"
    "ggml/src/ggml-gemmini/ggml-gemmini.cpp"
    "ggml/src/ggml-gemmini/ops.cpp"
    "ggml/src/ggml-gemmini/matmul/dense.cpp"
    "ggml/src/ggml-gemmini/matmul/execution.cpp"
    "ggml/src/ggml-npu-common/matmul/options.cpp"
    "ggml/src/ggml-gemmini/matmul/types.hpp"
    "ggml/src/ggml-gemmini/matmul/detail.hpp"
    "ggml/src/ggml-npu-common/profiling/matmul-types.hpp"
    "ggml/src/ggml-npu-common/profiling/matmul-detail.hpp"
    "ggml/src/ggml-npu-common/profiling/telemetry.cpp"
    "ggml/src/ggml-npu-common/quantization/activation/exsia/exsia.cpp"
    "ggml/src/ggml-npu-common/quantization/activation/exsia/exsia-profile.cpp"
    "ggml/src/ggml-npu-common/quantization/activation/exsia/local.cpp"
    "ggml/src/ggml-npu-common/quantization/activation/exsia/folding.cpp"
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
    "ggml/src/ggml-npu-common/residual/residual-capture.hpp"
    "ggml/src/ggml-npu-common/quantization/activation/exsia/exsia-profile.cpp")
foreach(relative_path IN LISTS timer_seam_sources)
    file(READ "${TEST_SOURCE_DIR}/${relative_path}" source)
    if (NOT source MATCHES "cycle::timestamp_ns[ \t\r\n]*\\(")
        message(FATAL_ERROR
            "Aggregate profiling source ${relative_path} must use the instrumented cycle::timestamp_ns seam")
    endif()
endforeach()
