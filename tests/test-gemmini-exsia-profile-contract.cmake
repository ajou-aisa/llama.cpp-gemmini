if(NOT DEFINED EXSIA_SOURCE OR NOT DEFINED EXSIA_HEADER)
    message(FATAL_ERROR "ExSIA profile contract requires source and header")
endif()
file(READ "${EXSIA_SOURCE}" source)
file(READ "${EXSIA_HEADER}" header)

function(require_count text pattern expected label)
    string(REGEX MATCHALL "${pattern}" matches "${text}")
    list(LENGTH matches actual)
    if(NOT actual EQUAL expected)
        message(FATAL_ERROR "${label}: expected ${expected}, found ${actual}")
    endif()
endfunction()

# Jetson profile intervals retain one native start/end sample; all other targets
# retain the original scalar cycle::read path and scalar ordering behavior.
require_count("${source}" "interval\\.start_sample = ggml::gemmini::cycle::read_sample\\(\\)" 1
    "profile start performs one native sample read")
require_count("${source}" "interval\\.end_sample = ggml::gemmini::cycle::read_sample\\(\\)" 1
    "profile end performs one native sample read")
require_count("${source}" "interval\\.start = profile_now\\(\\)" 1
    "non-Jetson profile start retains baseline scalar reader path")
require_count("${source}" "interval\\.end = profile_now\\(\\)" 1
    "non-Jetson profile end retains baseline scalar reader path")
require_count("${source}"
    "#if !defined\\(__linux__\\)[ ]+[|][|][ ]+!defined\\(__aarch64__\\)[\n\r ]+uint64_t profile_now\\(\\)"
    1 "scalar profile helper is excluded only from Linux AArch64")
require_count("${source}" "return interval\\.end >= interval\\.start" 1
    "non-Jetson profile validity retains scalar ordering")

# Local/P0-P3, Mask, Exponent, and Folding remain statistics with individual
# native provenance. They are diagnostics, never QuantizeWork inputs.
require_count("${source}" "const auto t[0-4] = EXSIA_STAGE_CYCLE_READ\\(\\)" 15
    "Linux-AArch64 P0-P3 paths share five native boundaries")
require_count("${source}" "const uint64_t t[0-4] = EXSIA_STAGE_CYCLE_READ\\(\\)" 15
    "non-Jetson P0-P3 paths retain scalar reads")
require_count("${source}" "record_stage_cycles\\(cycle_sample, \\{t0, t1, t2, t3, t4\\}\\)" 3
    "each native P0-P3 path consumes its shared boundaries")
foreach(stage RANGE 0 3)
    math(EXPR next "${stage} + 1")
    require_count("${source}"
        "cycle_sample\\.p${stage} = t${next} >= t${stage} \\? t${next} - t${stage} : 0"
        3 "non-Jetson P${stage} retains inline scalar subtraction")
endforeach()

require_count("${header}"
    "sum \\+= value;[\n\r ]*max = std::max\\(max, value\\);[\n\r ]*\\+\\+count"
    3 "P0-P3 statistics retain simple scalar accumulation")
if(NOT source MATCHES
    "for \\(size_t (worker|group) = 0; (worker|group) < profile\\.local_groups\\.size\\(\\); \\+\\+(worker|group)\\)")
    message(FATAL_ERROR "profile output must iterate configured workers individually")
endif()

foreach(forbidden IN ITEMS
        "QuantizeProfileCycles" "aggregate_quantize_profile" "checked_sum"
        "quantize_cpu_work" "arithmetic_overflow"
        "local_worker_start_ns" "local_worker_end_ns"
        "local_group3_start_cycle" "local_group3_end_cycle")
    if(source MATCHES "${forbidden}" OR header MATCHES "${forbidden}")
        message(FATAL_ERROR "ExSIA individual-provenance contract rejects ${forbidden}")
    endif()
endforeach()

if(NOT header MATCHES "std::array<ProfileInterval, EXSIA_LOCAL_WORKER_COUNT>[ \t\r\n]+local_groups")
    message(FATAL_ERROR "configured ExSIA workers must retain individual profile intervals")
endif()
if(NOT header MATCHES "NativeCycleSample start_sample" OR
   NOT header MATCHES "NativeCycleSample end_sample" OR
   NOT header MATCHES "array<ggml::gemmini::cycle::NativeCycleSample, 5>")
    message(FATAL_ERROR "worker intervals and P0-P3 endpoints must retain native provenance")
endif()
if(NOT source MATCHES "checked_profile_interval[\n\r ]*\\([\n\r ]*const ProfileInterval &interval")
    message(FATAL_ERROR "individual ExSIA provenance validator is unavailable")
endif()

message(STATUS "ExSIA individual profile source contract passed")
