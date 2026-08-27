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

require_count("${source}" "interval\\.start_sample = ggml::gemmini::cycle::read_sample\\(\\)" 1
    "profile start performs one native sample read")
require_count("${source}" "interval\\.end_sample = ggml::gemmini::cycle::read_sample\\(\\)" 1
    "profile end performs one native sample read")
require_count("${source}" "interval\\.start = profile_now\\(\\)" 1
    "non-Jetson profile start retains baseline scalar reader path")
require_count("${source}" "interval\\.end = profile_now\\(\\)" 1
    "non-Jetson profile end retains baseline scalar reader path")
require_count("${source}" "return interval\\.end >= interval\\.start" 1
    "non-Jetson profile validity retains baseline scalar ordering")
require_count("${header}" "sum \\+= value;[\n\r ]*max = std::max\\(max, value\\);[\n\r ]*\\+\\+count" 3
    "non-Jetson stage accumulation retains baseline scalar operations")
require_count("${source}" "const auto t[0-4] = EXSIA_STAGE_CYCLE_READ\\(\\)" 15
    "Linux-AArch64 P0-P3 paths share exactly five native boundaries")
require_count("${source}" "const uint64_t t[0-4] = EXSIA_STAGE_CYCLE_READ\\(\\)" 15
    "non-Jetson P0-P3 declarations retain the baseline scalar source path")
require_count("${source}" "record_stage_cycles\\(cycle_sample, \\{t0, t1, t2, t3, t4\\}\\)" 3
    "each Linux-AArch64 P0-P3 path consumes the shared boundaries")
foreach(stage RANGE 0 3)
    math(EXPR next "${stage} + 1")
    require_count("${source}"
        "cycle_sample\\.p${stage} = t${next} >= t${stage} \\? t${next} - t${stage} : 0"
        3 "non-Jetson P${stage} retains the baseline inline subtraction")
endforeach()
if(source MATCHES "array<uint64_t, 5> &endpoints")
    message(FATAL_ERROR "non-Jetson P0-P3 must not route through a scalar helper")
endif()
foreach(stage RANGE 0 3)
    require_count("${source}"
        "slot\\.cycle_stats\\.p${stage}\\.sum \\+= task_stats\\.p${stage}\\.sum"
        2 "non-Jetson P${stage} worker sum retains baseline inline operation")
    require_count("${source}"
        "slot\\.cycle_stats\\.p${stage}\\.count \\+= task_stats\\.p${stage}\\.count"
        2 "non-Jetson P${stage} worker count retains baseline inline operation")
endforeach()
require_count("${source}" "slot\\.cycle_stats\\.p[0-3]\\.merge\\(task_stats\\.p[0-3]\\)" 8
    "Linux-AArch64 stage aggregation keeps checked merge")

string(FIND "${source}" "QuantizeProfileCycles aggregate_quantize_profile" reducer_start)
if(reducer_start EQUAL -1)
    message(FATAL_ERROR "canonical Quantize reducer not found")
endif()
string(SUBSTRING "${source}" ${reducer_start} -1 reducer_tail)
string(FIND "${reducer_tail}" "\n#endif" reducer_length)
if(reducer_length EQUAL -1)
    message(FATAL_ERROR "canonical Quantize reducer boundary not found")
endif()
string(SUBSTRING "${reducer_tail}" 0 ${reducer_length} reducer)
foreach(forbidden IN ITEMS "profile.stats" "capture_finish" "rmd_pack" "stripe_total")
    if(reducer MATCHES "${forbidden}")
        message(FATAL_ERROR "canonical Quantize reducer includes forbidden ${forbidden}")
    endif()
endforeach()
if(NOT reducer MATCHES "worker < workers.size\\(\\)" OR
   reducer MATCHES "workers.size\\(\\)[ ]*-[ ]*1")
    message(FATAL_ERROR "canonical Quantize reducer must consume every configured worker")
endif()
if(header MATCHES "local_worker_start_ns|local_worker_end_ns|local_group3_start_cycle|local_group3_end_cycle")
    message(FATAL_ERROR "StripeReadyEvent must not carry a fixed worker-detail array")
endif()
if(NOT header MATCHES "NativeCycleSample start_sample" OR
   NOT header MATCHES "NativeCycleSample end_sample" OR
   NOT header MATCHES "array<ggml::gemmini::cycle::NativeCycleSample, 5>")
    message(FATAL_ERROR "profile and P0-P3 endpoints must retain native provenance")
endif()
message(STATUS "ExSIA profile source contract passed")
