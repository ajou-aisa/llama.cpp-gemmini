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

# The pipeline stripe-ready callback is a same-task handoff leaf. Its checked
# pair must remain adjacent to only the callback, carry real identity, and stay
# independent from callback acceptance. The non-pipeline callback is excluded.
require_count("${source}"
    "stripe_ready_handoff_start = ggml::gemmini::cycle::read_sample\\(\\)"
    1 "pipeline stripe-ready handoff performs one native start read")
require_count("${source}"
    "stripe_ready_handoff_end = ggml::gemmini::cycle::read_sample\\(\\)"
    1 "pipeline stripe-ready handoff performs one native end read")
require_count("${source}" "exsia\\.stripe_ready_handoff" 1
    "pipeline stripe-ready handoff has one detail label")
require_count("${source}"
    "GEMMINI_CYCLE_HAS_RUN_ID[ ]*[|][ ]*GEMMINI_CYCLE_HAS_STRIPE_ID[ ]*[|][\n\r ]*[ ]*GEMMINI_CYCLE_HAS_SLOT"
    1 "pipeline stripe-ready handoff carries run stripe and slot identity")
require_count("${source}"
    "gemmini_log_cycle_record_v2_checked_internal\\([\n\r ]*[ ]*&stripe_ready_handoff_record,[ ]*&stripe_ready_handoff_start_sample,[\n\r ]*[ ]*&stripe_ready_handoff_end_sample,[ ]*1\\)"
    1 "pipeline stripe-ready handoff is structurally same-owner checked")
string(FIND "${source}"
    "if (sink == nullptr || sink->on_ready == nullptr)\n                return true;"
    absent_callback_return)
if(absent_callback_return EQUAL -1)
    message(FATAL_ERROR "absent stripe-ready callback must return before the handoff pair")
endif()
require_count("${source}" "notify_stripe_ready\\(slot, run_id, true" 1
    "pipeline notify enables one handoff pair")
require_count("${source}" "notify_stripe_ready\\(slot, run_id, false" 1
    "non-pipeline notify disables the handoff pair")
string(FIND "${source}" "if (theta == std::numeric_limits<int16_t>::min())" theta_guard)
string(FIND "${source}" "stripe_ready_handoff_start = ggml::gemmini::cycle::read_sample()" handoff_start)
string(FIND "${source}" "const bool accepted = sink->on_ready(" handoff_callback)
string(FIND "${source}" "stripe_ready_handoff_end = ggml::gemmini::cycle::read_sample()" handoff_end)
string(FIND "${source}" "slot.release();" slot_release)
if(theta_guard EQUAL -1 OR handoff_start EQUAL -1 OR handoff_callback EQUAL -1 OR
   handoff_end EQUAL -1 OR slot_release EQUAL -1 OR
   NOT (theta_guard LESS handoff_start AND handoff_start LESS handoff_callback AND
   handoff_callback LESS handoff_end AND handoff_end LESS slot_release))
    message(FATAL_ERROR "stripe-ready native pair must follow validation and wrap only sink->on_ready")
endif()
math(EXPR pre_callback_length "${handoff_callback} - ${handoff_start}")
string(SUBSTRING "${source}" ${handoff_start} ${pre_callback_length} pre_callback_body)
if(pre_callback_body MATCHES "return")
    message(FATAL_ERROR "stripe-ready pair permits a no-invocation early return")
endif()
math(EXPR handoff_length "${handoff_end} - ${handoff_start}")
string(SUBSTRING "${source}" ${handoff_start} ${handoff_length} handoff_body)
foreach(forbidden IN ITEMS "slot.release" "stripe_total" "aggregate_now" "notify_stripe_ready")
    if(handoff_body MATCHES "${forbidden}")
        message(FATAL_ERROR "stripe-ready checked pair contains forbidden ${forbidden}")
    endif()
endforeach()
if(source MATCHES "exsia\\.slot_release" OR source MATCHES "stripe_ready_handoff_total")
    message(FATAL_ERROR "stripe-ready handoff must not add release or total records")
endif()
message(STATUS "ExSIA individual profile and stripe-ready handoff source contract passed")
