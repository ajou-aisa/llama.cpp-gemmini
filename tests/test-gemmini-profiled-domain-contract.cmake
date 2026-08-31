foreach(required IN ITEMS MATMUL_SOURCE MATMUL_HEADER MATMUL_TELEMETRY
                          GEMMINI_ENTRY_SOURCE GEMMINI_SOURCE_ROOT)
    if(NOT DEFINED ${required})
        message(FATAL_ERROR "${required} is required")
    endif()
endforeach()

file(READ "${MATMUL_SOURCE}" matmul)
file(READ "${MATMUL_HEADER}" matmul_header)
file(READ "${MATMUL_TELEMETRY}" transport)
file(READ "${GEMMINI_ENTRY_SOURCE}" gemmini_entry)
file(READ "${GEMMINI_SOURCE_ROOT}/quants/act/exsia/exsia.hpp" exsia_header)
file(READ "${GEMMINI_SOURCE_ROOT}/quants/act/exsia/exsia.cpp" exsia_source)
file(READ "${GEMMINI_SOURCE_ROOT}/ggml-gemmini-telemetry.cpp" telemetry_source)

string(FIND "${transport}"
    "std::string serialize_cycle_telemetry(const PipelineStripeTelemetry & record)"
    pipeline_serializer_begin)
if(pipeline_serializer_begin EQUAL -1)
    message(FATAL_ERROR "PIPELINE_STRIPE_SUMMARY serializer is unavailable")
endif()
string(SUBSTRING "${transport}" ${pipeline_serializer_begin} -1 pipeline_serializer_tail)
string(FIND "${pipeline_serializer_tail}" "\n#endif\n}" pipeline_serializer_end)
if(pipeline_serializer_end EQUAL -1)
    message(FATAL_ERROR "PIPELINE_STRIPE_SUMMARY serializer end is unavailable")
endif()
math(EXPR pipeline_serializer_length "${pipeline_serializer_end} + 8")
string(SUBSTRING "${pipeline_serializer_tail}" 0 ${pipeline_serializer_length} pipeline_serializer)

function(require_absent text token label)
    string(FIND "${text}" "${token}" position)
    if(NOT position EQUAL -1)
        message(FATAL_ERROR "${label}: unexpected ${token}")
    endif()
endfunction()

function(require_present text token label)
    string(FIND "${text}" "${token}" position)
    if(position EQUAL -1)
        message(FATAL_ERROR "${label}: missing ${token}")
    endif()
endfunction()

function(require_count text pattern expected label)
    string(REGEX MATCHALL "${pattern}" matches "${text}")
    list(LENGTH matches actual)
    if(NOT actual EQUAL expected)
        message(FATAL_ERROR "${label}: expected ${expected}, found ${actual}")
    endif()
endfunction()

# F2: ExSIA Local and stripe-total native endpoints are known cross-task. They
# may be checked where produced, but must not have a second unchecked consumer
# in StripeReadyEvent/capture/apply. Existing nanosecond transport is retained.
if(NOT DEFINED SEMANTIC_CASE OR SEMANTIC_CASE STREQUAL "F2")
    string(REGEX MATCH "struct StripeReadyEvent[^}]*\\};" ready_event
                 "${exsia_header}")
    if(ready_event STREQUAL "")
        message(FATAL_ERROR "F2: StripeReadyEvent is unavailable")
    endif()
    foreach(endpoint IN ITEMS local_start_cycle local_end_cycle
                              folding_start_cycle folding_end_cycle)
        require_absent("${ready_event}" "${endpoint}"
            "F2: cross-task native endpoint transport")
        require_absent("${transport}" "event.${endpoint}"
            "F2: unchecked cross-task collector arithmetic")
        require_absent("${exsia_source}" "event.${endpoint} ="
            "F2: duplicate raw native consumer")
    endforeach()
    foreach(legacy IN ITEMS local_start_ns local_end_ns folding_start_ns folding_end_ns)
        require_present("${ready_event}" "${legacy}" "F2: legacy ns transport")
        require_present("${transport}" "event.${legacy}" "F2: legacy ns collector")
    endforeach()
    foreach(structural_token IN ITEMS "exsia.local" "exsia.stripe_total"
                                      "checked_profile_interval(interval, !pipeline_cross_task)")
        require_present("${exsia_source}" "${structural_token}"
            "F2: producer-side structural unavailability")
    endforeach()
endif()

# F4: the LocalFoldingPipeline quantization envelope begins in a prepare task
# and ends in its dependent post task. It retains only the monotonic ns timeline;
# no scalar or native cycle endpoints may cross that structural task boundary.
if(NOT DEFINED SEMANTIC_CASE OR SEMANTIC_CASE STREQUAL "F4")
    string(FIND "${telemetry_source}"
        "std::string serialize_cycle_telemetry(const QuantizationStripeTelemetry & record)"
        quant_begin)
    string(FIND "${telemetry_source}"
        "std::string serialize_cycle_telemetry(const RmdTelemetryRecord & record)"
        quant_end)
    if(quant_begin EQUAL -1 OR quant_end EQUAL -1 OR quant_end LESS quant_begin)
        message(FATAL_ERROR "F4: quantization scalar serializer is unavailable")
    endif()
    math(EXPR quant_length "${quant_end} - ${quant_begin}")
    string(SUBSTRING "${telemetry_source}" ${quant_begin} ${quant_length} quant_serializer)
    foreach(required IN ITEMS
            "detail::null_field(out, \"start\")"
            "detail::null_field(out, \"end\")"
            "detail::null_field(out, \"delta\")"
            "detail::field(out, \"start_ns\", record.start_ns)"
            "detail::field(out, \"end_ns\", record.end_ns)"
            "detail::field(out, \"duration_ns\", record.end_ns - record.start_ns)"
            "detail::string_field(out, \"reason\", \"structurally_cross_task\")")
        require_present("${quant_serializer}" "${required}"
            "F4: structural cross-task quantization telemetry")
    endforeach()
    foreach(forbidden IN ITEMS
            "record.end - record.start"
            "NativeCycleSample"
            "scalar_provenance_unavailable")
        require_absent("${quant_serializer}" "${forbidden}"
            "F4: cross-task cycle arithmetic or provenance")
    endforeach()
    require_count("${exsia_source}"
        "slot\\.mark_quantization_started\\(0, aggregate_now_ns\\(\\)\\)" 3
        "F4: each pipeline prepare task retains only its ns boundary")
    require_count("${exsia_source}"
        "slot\\.mark_quantization_started\\(aggregate_now_tick\\(\\), aggregate_now_ns\\(\\)\\)" 1
        "F4: non-pipeline scalar-start behavior retains both domains")
    require_count("${exsia_source}"
        "aggregate_now_ns\\(\\), aggregate_now_tick\\(\\)" 1
        "F4: only the non-pipeline end boundary retains its scalar read")
    require_count("${exsia_source}"
        "slot\\.mark_folding_committed\\(aggregate_now_ns\\(\\)\\)" 1
        "F4: pipeline post task retains only its ns boundary")
endif()

# The canonical pipeline record is the origin/develop nanosecond schema. These
# names are forbidden there; standalone J-tile, Compose, Finalize, and
# capture_finish records are deliberately not matched by this list.
foreach(name IN ITEMS
        execution_route cpu_work_source cpu_work_unit accelerator
        npu_dispatch npu_wait rtl_cycles physical_provider_identity
        quantize_cpu_work_cycles quantize_cpu_work_reason
        dense_cpu_work_cycles dense_cpu_work_reason dense_cpu_work_coverage
        rmd_cpu_work_cycles rmd_cpu_work_reason rmd_cpu_work_coverage
        compose_cpu_work_cycles compose_cpu_work_reason
        finalize_cpu_work_cycles finalize_cpu_work_reason
        merge_cpu_work_cycles merge_cpu_work_reason merge_cpu_work_additive
        profiled_stripe_cpu_work_cycles profiled_stripe_cpu_work_reason
        cpu_work_status
        capture_finish_cycles capture_finish_valid capture_finish_route)
    require_absent("${pipeline_serializer}" "\"${name}\"" "canonical pipeline output contract")
endforeach()

# The aggregate model and its selectors are not an alternate internal API.
foreach(token IN ITEMS
        MatmulCpuWorkMetrics CpuWorkComponent CpuWorkCoverage CpuWorkAggregate
        aggregate_profiled_stripe_cpu_work finalize_profiled_cpu_work
        cpu_work_interval direct_cpu_work select_rmd_cpu_work
        aggregate_quantize_profile checked_sum)
    require_absent("${matmul}" "${token}" "matmul aggregate removal")
    require_absent("${matmul_header}" "${token}" "matmul model removal")
    require_absent("${transport}" "${token}" "pipeline aggregate removal")
    require_absent("${gemmini_entry}" "${token}" "entry aggregate removal")
endforeach()

# Named RMD/Quantize/PostFold totals are outside the approved detail boundary
# even when they are not serialized.
file(GLOB_RECURSE gemmini_sources
    "${GEMMINI_SOURCE_ROOT}/*.c" "${GEMMINI_SOURCE_ROOT}/*.cpp"
    "${GEMMINI_SOURCE_ROOT}/*.h" "${GEMMINI_SOURCE_ROOT}/*.hpp")
foreach(path IN LISTS gemmini_sources)
    file(READ "${path}" text)
    foreach(token IN ITEMS
            rmd_total_cycles quantize_total_cycles post_fold_cpu_work_cycles
            profiled_stripe_cpu_work)
        require_absent("${text}" "${token}" "forbidden aggregate in ${path}")
    endforeach()
endforeach()

if(EXISTS "${GEMMINI_SOURCE_ROOT}/ggml-gemmini-matmul-cpu-work.cpp" OR
   EXISTS "${GEMMINI_SOURCE_ROOT}/ggml-gemmini-matmul-cpu-work.hpp")
    message(FATAL_ERROR "canonical CPU-work framework files must be removed")
endif()

message(STATUS "approved detail and canonical schema contract passed")
