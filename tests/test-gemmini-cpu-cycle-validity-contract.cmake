if(NOT DEFINED CPU_SOURCE OR NOT DEFINED CPU_CMAKE OR NOT DEFINED GEMMINI_SOURCE)
    message(FATAL_ERROR "CPU_SOURCE, CPU_CMAKE, and GEMMINI_SOURCE are required")
endif()

file(READ "${CPU_SOURCE}" cpu_source)
file(READ "${CPU_CMAKE}" cpu_cmake)
file(READ "${GEMMINI_SOURCE}" gemmini_source)

function(require_cpu_count regex expected description)
    string(REGEX MATCHALL "${regex}" matches "${cpu_source}")
    list(LENGTH matches actual)
    if(NOT actual EQUAL expected)
        message(FATAL_ERROR "${description}: expected ${expected}, got ${actual}")
    endif()
endfunction()

function(require_gemmini_count regex expected description)
    string(REGEX MATCHALL "${regex}" matches "${gemmini_source}")
    list(LENGTH matches actual)
    if(NOT actual EQUAL expected)
        message(FATAL_ERROR "${description}: expected ${expected}, got ${actual}")
    endif()
endfunction()

set(expected_labels
    cpu.dup cpu.add cpu.add1 cpu.acc cpu.sub cpu.mul cpu.div cpu.sqr cpu.sqrt
    cpu.log cpu.sin cpu.cos cpu.sum cpu.sum_rows cpu.mean cpu.argmax cpu.count_equal
    cpu.repeat cpu.repeat_back cpu.concat cpu.silu_back cpu.norm cpu.rms_norm
    cpu.rms_norm_back cpu.group_norm cpu.l2_norm cpu.mul_mat cpu.mul_mat_id
    cpu.out_prod cpu.scale cpu.set cpu.cpy cpu.cont cpu.reshape cpu.view cpu.permute
    cpu.transpose cpu.get_rows cpu.get_rows_back cpu.diag cpu.diag_mask_inf
    cpu.diag_mask_zero cpu.softmax cpu.softmax_back cpu.rope cpu.rope_back cpu.clamp
    cpu.conv_transpose_1d cpu.im2col cpu.im2col_back cpu.conv_2d_dw
    cpu.conv_transpose_2d cpu.pool_1d cpu.pool_2d cpu.pool_2d_back cpu.upscale
    cpu.pad cpu.pad_reflect_1d cpu.arange cpu.timestep_embedding cpu.argsort
    cpu.leaky_relu cpu.flash_attn_ext cpu.flash_attn_back cpu.ssm_conv cpu.ssm_scan
    cpu.win_part cpu.win_unpart cpu.unary cpu.get_rel_pos cpu.add_rel_pos
    cpu.rwkv_wkv6 cpu.gated_linear_attn cpu.rwkv_wkv7 cpu.map_custom1
    cpu.map_custom2 cpu.map_custom3 cpu.custom cpu.cross_entropy_loss
    cpu.cross_entropy_loss_back cpu.opt_step_adamw)

string(REGEX MATCHALL
    "gemmini_log_cycle\\(layer, \"cpu\\.[a-zA-Z0-9_]+\", start, end\\)"
    records "${cpu_source}")
list(LENGTH records record_count)
if(NOT record_count EQUAL 81)
    message(FATAL_ERROR "generic CPU record count: expected 81, got ${record_count}")
endif()
set(actual_labels)
foreach(record IN LISTS records)
    string(REGEX REPLACE ".*\"(cpu\\.[a-zA-Z0-9_]+)\".*" "\\1" label "${record}")
    list(APPEND actual_labels "${label}")
endforeach()
if(NOT actual_labels STREQUAL expected_labels)
    message(FATAL_ERROR "generic CPU labels or ordering changed")
endif()

require_cpu_count("static inline void ggml_log_cpu_cycle\\(" 1 "shared CPU log helper")
require_cpu_count("const gemmini_cpu_sample \\*start, const gemmini_cpu_sample \\*end" 1
    "CPU logger preserves native and host endpoints")
require_cpu_count("gemmini_cpu_timing_record\\(&record, start, end\\)" 1
    "shared checked CPU logger")
foreach(token IN ITEMS ".run_id = params->run_id" ".node_id = params->node_id"
                       ".worker_id = (uint64_t) params->ith"
                       "gemmini_cpu_timing_add(&state->cpu_totals, &cpu_start, &cpu_end)"
                       "gemmini_cpu_timing_record(&worker_record, &cpu_start, &cpu_end)"
                       "gemmini_cpu_timing_merge(&cpu_totals, &threadpool->workers[ith].cpu_totals)")
    string(FIND "${cpu_source}" "${token}" cpu_token)
    if(cpu_token EQUAL -1)
        message(FATAL_ERROR "CPU worker identity/resource endpoint missing ${token}")
    endif()
endforeach()

string(FIND "${cpu_source}"
    "static void ggml_compute_forward(struct ggml_compute_params" forward_start)
string(FIND "${cpu_source}" "switch (tensor->op)" switch_start)
if(forward_start EQUAL -1 OR switch_start EQUAL -1 OR switch_start LESS forward_start)
    message(FATAL_ERROR "ggml_compute_forward preamble is missing")
endif()
math(EXPR forward_length "${switch_start} - ${forward_start}")
string(SUBSTRING "${cpu_source}" ${forward_start} ${forward_length} forward_preamble)
string(REGEX MATCHALL "gemmini_cpu_sample start, end" scalar_declarations "${forward_preamble}")
list(LENGTH scalar_declarations scalar_declaration_count)
if(NOT scalar_declaration_count EQUAL 1)
    message(FATAL_ERROR
        "shared native endpoint declaration: expected 1, got ${scalar_declaration_count}")
endif()

string(SUBSTRING "${cpu_source}" ${forward_start} -1 cycle_region)
string(FIND "${cycle_region}"
    "#if defined(__linux__) && defined(__aarch64__)" endpoint_branch_pos)
if(NOT endpoint_branch_pos EQUAL -1)
    message(FATAL_ERROR "ggml-cpu retains a repeated Linux-AArch64 endpoint branch")
endif()

string(SUBSTRING "${cpu_source}" ${switch_start} -1 remaining)
foreach(label IN LISTS expected_labels)
    set(start_token "start = gemmini_cpu_timing_read();")
    set(end_token "end = gemmini_cpu_timing_read();")
    set(log_token "gemmini_log_cycle(layer, \"${label}\", start, end)")
    string(FIND "${remaining}" "${start_token}" start_pos)
    string(FIND "${remaining}" "${end_token}" end_pos)
    string(FIND "${remaining}" "${log_token}" log_pos)
    if(start_pos EQUAL -1 OR end_pos EQUAL -1 OR log_pos EQUAL -1 OR
       NOT start_pos LESS end_pos OR NOT end_pos LESS log_pos)
        message(FATAL_ERROR
            "${label}: expected native start, operation, native end, log")
    endif()
    string(LENGTH "${start_token}" start_length)
    math(EXPR operation_start "${start_pos} + ${start_length}")
    math(EXPR operation_length "${end_pos} - ${operation_start}")
    string(SUBSTRING "${remaining}" ${operation_start} ${operation_length} operation_source)
    string(REGEX REPLACE "#[^\r\n]*" "" operation_source "${operation_source}")
    if(NOT operation_source MATCHES "[a-zA-Z_][a-zA-Z0-9_]* *\\(")
        message(FATAL_ERROR "${label}: operation body is missing between native endpoints")
    endif()
    string(LENGTH "${log_token}" log_length)
    math(EXPR next_pos "${log_pos} + ${log_length}")
    string(SUBSTRING "${remaining}" ${next_pos} -1 remaining)
endforeach()

set(forbidden_cpu_tokens
    "gemmini_read_cycles()"
    "gemmini_log_cycle_record_v2(&record)"
    "if (params->run_id)"
    "if (params->ith)"
    "gemmini_read_native_cycle_sample_internal"
    "gemmini_native_cycle_sample_internal"
    "gemmini_log_cycle_record_v2_checked_internal"
    "cycle_reader_internal.h"
    "NativeCycleSample"
    "read_sample()")
foreach(token IN LISTS forbidden_cpu_tokens)
    string(FIND "${cpu_source}" "${token}" token_pos)
    if(NOT token_pos EQUAL -1)
        message(FATAL_ERROR "ggml-cpu retains forbidden private/sample plumbing: ${token}")
    endif()
endforeach()

foreach(token IN ITEMS "ggml-gemmini-utils/src" "cycle_reader_internal.h")
    string(FIND "${cpu_cmake}" "${token}" token_pos)
    if(NOT token_pos EQUAL -1)
        message(FATAL_ERROR "ggml-cpu retains forbidden private include path: ${token}")
    endif()
endforeach()

require_gemmini_count("ggml::gemmini::cycle::read\\(\\)" 0
    "Gemmini outer scopes have no unowned scalar cycle endpoints")
require_gemmini_count("gemmini_cpu_sample start\\{\\}, end\\{\\}" 1
    "Gemmini shared native/host endpoint storage")
require_gemmini_count(
    "rmd_telemetry_invocation_start = ggml::gemmini::read_matmul_cpu_sample\\(\\)" 1
    "Gemmini invocation start boundary")
require_gemmini_count(
    "rmd_telemetry_invocation_end = ggml::gemmini::read_matmul_cpu_sample\\(\\)" 1
    "Gemmini invocation end boundary")
require_gemmini_count(
    "quantize_start = gemmini_cpu_timing_read\\(\\)" 1
    "Gemmini quantization start boundary")
require_gemmini_count(
    "quantize_end = gemmini_cpu_timing_read\\(\\)" 1
    "Gemmini quantization end boundary")

set(expected_gemmini_labels
    gemmini.prepare_args
    gemmini.select_tile
    gemmini.activation_buffer_preparation
    gemmini.quantize_activation
    gemmini.prepare_dense_i8_weight
    gemmini.convert_q4_0_to_q4_h1
    gemmini.convert_q8_0_to_q8_h1
    gemmini.prepare_weight
    gemmini.output_preparation)
string(REGEX MATCHALL
    "log_outer_cpu_interval\\(args,[^;]*\"gemmini\\.[a-zA-Z0-9_]+\"[^;]*\\)"
    gemmini_records "${gemmini_source}")
list(LENGTH gemmini_records gemmini_record_count)
if(NOT gemmini_record_count EQUAL 9)
    message(FATAL_ERROR
        "Gemmini native interval count: expected 9, got ${gemmini_record_count}")
endif()

foreach(token IN ITEMS "overlaps_rtl=true" "excluded_from_cycle_sink=true")
    string(FIND "${gemmini_source}" "${token}" overlap_token)
    if(NOT overlap_token EQUAL -1)
        message(FATAL_ERROR "Gemmini must preserve timestamps without asserting overlap: ${token}")
    endif()
endforeach()
foreach(token IN ITEMS "gemmini_cpu_timing_add(&totals, &start, &end)"
                       "cycle::serialize_cpu_native(start, end)"
                       "cycle::serialize_host_timing(start.ns, end.ns, start.tid, end.tid)"
                       "matmul_invocation_id" "MATMUL_CONFIGURATION"
                       "quantize_start, quantize_end, false, result")
    string(FIND "${gemmini_source}" "${token}" required_token)
    if(required_token EQUAL -1)
        message(FATAL_ERROR "Gemmini native context/endpoint contract missing ${token}")
    endif()
endforeach()
set(actual_gemmini_labels)
foreach(record IN LISTS gemmini_records)
    string(REGEX REPLACE
        ".*\"(gemmini\\.[a-zA-Z0-9_]+)\".*" "\\1" label "${record}")
    list(APPEND actual_gemmini_labels "${label}")
endforeach()
if(NOT actual_gemmini_labels STREQUAL expected_gemmini_labels)
    message(FATAL_ERROR "Gemmini scalar labels, ordering, or duplicate pairs changed")
endif()

set(required_gemmini_operations
    "args.transpose_B = (TRANSPOSE_B != 0)"
    "ggml::gemmini::gemmini_set_tile_ws(&args)"
    "ggml::gemmini::quants::quantize_activation(src1, args)"
    "ggml::gemmini::prepare_q4_0_rows_for_q4_h1("
    "ggml::gemmini::prepare_q8_0_rows_for_q8_h1("
    "pipeline_stripe_telemetry(layer, profile)"
    "ggml::gemmini::evaluate_matmul_cpu_interval("
    "rmd_telemetry_invocation_start, rmd_telemetry_invocation_end)")
foreach(operation IN LISTS required_gemmini_operations)
    string(FIND "${gemmini_source}" "${operation}" operation_pos)
    if(operation_pos EQUAL -1)
        message(FATAL_ERROR
            "Gemmini scalar boundary source operation changed or disappeared: ${operation}")
    endif()
endforeach()

set(forbidden_gemmini_tokens
    "cycle_reader_internal.h"
    "read_sample()"
    "gemmini_read_native_cycle_sample_internal"
    "gemmini_native_cycle_sample_internal"
    "gemmini_log_cycle_record_v2_checked_internal"
    "log_native_cycle_interval"
    "#if defined(__linux__) && defined(__aarch64__)"
    "PipelineExecutionRoute"
    "telemetry_execution_route"
    "invocation_valid"
    "invocation_reason"
    "CpuWork")
foreach(token IN LISTS forbidden_gemmini_tokens)
    string(FIND "${gemmini_source}" "${token}" token_pos)
    if(NOT token_pos EQUAL -1)
        message(FATAL_ERROR
            "ggml-gemmini retains forbidden native/canonical cycle plumbing: ${token}")
    endif()
endforeach()

# Adapter boundaries must exist at the real callsites, not only in a name table.
get_filename_component(gemmini_source_dir "${GEMMINI_SOURCE}" DIRECTORY)
file(READ "${gemmini_source_dir}/ggml-gemmini-im2p.cpp" im2p_source)
foreach(operation IN ITEMS
        im2p.host_input_preparation
        im2p.stripe_input_capture
        im2p.stripe_submit_host_call
        im2p.frontend_start_host_call
        im2p.fence_host_call
        im2p.residual_metadata_preparation
        im2p.residual_backend_host_call
        im2p.residual_simulator_host_call
        im2p.output_correction_apply
        im2p.post_fence_validation
        im2p.output_authorize_host_call
        im2p.output_buffer_copy)
    string(FIND "${im2p_source}" "\"${operation}\"" operation_pos)
    if(operation_pos EQUAL -1)
        message(FATAL_ERROR "Missing real IM2P host CPU boundary: ${operation}")
    endif()
endforeach()
# The captured FULL and PIPELINE callbacks bind the actual event, while the
# upstream baseline FULL path obtains the optional run from its own metadata.
string(REGEX MATCHALL "direct_metrics[.]run_id[ ]*=[^;]*" direct_run_bindings "${im2p_source}")
list(LENGTH direct_run_bindings direct_run_count)
if(NOT direct_run_count EQUAL 3)
    message(FATAL_ERROR "All three direct worker callsites must preserve run identity, including zero")
endif()
set(event_run_count 0)
set(metadata_run_count 0)
foreach(binding IN LISTS direct_run_bindings)
    if(binding STREQUAL "direct_metrics.run_id = event.run_id")
        math(EXPR event_run_count "${event_run_count} + 1")
    elseif(binding STREQUAL "direct_metrics.run_id = metadata->run_id")
        math(EXPR metadata_run_count "${metadata_run_count} + 1")
    else()
        message(FATAL_ERROR "Direct worker identity must come from its actual event or metadata: ${binding}")
    endif()
endforeach()
if(NOT event_run_count EQUAL 2 OR NOT metadata_run_count EQUAL 1)
    message(FATAL_ERROR "Expected two event bindings and one baseline metadata binding")
endif()

message(STATUS
    "CPU operation coverage and IM2P host boundary contract passed")
