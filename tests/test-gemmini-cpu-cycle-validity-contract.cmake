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

require_cpu_count("gemmini_read_cycles\\(\\)" 162 "public scalar CPU endpoints")
require_cpu_count("static inline void ggml_log_cpu_cycle\\(" 1 "shared CPU log helper")
require_cpu_count("uint64_t start, uint64_t end\\)" 1 "scalar CPU logger endpoints")
require_cpu_count("gemmini_log_cycle_record_v2\\(&record\\)" 1 "public scalar CPU logger")

string(FIND "${cpu_source}"
    "static void ggml_compute_forward(struct ggml_compute_params" forward_start)
string(FIND "${cpu_source}" "switch (tensor->op)" switch_start)
if(forward_start EQUAL -1 OR switch_start EQUAL -1 OR switch_start LESS forward_start)
    message(FATAL_ERROR "ggml_compute_forward preamble is missing")
endif()
math(EXPR forward_length "${switch_start} - ${forward_start}")
string(SUBSTRING "${cpu_source}" ${forward_start} ${forward_length} forward_preamble)
string(REGEX MATCHALL "uint64_t start, end" scalar_declarations "${forward_preamble}")
list(LENGTH scalar_declarations scalar_declaration_count)
if(NOT scalar_declaration_count EQUAL 1)
    message(FATAL_ERROR
        "shared scalar endpoint declaration: expected 1, got ${scalar_declaration_count}")
endif()

string(SUBSTRING "${cpu_source}" ${forward_start} -1 cycle_region)
string(FIND "${cycle_region}"
    "#if defined(__linux__) && defined(__aarch64__)" endpoint_branch_pos)
if(NOT endpoint_branch_pos EQUAL -1)
    message(FATAL_ERROR "ggml-cpu retains a repeated Linux-AArch64 endpoint branch")
endif()

string(SUBSTRING "${cpu_source}" ${switch_start} -1 remaining)
foreach(label IN LISTS expected_labels)
    set(start_token "start = gemmini_read_cycles();")
    set(end_token "end = gemmini_read_cycles();")
    set(log_token "gemmini_log_cycle(layer, \"${label}\", start, end)")
    string(FIND "${remaining}" "${start_token}" start_pos)
    string(FIND "${remaining}" "${end_token}" end_pos)
    string(FIND "${remaining}" "${log_token}" log_pos)
    if(start_pos EQUAL -1 OR end_pos EQUAL -1 OR log_pos EQUAL -1 OR
       NOT start_pos LESS end_pos OR NOT end_pos LESS log_pos)
        message(FATAL_ERROR
            "${label}: expected scalar start -> operation -> scalar end -> log")
    endif()
    math(EXPR operation_start "${start_pos} + 31")
    math(EXPR operation_length "${end_pos} - ${operation_start}")
    string(SUBSTRING "${remaining}" ${operation_start} ${operation_length} operation_source)
    string(REGEX REPLACE "#[^\r\n]*" "" operation_source "${operation_source}")
    if(NOT operation_source MATCHES "[a-zA-Z_][a-zA-Z0-9_]* *\\(")
        message(FATAL_ERROR "${label}: operation body is missing between scalar endpoints")
    endif()
    string(LENGTH "${log_token}" log_length)
    math(EXPR next_pos "${log_pos} + ${log_length}")
    string(SUBSTRING "${remaining}" ${next_pos} -1 remaining)
endforeach()

set(forbidden_cpu_tokens
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

require_gemmini_count("ggml::gemmini::cycle::read\\(\\)" 15
    "Gemmini public scalar cycle endpoints")
require_gemmini_count("uint64_t start = 0" 1 "Gemmini shared scalar start endpoint")
require_gemmini_count("uint64_t end = 0" 1 "Gemmini shared scalar end endpoint")
require_gemmini_count(
    "rmd_telemetry_invocation_start = ggml::gemmini::cycle::read\\(\\)" 1
    "Gemmini invocation start boundary")
require_gemmini_count(
    "rmd_telemetry_invocation_end = ggml::gemmini::cycle::read\\(\\)" 1
    "Gemmini invocation end boundary")
require_gemmini_count(
    "quantize_start =[
 ]*ggml::gemmini::cycle::read\\(\\)" 1
    "Gemmini quantization start boundary")
require_gemmini_count(
    "quantize_end =[
 ]*ggml::gemmini::cycle::read\\(\\)" 1
    "Gemmini quantization end boundary")

set(expected_gemmini_labels
    gemmini.prepare_args
    gemmini.select_tile
    gemmini.quantize_activation
    gemmini.prepare_dense_i8_weight
    gemmini.convert_q4_0_to_q4_h1
    gemmini.convert_q8_0_to_q8_h1
    gemmini.prepare_args)
string(REGEX MATCHALL
    "ggml::gemmini::log::cycle\\([^;]*\"gemmini\\.[a-zA-Z0-9_]+\"[^;]*\\)"
    gemmini_records "${gemmini_source}")
list(LENGTH gemmini_records gemmini_record_count)
if(NOT gemmini_record_count EQUAL 7)
    message(FATAL_ERROR
        "Gemmini scalar record count: expected 7, got ${gemmini_record_count}")
endif()
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
    "rmd_telemetry_invocation_end >= rmd_telemetry_invocation_start")
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
    "cpu_work"
    "CpuWork")
foreach(token IN LISTS forbidden_gemmini_tokens)
    string(FIND "${gemmini_source}" "${token}" token_pos)
    if(NOT token_pos EQUAL -1)
        message(FATAL_ERROR
            "ggml-gemmini retains forbidden native/canonical cycle plumbing: ${token}")
    endif()
endforeach()

message(STATUS
    "scalar cycle contract passed: 81 CPU records, 15 Gemmini endpoints, zero forbidden consumers")
