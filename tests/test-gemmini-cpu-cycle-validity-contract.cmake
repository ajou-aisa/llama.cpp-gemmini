if(NOT DEFINED CPU_SOURCE OR NOT DEFINED GEMMINI_SOURCE OR NOT DEFINED INTERNAL_HEADER OR
   NOT DEFINED TELEMETRY_HEADER OR NOT DEFINED TELEMETRY_SOURCE OR NOT DEFINED MATMUL_SOURCE)
    message(FATAL_ERROR "all Task-3 source paths are required")
endif()
file(READ "${CPU_SOURCE}" cpu_source)
file(READ "${GEMMINI_SOURCE}" gemmini_source)

function(require_count source regex expected description)
    string(REGEX MATCHALL "${regex}" matches "${source}")
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
string(REGEX MATCHALL "gemmini_log_cycle\\(layer, \"cpu\\.[a-zA-Z0-9_]+\", start, end\\)" records "${cpu_source}")
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

require_count("${cpu_source}" "static inline void ggml_log_cpu_cycle\\(" 1 "shared CPU log helper")

if(EXISTS "${INTERNAL_HEADER}")
    file(READ "${TELEMETRY_HEADER}" telemetry_header)
    file(READ "${TELEMETRY_SOURCE}" telemetry_source)
    file(READ "${MATMUL_SOURCE}" matmul_source)
    foreach(label IN ITEMS linux_perf_cpu_cycles host_tick riscv_cycle)
        if(NOT telemetry_header MATCHES "${label}")
            message(FATAL_ERROR "missing native telemetry label ${label}")
        endif()
    endforeach()
    string(FIND "${telemetry_source}" "scalar_provenance_unavailable" stripe_reason)
    string(FIND "${matmul_source}" "queue_reason" queue_reason)
    if(stripe_reason EQUAL -1 OR queue_reason EQUAL -1)
        message(FATAL_ERROR "Jetson stripe or queue fail-closed serialization is missing")
    endif()
    require_count("${gemmini_source}" "ggml::gemmini::cycle::read\\(\\)" 0 "post-edit top-level legacy cycle reads")
    require_count("${cpu_source}" "gemmini_read_cycles\\(\\)" 0 "post-edit public scalar CPU reads")
    require_count("${cpu_source}" "gemmini_read_native_cycle_sample_internal\\(\\)" 162 "post-edit private CPU sample reads")
    require_count("${cpu_source}" "gemmini_native_cycle_sample_internal start, end" 1 "shared private endpoint declaration")
    require_count("${gemmini_source}" "gemmini_read_native_cycle_sample_internal\\(\\)" 15 "top-level private sample reads")
    require_count("${gemmini_source}" "log_native_cycle_interval\\(" 8 "top-level checked interval helper and handoffs")
    require_count("${gemmini_source}" "rmd_telemetry_invocation_start_sample" 3 "invocation start sample propagation")
    require_count("${gemmini_source}" "rmd_telemetry_invocation_end_sample" 3 "invocation end sample propagation")
else()
    require_count("${gemmini_source}" "ggml::gemmini::cycle::read\\(\\)" 15 "baseline top-level legacy cycle reads")
    string(FIND "${cpu_source}" "static void ggml_compute_forward(struct ggml_compute_params" forward_start)
    string(FIND "${cpu_source}" "switch (tensor->op)" switch_start)
    math(EXPR forward_length "${switch_start} - ${forward_start}")
    string(SUBSTRING "${cpu_source}" ${forward_start} ${forward_length} forward_preamble)
    require_count("${forward_preamble}" "uint64_t start, end" 1 "baseline shared scalar endpoint declaration")
    require_count("${cpu_source}" "gemmini_read_cycles\\(\\)" 162 "baseline public scalar CPU reads")
endif()
message(STATUS "cycle validity source contract passed: 15 top-level boundaries, 81 records, 162 CPU endpoints")
