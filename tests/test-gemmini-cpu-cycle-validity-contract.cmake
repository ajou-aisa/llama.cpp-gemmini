if(NOT DEFINED CPU_SOURCE OR NOT DEFINED GEMMINI_SOURCE OR NOT DEFINED INTERNAL_HEADER OR
   NOT DEFINED CYCLE_HEADER OR NOT DEFINED CYCLE_CAPI OR NOT DEFINED LOG_CAPI OR
   NOT DEFINED SERIALIZATION_SOURCE OR NOT DEFINED CPU_CMAKE OR NOT DEFINED GEMMINI_CMAKE OR
   NOT DEFINED TELEMETRY_HEADER OR NOT DEFINED TELEMETRY_SOURCE OR NOT DEFINED MATMUL_SOURCE OR
   NOT DEFINED MATMUL_HEADER)
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

    find_program(UNIFDEF_EXECUTABLE unifdef)
    set(contract_failures)
    if(UNIFDEF_EXECUTABLE)
    function(check_architecture name expected_cpu_private expected_cpu_original
            expected_gemmini_private expected_gemmini_original expected_private_support)
        set(flags ${ARGN})
        execute_process(
            COMMAND "${UNIFDEF_EXECUTABLE}" -k ${flags} "${CPU_SOURCE}"
            RESULT_VARIABLE cpu_result OUTPUT_VARIABLE cpu_output ERROR_VARIABLE cpu_error)
        execute_process(
            COMMAND "${UNIFDEF_EXECUTABLE}" -k ${flags} "${GEMMINI_SOURCE}"
            RESULT_VARIABLE gemmini_result OUTPUT_VARIABLE gemmini_output ERROR_VARIABLE gemmini_error)
        if(NOT cpu_result MATCHES "^[01]$" OR NOT gemmini_result MATCHES "^[01]$")
            message(FATAL_ERROR "${name} preprocessing failed: ${cpu_error}${gemmini_error}")
        endif()
        string(REGEX MATCHALL "gemmini_read_native_cycle_sample_internal\\(\\)" cpu_private "${cpu_output}")
        string(REGEX MATCHALL "gemmini_read_cycles\\(\\)" cpu_original "${cpu_output}")
        string(REGEX MATCHALL "gemmini_read_native_cycle_sample_internal\\(\\)" gemmini_private "${gemmini_output}")
        string(REGEX MATCHALL "ggml::gemmini::cycle::read\\(\\)" gemmini_original "${gemmini_output}")
        list(LENGTH cpu_private cpu_private_count)
        list(LENGTH cpu_original cpu_original_count)
        list(LENGTH gemmini_private gemmini_private_count)
        list(LENGTH gemmini_original gemmini_original_count)
        set(actual "${cpu_private_count}/${cpu_original_count}/${gemmini_private_count}/${gemmini_original_count}")
        set(expected "${expected_cpu_private}/${expected_cpu_original}/${expected_gemmini_private}/${expected_gemmini_original}")
        set(support_output "${cpu_output}${gemmini_output}")
        foreach(support_source IN ITEMS
                "${CYCLE_HEADER}" "${CYCLE_CAPI}" "${LOG_CAPI}"
                "${SERIALIZATION_SOURCE}" "${MATMUL_SOURCE}" "${MATMUL_HEADER}")
            execute_process(
                COMMAND "${UNIFDEF_EXECUTABLE}" -k ${flags} "${support_source}"
                RESULT_VARIABLE support_result OUTPUT_VARIABLE one_support_output
                ERROR_VARIABLE support_error)
            if(NOT support_result MATCHES "^[01]$")
                message(FATAL_ERROR "${name} support preprocessing failed: ${support_error}")
            endif()
            string(APPEND support_output "${one_support_output}")
        endforeach()
        string(REGEX MATCHALL "cycle_reader_internal.h|gemmini_log_cycle_record_v2_checked_internal|gemmini_native_cycle_sample_internal|NativeCycleSample|serialize_checked_cycle_record|invocation_valid|invocation_reason" private_support "${support_output}")
        list(LENGTH private_support private_support_count)
        if(name STREQUAL "riscv" AND NOT support_output MATCHES "rdcycle")
            list(APPEND contract_failures "riscv: missing direct rdcycle operation")
        elseif(name STREQUAL "apple" AND NOT support_output MATCHES "mach_absolute_time")
            list(APPEND contract_failures "apple: missing direct mach_absolute_time operation")
        elseif(name STREQUAL "generic" AND NOT support_output MATCHES "steady_clock")
            list(APPEND contract_failures "generic: missing direct steady_clock operation")
        endif()
        if(NOT actual STREQUAL expected OR
           (expected_private_support AND private_support_count EQUAL 0) OR
           (NOT expected_private_support AND NOT private_support_count EQUAL 0))
            list(APPEND contract_failures
                "${name}: endpoints private/original CPU/Gemmini=${actual}, expected=${expected}, private-support-tokens=${private_support_count}")
            set(contract_failures "${contract_failures}" PARENT_SCOPE)
        endif()
    endfunction()

    check_architecture(jetson 162 0 15 0 TRUE
        -D__linux__ -D__aarch64__ -U__riscv -U__APPLE__)
    check_architecture(riscv 0 162 0 15 FALSE
        -U__linux__ -U__aarch64__ -D__riscv -U__APPLE__)
    check_architecture(apple 0 162 0 15 FALSE
        -U__linux__ -D__aarch64__ -U__riscv -D__APPLE__)
    check_architecture(generic 0 162 0 15 FALSE
        -U__linux__ -U__aarch64__ -U__riscv -U__APPLE__)
    check_architecture(linux_only 0 162 0 15 FALSE
        -D__linux__ -U__aarch64__ -U__riscv -U__APPLE__)
    check_architecture(aarch64_only 0 162 0 15 FALSE
        -U__linux__ -D__aarch64__ -U__riscv -U__APPLE__)
    execute_process(
        COMMAND "${UNIFDEF_EXECUTABLE}" -k -DCYCLE_LOG=0
            -D__linux__ -D__aarch64__ -U__riscv -U__APPLE__ "${CPU_SOURCE}"
        RESULT_VARIABLE disabled_result OUTPUT_VARIABLE disabled_cpu)
    if(NOT disabled_result MATCHES "^[01]$" OR disabled_cpu MATCHES "cycle_reader_internal.h")
        list(APPEND contract_failures "disabled Jetson CPU logging retains a private include dependency")
    endif()
    else()
        require_count("${cpu_source}" "gemmini_read_native_cycle_sample_internal\\(\\)" 162 "raw private CPU endpoints")
        require_count("${cpu_source}" "gemmini_read_cycles\\(\\)" 162 "raw original CPU endpoints")
        require_count("${gemmini_source}" "gemmini_read_native_cycle_sample_internal\\(\\)" 15 "raw private Gemmini endpoints")
        require_count("${gemmini_source}" "ggml::gemmini::cycle::read\\(\\)" 15 "raw original Gemmini endpoints")
        string(REPLACE ";" "<semi>" cpu_guard_source "${cpu_source}")
        string(REPLACE ";" "<semi>" gemmini_guard_source "${gemmini_source}")
        require_count("${cpu_guard_source}" "#if defined\\(__linux__\\) && defined\\(__aarch64__\\)[ \t\r\n]+start = gemmini_read_native_cycle_sample_internal\\(\\)<semi>[ \t\r\n]+#else[ \t\r\n]+start = gemmini_read_cycles\\(\\)<semi>" 81 "guarded CPU start pairs")
        require_count("${cpu_guard_source}" "#if defined\\(__linux__\\) && defined\\(__aarch64__\\)[ \t\r\n]+end = gemmini_read_native_cycle_sample_internal\\(\\)<semi>[ \t\r\n]+#else[ \t\r\n]+end = gemmini_read_cycles\\(\\)<semi>" 81 "guarded CPU end pairs")
        require_count("${gemmini_guard_source}" "#if defined\\(__linux__\\) && defined\\(__aarch64__\\)[^#]*gemmini_read_native_cycle_sample_internal\\(\\)<semi>[^#]*#else[^#]*ggml::gemmini::cycle::read\\(\\)" 15 "guarded Gemmini endpoint pairs")
        message(STATUS "unifdef unavailable; exact guarded-pair source contract used")
    endif()
    file(READ "${CPU_CMAKE}" cpu_cmake)
    file(READ "${GEMMINI_CMAKE}" gemmini_cmake)
    foreach(cmake_source IN ITEMS cpu_cmake gemmini_cmake)
        set(cmake_text "${${cmake_source}}")
        string(FIND "${cmake_text}" "CMAKE_SYSTEM_NAME STREQUAL \"Linux\"" system_guard)
        string(FIND "${cmake_text}" "CMAKE_SYSTEM_PROCESSOR MATCHES \"^(aarch64|arm64)$\"" processor_guard)
        if(system_guard EQUAL -1 OR processor_guard EQUAL -1)
            list(APPEND contract_failures "${cmake_source}: private include is not architecture-isolated")
        endif()
    endforeach()
    if(contract_failures)
        list(JOIN contract_failures "\n" failure_report)
        message(FATAL_ERROR "platform source-isolation contract failed:\n${failure_report}")
    endif()
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
