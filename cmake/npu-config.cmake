# Common NPU hardware-parameter generator (Gemmini and PoTAL).
#
# The NPU geometry is a build input: GGML_NPU, NPU_PRECISION, NPU_D,
# NPU_SPAD_BANKS, NPU_SPAD_BANK_ROWS, NPU_ACC_BANKS, NPU_ACC_ROWS (total rows),
# NPU_DMA_MAX_BYTES, NPU_INDEX_ENTRIES / NPU_SCALE_ENTRIES (PoTAL) and
# NPU_TRANSPORT. Nothing is inferred: an enabled NPU needs every parameter,
# either on the command line or from a CMakePresets.json preset. The only
# other source is an explicit legacy PoTAL header (-DPOTAL_HW_PARAMS), whose
# values are imported and must agree with any NPU_* value given alongside.
#
# Outputs in ${CMAKE_BINARY_DIR}/generated:
#   npu_config.h, npu_config.json, npu_build_manifest.json
#   gemmini_params.h           (GEMMINI: the backend's parameter header)
#   npu/gemmini_potal*_params.h (POTAL, RTL file name: fed to cmake/potal-hw-params.cmake)
# Legacy options (GGML_GEMMINI, GGML_POTAL, GGML_POTAL_W4A4, GGML_GEMMINI_DIM,
# POTAL_EXPECTED_DIM, GGML_GEMMINI_OPTION, POTAL_API_ONLY, ...) are derived
# from NPU_*; a conflicting explicit legacy value is a configure error.

include(${CMAKE_CURRENT_LIST_DIR}/npu-validation.cmake)

# CMake owns option precedence: explicit cache/CLI, environment, then defaults.
execute_process(COMMAND "${CMAKE_COMMAND}" -E environment OUTPUT_VARIABLE _npu_environment)
string(REPLACE "\n" ";" _npu_environment "${_npu_environment}")
set(_npu_environment_names)
foreach(_npu_entry IN LISTS _npu_environment)
    if(_npu_entry MATCHES "^((GGML_|NPU_|POTAL_|LLAMA_|CMAKE_|GEMMINI_)[A-Za-z0-9_]+|LOG_DEBUG|LOG_CYCLE|LOG_DUMP|LOG_DUMP_SCALE|CYCLE_DETAIL|TRANSPOSE_[AB]|FULL_C|LOW_D|EXSIA_VALIDATION)(_DEFAULT)?=(.*)$")
        set(_npu_name "${CMAKE_MATCH_1}")
        set(_npu_value "${CMAKE_MATCH_4}")
        if(_npu_entry MATCHES "^([^=]+)_DEFAULT=")
            set(_npu_name "${CMAKE_MATCH_1}")
            set(_npu_environment_default_${_npu_name} "${_npu_value}")
        else()
            set(_npu_environment_value_${_npu_name} "${_npu_value}")
        endif()
        list(APPEND _npu_environment_names "${_npu_name}")
    endif()
endforeach()
foreach(_npu_name IN LISTS _npu_environment_names)
    if(NOT DEFINED CACHE{${_npu_name}})
        if(DEFINED _npu_environment_value_${_npu_name})
            set(${_npu_name} "${_npu_environment_value_${_npu_name}}" CACHE STRING "Environment build option")
        elseif(DEFINED _npu_environment_default_${_npu_name})
            set(${_npu_name} "${_npu_environment_default_${_npu_name}}" CACHE STRING "Environment build default")
        endif()
    endif()
endforeach()

if(NOT DEFINED GGML_CPU_CYCLE_LOG)
    if(LOG_CYCLE)
        set(GGML_CPU_CYCLE_LOG ON CACHE BOOL "Enable CPU cycle logging")
    else()
        set(GGML_CPU_CYCLE_LOG OFF CACHE BOOL "Enable CPU cycle logging")
    endif()
endif()

set(GGML_NPU "" CACHE STRING "NPU architecture: NONE, GEMMINI or POTAL (empty: derived from GGML_GEMMINI/GGML_POTAL)")
set_property(CACHE GGML_NPU PROPERTY STRINGS "" NONE GEMMINI POTAL)
set(NPU_PRECISION "" CACHE STRING "Single NPU precision: INT8 or INT4 (POTAL); INT8 or INT16 (GEMMINI, INT16 REFERENCE only)")
set(NPU_TRANSPORT "" CACHE STRING "Command transport: ROCC or REFERENCE")
set(NPU_MATMUL_ALGORITHM "" CACHE STRING "Gemmini matmul algorithm: BASELINE or PARTIAL_SUM (empty: derive from GGML_GEMMINI_PB)")
set_property(CACHE NPU_MATMUL_ALGORITHM PROPERTY STRINGS "" BASELINE PARTIAL_SUM)
foreach(_npu_param IN ITEMS NPU_D NPU_SPAD_BANKS NPU_SPAD_BANK_ROWS NPU_ACC_BANKS NPU_ACC_ROWS
        NPU_DMA_MAX_BYTES NPU_INDEX_ENTRIES NPU_SCALE_ENTRIES)
    set(${_npu_param} "" CACHE STRING "NPU hardware parameter (see cmake/npu-config.cmake)")
endforeach()
option(NPU_API_ONLY "Configure only the standalone NPU API and its tests" OFF)
option(NPU_PROFILING "Record NPU profiling configuration in the build manifest" OFF)

set(NPU_GENERATED_DIR "${CMAKE_BINARY_DIR}/generated")
set(_NPU_TEMPLATE_DIR "${CMAKE_CURRENT_LIST_DIR}/../ggml/src/ggml-npu-common/config")

# A legacy cache variable derived from NPU_*: an explicit, different user
# value is a conflict; a value this module wrote before is updated.
function(npu_derive_legacy name value type)
    if(DEFINED CACHE{${name}} OR DEFINED ${name})
        set(current "${${name}}")
        if(type STREQUAL "BOOL")
            if(current)
                set(current ON)
            else()
                set(current OFF)
            endif()
        endif()
        if(NOT current STREQUAL "${value}" AND NOT "${${name}}" STREQUAL "${_NPU_OWNED_${name}}")
            message(FATAL_ERROR "NPU: legacy option ${name}=${${name}} conflicts with the NPU_* configuration (${name}=${value})")
        endif()
    endif()
    set(${name} "${value}" CACHE ${type} "Derived from NPU_* by cmake/npu-config.cmake" FORCE)
    set(_NPU_OWNED_${name} "${value}" CACHE INTERNAL "")
endfunction()

function(_npu_header_int content name out)
    if(content MATCHES "\n[ \t]*#[ \t]*define[ \t]+${name}[ \t]+([0-9]+)[ \t\r]*\n")
        set(${out} "${CMAKE_MATCH_1}" PARENT_SCOPE)
    else()
        set(${out} "" PARENT_SCOPE)
    endif()
endfunction()

# Legacy route: values of an explicit PoTAL header become NPU_* values.
function(_npu_import_potal_header path)
    if(NOT IS_ABSOLUTE "${path}" OR NOT EXISTS "${path}" OR IS_DIRECTORY "${path}")
        message(FATAL_ERROR "NPU: POTAL_HW_PARAMS must name an existing absolute header, got '${path}'")
    endif()
    file(READ "${path}" content)
    set(content "\n${content}\n")
    foreach(pair IN ITEMS DIM:NPU_D BANK_NUM:NPU_SPAD_BANKS BANK_ROWS:NPU_SPAD_BANK_ROWS ACC_ROWS:NPU_ACC_ROWS
            MAX_BYTES:NPU_DMA_MAX_BYTES POTAL_INDEX_ENTRIES:NPU_INDEX_ENTRIES POTAL_SCALE_ENTRIES:NPU_SCALE_ENTRIES)
        string(REPLACE ":" ";" pair "${pair}")
        list(GET pair 0 macro)
        list(GET pair 1 param)
        _npu_header_int("${content}" ${macro} value)
        if(value STREQUAL "")
            continue() # cmake/potal-hw-params.cmake reports the missing field
        endif()
        if(NOT "${${param}}" STREQUAL "" AND NOT "${${param}}" STREQUAL "${value}")
            message(FATAL_ERROR "NPU: ${param}=${${param}} conflicts with ${macro}=${value} in POTAL_HW_PARAMS=${path}")
        endif()
        set(${param} "${value}" PARENT_SCOPE)
    endforeach()
    _npu_header_int("${content}" POTAL_INPUT_BITS input_bits)
    if(input_bits STREQUAL "4")
        set(precision INT4)
    else()
        set(precision INT8)
    endif()
    if(NOT NPU_PRECISION STREQUAL "" AND NOT NPU_PRECISION STREQUAL precision)
        message(FATAL_ERROR "NPU: NPU_PRECISION=${NPU_PRECISION} conflicts with POTAL_HW_PARAMS=${path} (${precision})")
    endif()
    set(NPU_PRECISION "${precision}" PARENT_SCOPE)
endfunction()

# ---- architecture --------------------------------------------------------
set(_npu_arch "${GGML_NPU}")
set(_npu_legacy_route OFF)
if(_npu_arch STREQUAL "")
    set(_npu_legacy_route ON)
    if(GGML_GEMMINI AND (GGML_POTAL OR POTAL_API_ONLY))
        message(FATAL_ERROR "NPU: GGML_GEMMINI and GGML_POTAL/POTAL_API_ONLY select different NPUs; use one GGML_NPU")
    elseif(GGML_GEMMINI)
        set(_npu_arch GEMMINI)
    elseif(GGML_POTAL OR POTAL_API_ONLY)
        set(_npu_arch POTAL)
    else()
        set(_npu_arch NONE)
    endif()
endif()
npu_require_one_of(_npu_arch NONE GEMMINI POTAL)
set(GGML_NPU "${_npu_arch}")

if(DEFINED GEMMINI_SW_PATH AND NOT GEMMINI_SW_PATH STREQUAL "" AND
   NOT GGML_GEMMINI_FP32)
    message(FATAL_ERROR "NPU: GEMMINI_SW_PATH is no longer used: the Gemmini API is internal "
                        "(ggml/src/ggml-gemmini/api). Remove it and pass NPU_* or a CMake preset; "
                        "only the FP32 overlay still reads an external SDK.")
endif()

if(GGML_NPU STREQUAL "NONE")
    foreach(_npu_param IN ITEMS NPU_PRECISION NPU_TRANSPORT NPU_MATMUL_ALGORITHM NPU_D NPU_SPAD_BANKS NPU_SPAD_BANK_ROWS NPU_ACC_BANKS
            NPU_ACC_ROWS NPU_DMA_MAX_BYTES NPU_INDEX_ENTRIES NPU_SCALE_ENTRIES)
        if(NOT "${${_npu_param}}" STREQUAL "")
            message(FATAL_ERROR "NPU: ${_npu_param}=${${_npu_param}} needs GGML_NPU=GEMMINI or POTAL")
        endif()
    endforeach()
    if(NPU_API_ONLY)
        message(FATAL_ERROR "NPU: NPU_API_ONLY=ON needs GGML_NPU=GEMMINI or POTAL")
    endif()
    if(NOT _npu_legacy_route)
        npu_derive_legacy(GGML_GEMMINI OFF BOOL)
        npu_derive_legacy(GGML_POTAL OFF BOOL)
    endif()
    return()
endif()

if(NOT _npu_legacy_route)
    if(GGML_NPU STREQUAL "GEMMINI")
        npu_derive_legacy(GGML_GEMMINI ON BOOL)
        npu_derive_legacy(GGML_POTAL OFF BOOL)
    else()
        npu_derive_legacy(GGML_GEMMINI OFF BOOL)
        if(NPU_API_ONLY)
            npu_derive_legacy(GGML_POTAL OFF BOOL)
        else()
            npu_derive_legacy(GGML_POTAL ON BOOL)
        endif()
    endif()
endif()
if(GGML_NPU STREQUAL "POTAL")
    if(POTAL_API_ONLY AND NOT NPU_API_ONLY)
        set(NPU_API_ONLY ON CACHE BOOL "Configure only the standalone NPU API and its tests" FORCE)
    endif()
    npu_derive_legacy(POTAL_API_ONLY ${NPU_API_ONLY} BOOL)
endif()

# ---- parameter sources ---------------------------------------------------
set(_npu_source "build parameters")
if(GGML_NPU STREQUAL "POTAL" AND DEFINED POTAL_HW_PARAMS AND NOT POTAL_HW_PARAMS STREQUAL "" AND
   NOT POTAL_HW_PARAMS STREQUAL "${_NPU_OWNED_POTAL_HW_PARAMS}")
    _npu_import_potal_header("${POTAL_HW_PARAMS}")
    set(_npu_source "legacy POTAL_HW_PARAMS=${POTAL_HW_PARAMS}")
    set(_npu_legacy_potal_header "${POTAL_HW_PARAMS}")
endif()
if(NPU_TRANSPORT STREQUAL "" AND _npu_legacy_route AND GGML_NPU STREQUAL "POTAL")
    # Previous PoTAL behaviour: real RoCC on RISC-V, host unit tests elsewhere.
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^riscv")
        set(NPU_TRANSPORT ROCC)
    else()
        set(NPU_TRANSPORT REFERENCE)
    endif()
endif()

set(_npu_required NPU_PRECISION NPU_TRANSPORT NPU_D NPU_SPAD_BANKS NPU_SPAD_BANK_ROWS NPU_ACC_ROWS NPU_DMA_MAX_BYTES)
if(GGML_NPU STREQUAL "GEMMINI")
    list(APPEND _npu_required NPU_ACC_BANKS)
else()
    list(APPEND _npu_required NPU_INDEX_ENTRIES NPU_SCALE_ENTRIES)
    if(NOT _npu_legacy_potal_header)
        list(APPEND _npu_required NPU_ACC_BANKS)
    endif()
endif()
foreach(_npu_param IN LISTS _npu_required)
    if("${${_npu_param}}" STREQUAL "")
        message(FATAL_ERROR "NPU: GGML_NPU=${GGML_NPU} requires ${_npu_param} (no default is inferred; "
                            "pass -D${_npu_param}=... or use a CMakePresets.json npu-* preset)")
    endif()
endforeach()

# ---- transport -----------------------------------------------------------
if(NPU_TRANSPORT MATCHES "^(RPC|SIM_DEVICE)$")
    message(FATAL_ERROR "NPU: NPU_TRANSPORT=${NPU_TRANSPORT} is not implemented yet (Phase 1 supports ROCC and REFERENCE)")
endif()
npu_require_one_of(NPU_TRANSPORT ROCC REFERENCE)
if(NPU_TRANSPORT STREQUAL "ROCC")
    if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^riscv64$")
        message(FATAL_ERROR "NPU: NPU_TRANSPORT=ROCC issues RoCC instructions and needs a riscv64 target "
                            "(CMAKE_SYSTEM_PROCESSOR=${CMAKE_SYSTEM_PROCESSOR}); use NPU_TRANSPORT=REFERENCE on this host "
                            "or cmake/toolchains/riscv64-linux.cmake")
    endif()
endif()
if(GGML_NPU STREQUAL "GEMMINI")
    if(NPU_MATMUL_ALGORITHM STREQUAL "")
        if(GGML_GEMMINI_PB)
            set(NPU_MATMUL_ALGORITHM PARTIAL_SUM)
        else()
            set(NPU_MATMUL_ALGORITHM BASELINE)
        endif()
    endif()
    npu_require_one_of(NPU_MATMUL_ALGORITHM BASELINE PARTIAL_SUM)
    if(NPU_MATMUL_ALGORITHM STREQUAL "PARTIAL_SUM")
        if(NOT NPU_PRECISION STREQUAL "INT8" OR NOT NPU_D STREQUAL "32")
            message(FATAL_ERROR "NPU: PARTIAL_SUM requires Gemmini INT8 D32")
        endif()
        npu_derive_legacy(GGML_GEMMINI_PB ON BOOL)
        # PB always lowers K blocks through the WS tiler. REFERENCE selects
        # its command-functional transport, not the generic CPU kernel.
        npu_derive_legacy(GGML_GEMMINI_OPTION WS STRING)
    elseif(NPU_TRANSPORT STREQUAL "REFERENCE")
        npu_derive_legacy(GGML_GEMMINI_PB OFF BOOL)
        npu_derive_legacy(GGML_GEMMINI_OPTION CPU STRING)
    else()
        npu_derive_legacy(GGML_GEMMINI_PB OFF BOOL)
        npu_derive_legacy(GGML_GEMMINI_OPTION WS STRING)
    endif()
elseif(NOT NPU_MATMUL_ALGORITHM STREQUAL "")
    message(FATAL_ERROR "NPU: NPU_MATMUL_ALGORITHM applies to GEMMINI only")
endif()

# ---- precision -----------------------------------------------------------
if(GGML_NPU STREQUAL "GEMMINI")
    npu_require_one_of(NPU_PRECISION INT8 INT16 FP32)
    if(NPU_PRECISION STREQUAL "FP32")
        # Legacy FP32 DIM8 experiment: its own SDK overlay (GEMMINI_SW_PATH)
        # defines the hardware; no header is generated here.
        if(NOT NPU_D EQUAL 8)
            message(FATAL_ERROR "NPU: Gemmini FP32 is the DIM8 experiment lane; NPU_D must be 8")
        endif()
        npu_derive_legacy(GGML_GEMMINI_FP32 ON BOOL)
    elseif(GGML_GEMMINI_FP32)
        message(FATAL_ERROR "NPU: GGML_GEMMINI_FP32=ON needs NPU_PRECISION=FP32")
    elseif(NPU_PRECISION STREQUAL "INT16")
        if(NOT NPU_TRANSPORT STREQUAL "REFERENCE")
            message(FATAL_ERROR "NPU: Gemmini INT16 has no RTL configuration; it is supported with NPU_TRANSPORT=REFERENCE only")
        endif()
        npu_derive_legacy(GGML_GEMMINI_ACTIVATION_BITS 16 STRING)
        npu_derive_legacy(GGML_GEMMINI_WEIGHT_BITS 16 STRING)
    else()
        # INT8 datapath; GGML_GEMMINI_ACTIVATION/WEIGHT_BITS (4 or 8) stay the
        # software quantization width of values held in int8 elements.
        foreach(_npu_bits IN ITEMS GGML_GEMMINI_ACTIVATION_BITS GGML_GEMMINI_WEIGHT_BITS)
            if(DEFINED ${_npu_bits} AND "${${_npu_bits}}" STREQUAL "16")
                message(FATAL_ERROR "NPU: ${_npu_bits}=16 needs NPU_PRECISION=INT16")
            endif()
        endforeach()
    endif()
    npu_require_unset_or_zero(GEMMINI NPU_INDEX_ENTRIES)
    npu_require_unset_or_zero(GEMMINI NPU_SCALE_ENTRIES)
    set(NPU_INDEX_ENTRIES "")
    set(NPU_SCALE_ENTRIES "")
else()
    npu_require_one_of(NPU_PRECISION INT8 INT4)
    # PoTAL quantizes activations and weights at the datapath width.
    if(NPU_PRECISION STREQUAL "INT4")
        npu_derive_legacy(GGML_POTAL_W4A4 ON BOOL)
        set(_npu_potal_bits 4)
    else()
        npu_derive_legacy(GGML_POTAL_W4A4 OFF BOOL)
        set(_npu_potal_bits 8)
    endif()
    if(NOT NPU_API_ONLY)
        npu_derive_legacy(GGML_GEMMINI_ACTIVATION_BITS ${_npu_potal_bits} STRING)
        npu_derive_legacy(GGML_GEMMINI_WEIGHT_BITS ${_npu_potal_bits} STRING)
    endif()
endif()

# ---- geometry ------------------------------------------------------------
npu_validate_common_geometry()
if(NPU_PRECISION STREQUAL "FP32")
    npu_require_one_of(NPU_D 8)
else()
    npu_require_one_of(NPU_D 16 32 64)
endif()
if(GGML_NPU STREQUAL "POTAL")
    npu_require_uint(NPU_INDEX_ENTRIES)
    npu_require_uint(NPU_SCALE_ENTRIES)
    if(NOT NPU_SCALE_ENTRIES EQUAL 256)
        message(FATAL_ERROR "NPU: NPU_SCALE_ENTRIES=${NPU_SCALE_ENTRIES} is unsupported; the PoTAL ISA encodes an 8-bit scale slot (256 entries)")
    endif()
    if(NPU_INDEX_ENTRIES LESS NPU_D)
        message(FATAL_ERROR "NPU: NPU_INDEX_ENTRIES=${NPU_INDEX_ENTRIES} must hold at least one DIM=${NPU_D} K pass")
    endif()
    if(NPU_PRECISION STREQUAL "INT4" AND NOT NPU_D EQUAL 32)
        message(FATAL_ERROR "NPU: PoTAL INT4 packing ABI 1 is defined for D=32 only (POTAL_SP_ROW_BYTES=16)")
    endif()
    # Every PoTAL RTL configuration and the paired planner use 4 SPAD banks and
    # 2 ACC banks; other topologies would need a scheduler redesign.
    if(NOT NPU_SPAD_BANKS EQUAL 4 OR (NOT NPU_ACC_BANKS STREQUAL "" AND NOT NPU_ACC_BANKS EQUAL 2))
        message(FATAL_ERROR "NPU: PoTAL supports NPU_SPAD_BANKS=4 and NPU_ACC_BANKS=2 only "
                            "(got ${NPU_SPAD_BANKS} and ${NPU_ACC_BANKS})")
    endif()
endif()
npu_derive_legacy(GGML_GEMMINI_DIM ${NPU_D} STRING)
if(GGML_NPU STREQUAL "POTAL")
    npu_derive_legacy(POTAL_EXPECTED_DIM ${NPU_D} STRING)
endif()

# ---- derived values ------------------------------------------------------
set(_NPU_ARCH_GEMMINI 0)
set(_NPU_ARCH_POTAL 0)
set(_NPU_ARCH_${GGML_NPU} 1)
set(_NPU_TRANSPORT_ROCC 0)
set(_NPU_TRANSPORT_REFERENCE 0)
if(NPU_TRANSPORT STREQUAL "ROCC")
    set(_NPU_TRANSPORT_ROCC 1)
elseif(NPU_TRANSPORT STREQUAL "REFERENCE")
    set(_NPU_TRANSPORT_REFERENCE 1)
endif()
set(_NPU_ACC_BITS 32)
math(EXPR _NPU_ACC_ROW_BYTES "${NPU_D} * ${_NPU_ACC_BITS} / 8")
math(EXPR _NPU_SPAD_TOTAL_ROWS "${NPU_SPAD_BANKS} * ${NPU_SPAD_BANK_ROWS}")
set(_NPU_PACKING_ABI 0)
if(NPU_PRECISION STREQUAL "FP32")
    set(_NPU_PRECISION_BITS 32)
    set(_NPU_INPUT_STORAGE_BYTES 4)
elseif(NPU_PRECISION STREQUAL "INT16")
    set(_NPU_PRECISION_BITS 16)
    set(_NPU_INPUT_STORAGE_BYTES 2)
elseif(NPU_PRECISION STREQUAL "INT4")
    set(_NPU_PRECISION_BITS 4)
    set(_NPU_INPUT_STORAGE_BYTES 1) # int8_t container in DRAM; packed in SPAD
    set(_NPU_PACKING_ABI 1)
else()
    set(_NPU_PRECISION_BITS 8)
    set(_NPU_INPUT_STORAGE_BYTES 1)
endif()
if(_NPU_PACKING_ABI EQUAL 1)
    set(_NPU_SPAD_ROW_BYTES 16) # potalw4 GemminiConfigs.scala: POTAL_SP_ROW_BYTES
    math(EXPR _NPU_MAX_BLOCK_LEN "${NPU_DMA_MAX_BYTES} / ${_NPU_SPAD_ROW_BYTES}")
else()
    math(EXPR _NPU_SPAD_ROW_BYTES "${NPU_D} * ${_NPU_INPUT_STORAGE_BYTES}")
    if(_NPU_SPAD_ROW_BYTES GREATER NPU_DMA_MAX_BYTES)
        set(_NPU_MAX_BLOCK_LEN 1)
    else()
        math(EXPR _NPU_MAX_BLOCK_LEN "${NPU_DMA_MAX_BYTES} / ${_NPU_SPAD_ROW_BYTES}")
    endif()
endif()
if(_NPU_ACC_ROW_BYTES GREATER NPU_DMA_MAX_BYTES)
    set(_NPU_MAX_BLOCK_LEN_ACC 1)
else()
    math(EXPR _NPU_MAX_BLOCK_LEN_ACC "${NPU_DMA_MAX_BYTES} / ${_NPU_ACC_ROW_BYTES}")
endif()
math(EXPR _NPU_SPAD_CAPACITY_BYTES "${_NPU_SPAD_TOTAL_ROWS} * ${_NPU_SPAD_ROW_BYTES}")
math(EXPR _NPU_ACC_CAPACITY_BYTES "${NPU_ACC_ROWS} * ${_NPU_ACC_ROW_BYTES}")
if(NPU_ACC_BANKS STREQUAL "")
    set(_NPU_ACC_BANKS_VALUE 0)
    set(_NPU_ACC_BANK_ROWS 0)
    set(_NPU_ACC_BANKS_JSON null)
    set(_NPU_ACC_BANK_ROWS_JSON null)
else()
    set(_NPU_ACC_BANKS_VALUE ${NPU_ACC_BANKS})
    math(EXPR _NPU_ACC_BANK_ROWS "${NPU_ACC_ROWS} / ${NPU_ACC_BANKS}")
    set(_NPU_ACC_BANKS_JSON ${NPU_ACC_BANKS})
    set(_NPU_ACC_BANK_ROWS_JSON ${_NPU_ACC_BANK_ROWS})
endif()
if(GGML_NPU STREQUAL "POTAL")
    set(_NPU_INDEX_ENTRIES_VALUE ${NPU_INDEX_ENTRIES})
    set(_NPU_SCALE_ENTRIES_VALUE ${NPU_SCALE_ENTRIES})
    set(_NPU_ISA_REVISION "potal-chipyard-aisa-potal-fab3d262")
else()
    set(_NPU_INDEX_ENTRIES_VALUE 0)
    set(_NPU_SCALE_ENTRIES_VALUE 0)
    set(_NPU_ISA_REVISION "gemmini-aisa-185293e8")
endif()

# ---- generated files -----------------------------------------------------
file(MAKE_DIRECTORY "${NPU_GENERATED_DIR}/npu")
set(_NPU_CONFIG_HASH "")
configure_file("${_NPU_TEMPLATE_DIR}/npu_config.json.in" "${NPU_GENERATED_DIR}/npu_config.json" @ONLY)
file(SHA256 "${NPU_GENERATED_DIR}/npu_config.json" _NPU_CONFIG_HASH)
configure_file("${_NPU_TEMPLATE_DIR}/npu_config.h.in" "${NPU_GENERATED_DIR}/npu_config.h" @ONLY)

if(_NPU_PRECISION_BITS EQUAL 16)
    set(_NPU_ELEM_T int16_t)
    set(_NPU_ELEM_MAX 32767)
    set(_NPU_ELEM_MIN -32768)
    set(_NPU_ELEM_C_MAX INT16_MAX)
    set(_NPU_ELEM_C_MIN INT16_MIN)
elseif(_NPU_PRECISION_BITS EQUAL 4)
    set(_NPU_ELEM_MAX 7)
    set(_NPU_ELEM_MIN -8)
else()
    set(_NPU_ELEM_T int8_t)
    set(_NPU_ELEM_MAX 127)
    set(_NPU_ELEM_MIN -128)
    set(_NPU_ELEM_C_MAX INT8_MAX)
    set(_NPU_ELEM_C_MIN INT8_MIN)
endif()
# GemminiConfigs.scala generateHeader: the formula when one row fits MAX_BYTES, else 1.
if(_NPU_SPAD_ROW_BYTES GREATER NPU_DMA_MAX_BYTES)
    set(_npu_block_line "#define MAX_BLOCK_LEN 1")
else()
    set(_npu_block_line "#define MAX_BLOCK_LEN (MAX_BYTES/(DIM*${_NPU_INPUT_STORAGE_BYTES}))")
endif()
if(_NPU_ACC_ROW_BYTES GREATER NPU_DMA_MAX_BYTES)
    set(_npu_block_acc_line "#define MAX_BLOCK_LEN_ACC 1")
else()
    set(_npu_block_acc_line "#define MAX_BLOCK_LEN_ACC (MAX_BYTES/(DIM*4))")
endif()

unset(_npu_params_header)
if(NPU_PRECISION STREQUAL "FP32")
    set(_npu_source "legacy FP32 overlay GEMMINI_SW_PATH=${GEMMINI_SW_PATH}")
elseif(GGML_NPU STREQUAL "GEMMINI")
    set(_NPU_GEMMINI_MAX_BLOCK_LEN_LINE "${_npu_block_line}")
    set(_NPU_GEMMINI_MAX_BLOCK_LEN_ACC_LINE "${_npu_block_acc_line}")
    set(_npu_params_header "${NPU_GENERATED_DIR}/gemmini_params.h")
    configure_file("${_NPU_TEMPLATE_DIR}/gemmini_params.h.in" "${_npu_params_header}" @ONLY)
elseif(NOT _npu_legacy_potal_header)
    set(_NPU_SCALE_SLOT_BITS 8)
    if(_NPU_PACKING_ABI EQUAL 1)
        set(_NPU_POTAL_BLOCK_LINES "#define POTAL_INPUT_BITS 4\n#define POTAL_ACC_BITS 32\n#define POTAL_PACKING_ABI 1\n#define POTAL_SP_ROW_BYTES ${_NPU_SPAD_ROW_BYTES}\n#define MAX_BLOCK_LEN ${_NPU_MAX_BLOCK_LEN}\n${_npu_block_acc_line}")
    else()
        set(_NPU_POTAL_BLOCK_LINES "${_npu_block_line}\n${_npu_block_acc_line}")
    endif()
    string(REPLACE "\\n" "\n" _NPU_POTAL_BLOCK_LINES "${_NPU_POTAL_BLOCK_LINES}")
    # File names of the RTL generator (PoTalConfigs.scala headerFileName); the
    # W4A4 backend checks the INT4 name.
    if(_NPU_PACKING_ABI EQUAL 1)
        set(_npu_potal_header_name "gemmini_potal_w4a4_d${NPU_D}_params.h")
    elseif(NPU_D EQUAL 16)
        set(_npu_potal_header_name "gemmini_potal_params.h")
    else()
        set(_npu_potal_header_name "gemmini_potal_d${NPU_D}_params.h")
    endif()
    set(_npu_params_header "${NPU_GENERATED_DIR}/npu/${_npu_potal_header_name}")
    configure_file("${_NPU_TEMPLATE_DIR}/potal_params.h.in" "${_npu_params_header}" @ONLY)
    npu_derive_legacy(POTAL_HW_PARAMS "${_npu_params_header}" FILEPATH)
endif()

# RTL reference headers whose content the generator must reproduce byte for
# byte for the configurations used in the lab's experiments.
set(_npu_reference_sha256
    "GEMMINI:INT8:16:4:4096:2:1024::=d6db18f8edbfeb854780286d8c2afb81ae3b9103224522478a9047d5550272e5"
    "POTAL:INT8:16:4:4096:2:1024:32:256=b4cd8b9efef9f3cd9b598ae3cc84af6478f83e628ed73d94a77fb1c452df1fc7"
    "POTAL:INT8:32:4:2048:2:512:64:256=3580b84f15ed8de9f9d311edea64eaa7c1f9bb308b1552a8f79453ec2c05ea33"
    "POTAL:INT8:64:4:1024:2:256:128:256=5c3b625cf21748322dab3ef32112f0edf43aee89e0c17e763c097ddda484a506"
    "POTAL:INT4:32:4:2048:2:512:64:256=4b29d84eff0bd4a01c6db0519c79904b808ba0f2dc9bc3727cb395ca9e30eaf0")
set(_npu_key "${GGML_NPU}:${NPU_PRECISION}:${NPU_D}:${NPU_SPAD_BANKS}:${NPU_SPAD_BANK_ROWS}:${NPU_ACC_BANKS}:${NPU_ACC_ROWS}:${NPU_INDEX_ENTRIES}:${NPU_SCALE_ENTRIES}")
set(_npu_rtl_status "unverified: geometry has no RTL reference header in this tree")
if(NPU_PRECISION STREQUAL "FP32")
    set(_npu_rtl_status "unverified: FP32 DIM8 overlay is outside the audited RTL")
endif()
set(_npu_params_sha256 "")
if(_npu_legacy_potal_header)
    file(SHA256 "${_npu_legacy_potal_header}" _npu_params_sha256)
    set(_npu_rtl_status "legacy header supplied by POTAL_HW_PARAMS (validated by cmake/potal-hw-params.cmake)")
elseif(_npu_params_header)
    file(SHA256 "${_npu_params_header}" _npu_params_sha256)
endif()
foreach(_npu_ref IN LISTS _npu_reference_sha256)
    if(_npu_ref MATCHES "^${_npu_key}=(.*)$" AND NOT _npu_legacy_potal_header)
        if(CMAKE_MATCH_1 STREQUAL _npu_params_sha256)
            set(_npu_rtl_status "matches RTL reference header sha256 ${CMAKE_MATCH_1}")
        else()
            message(FATAL_ERROR "NPU: generated parameter header for ${_npu_key} differs from its RTL reference "
                                "(sha256 ${_npu_params_sha256}, expected ${CMAKE_MATCH_1})")
        endif()
    endif()
endforeach()

find_package(Git QUIET)
set(_npu_sw_commit "unknown")
if(GIT_EXECUTABLE)
    execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${CMAKE_CURRENT_LIST_DIR}/.." rev-parse HEAD
        OUTPUT_VARIABLE _npu_sw_commit ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${CMAKE_CURRENT_LIST_DIR}/.." status --porcelain --untracked-files=no
        OUTPUT_VARIABLE _npu_sw_dirty ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(_npu_sw_dirty)
        string(APPEND _npu_sw_commit " (tracked files modified)")
    endif()
endif()
if(GGML_NPU STREQUAL "GEMMINI")
    set(_npu_rtl_ref "https://github.com/ajou-aisa/gemmini-aisa aisa-customizations 185293e87cbbc2a336410f49a3fd73d01ed8d04e")
else()
    set(_npu_rtl_ref "https://github.com/ajou-aisa/chipyard-aisa potal fab3d2620c0d8b688bd8c151ce9f3c775737175d (generators/potal)")
endif()
file(WRITE "${NPU_GENERATED_DIR}/npu_build_manifest.json" "{
  \"schema\": \"socmulator.npu_build_manifest/1\",
  \"architecture\": \"${GGML_NPU}\",
  \"precision\": \"${NPU_PRECISION}\",
  \"geometry\": {\"d\": ${NPU_D}, \"spad_banks\": ${NPU_SPAD_BANKS}, \"spad_bank_rows\": ${NPU_SPAD_BANK_ROWS}, \"acc_banks\": ${_NPU_ACC_BANKS_JSON}, \"acc_rows\": ${NPU_ACC_ROWS}},
  \"potal\": {\"index_entries\": ${_NPU_INDEX_ENTRIES_VALUE}, \"scale_entries\": ${_NPU_SCALE_ENTRIES_VALUE}},
  \"dma\": {\"max_bytes\": ${NPU_DMA_MAX_BYTES}},
  \"isa_revision\": \"${_NPU_ISA_REVISION}\",
  \"packing_abi\": ${_NPU_PACKING_ABI},
  \"transport\": \"${NPU_TRANSPORT}\",
  \"matmul_algorithm\": \"${NPU_MATMUL_ALGORITHM}\",
  \"execution_policy\": \"${GGML_GEMMINI_OPTION}\",
  \"api_only\": \"${NPU_API_ONLY}\",
  \"profiling\": \"${NPU_PROFILING}\",
  \"parameter_source\": \"${_npu_source}\",
  \"parameter_header\": \"${_npu_params_header}${_npu_legacy_potal_header}\",
  \"parameter_header_sha256\": \"${_npu_params_sha256}\",
  \"rtl_reference\": \"${_npu_rtl_ref}\",
  \"rtl_validation\": \"${_npu_rtl_status}\",
  \"software_commit\": \"${_npu_sw_commit}\",
  \"c_compiler\": \"${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION} ${CMAKE_C_COMPILER}\",
  \"cxx_compiler\": \"${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION} ${CMAKE_CXX_COMPILER}\",
  \"toolchain_file\": \"${CMAKE_TOOLCHAIN_FILE}\",
  \"system_processor\": \"${CMAKE_SYSTEM_PROCESSOR}\",
  \"build_type\": \"${CMAKE_BUILD_TYPE}\",
  \"config_hash\": \"${_NPU_CONFIG_HASH}\"
}
")

message(STATUS "NPU: ${GGML_NPU} ${NPU_PRECISION} D=${NPU_D} SPAD=${NPU_SPAD_BANKS}x${NPU_SPAD_BANK_ROWS} "
               "ACC=${NPU_ACC_ROWS} rows/${NPU_ACC_BANKS} banks DMA=${NPU_DMA_MAX_BYTES} "
               "transport=${NPU_TRANSPORT} (${_npu_rtl_status})")
