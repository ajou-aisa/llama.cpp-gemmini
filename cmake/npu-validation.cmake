# Validation helpers for cmake/npu-config.cmake. Every check fails configure;
# nothing is clamped or replaced by a default.

# Positive decimal integer without leading zero, at most 2^31-1.
function(npu_require_uint name)
    set(value "${${name}}")
    if(NOT value MATCHES "^[1-9][0-9]*$")
        message(FATAL_ERROR "NPU: ${name} must be a positive decimal integer, got '${value}'")
    endif()
    string(LENGTH "${value}" digits)
    if(digits GREATER 10 OR (digits EQUAL 10 AND value STRGREATER "2147483647"))
        message(FATAL_ERROR "NPU: ${name}=${value} exceeds 2147483647")
    endif()
endfunction()

function(npu_require_pow2 name)
    npu_require_uint(${name})
    math(EXPR low "${${name}} & (${${name}} - 1)")
    if(NOT low EQUAL 0)
        message(FATAL_ERROR "NPU: ${name}=${${name}} must be a power of two")
    endif()
endfunction()

function(npu_require_one_of name)
    set(value "${${name}}")
    if(NOT value IN_LIST ARGN)
        string(JOIN "|" allowed ${ARGN})
        message(FATAL_ERROR "NPU: ${name}='${value}' is not supported here (supported: ${allowed})")
    endif()
endfunction()

function(npu_require_unset_or_zero arch name)
    if(DEFINED ${name} AND NOT "${${name}}" STREQUAL "" AND NOT "${${name}}" STREQUAL "0")
        message(FATAL_ERROR "NPU: ${name}=${${name}} does not apply to ${arch}")
    endif()
endfunction()

# 64-bit products stay exact in CMake math(); reject anything that would not
# fit the 32-bit fields the encoders use.
function(npu_require_le name value limit what)
    if(value GREATER limit)
        message(FATAL_ERROR "NPU: ${name}: ${what} (${value} > ${limit})")
    endif()
endfunction()

# Geometry shared by Gemmini and PoTAL (LoopMatmul, LocalAddr, tiled_matmul_auto).
function(npu_validate_common_geometry)
    foreach(name IN ITEMS NPU_D NPU_SPAD_BANKS NPU_SPAD_BANK_ROWS NPU_ACC_ROWS NPU_DMA_MAX_BYTES)
        npu_require_pow2(${name})
    endforeach()
    if(DEFINED NPU_ACC_BANKS AND NOT NPU_ACC_BANKS STREQUAL "")
        npu_require_pow2(NPU_ACC_BANKS)
        math(EXPR rem "${NPU_ACC_ROWS} % ${NPU_ACC_BANKS}")
        if(NOT rem EQUAL 0 OR NPU_ACC_ROWS LESS NPU_ACC_BANKS)
            message(FATAL_ERROR "NPU: NPU_ACC_ROWS=${NPU_ACC_ROWS} (total rows) is not divisible by NPU_ACC_BANKS=${NPU_ACC_BANKS}")
        endif()
    endif()
    # The RTL header generators fix MAX_BYTES at 64 (GemminiConfigs.scala
    # generateHeader); the software tilers are validated only for that value.
    if(NOT NPU_DMA_MAX_BYTES EQUAL 64)
        message(FATAL_ERROR "NPU: NPU_DMA_MAX_BYTES=${NPU_DMA_MAX_BYTES} is unsupported; the RTL header contract fixes MAX_BYTES=64")
    endif()
    math(EXPR spad_rows "${NPU_SPAD_BANKS} * ${NPU_SPAD_BANK_ROWS}")
    # LocalAddr.scala asserts data bits + 6 metadata bits < 32: at most 25 row bits.
    npu_require_le(NPU_SPAD_BANK_ROWS ${spad_rows} 33554432 "SPAD rows exceed the 25-bit local address")
    npu_require_le(NPU_ACC_ROWS ${NPU_ACC_ROWS} 33554432 "ACC rows exceed the 25-bit local address")
    # LOOP_WS runs two loops concurrently: each half must hold one A/B tile
    # pair and one C tile (also the minimum tiled_matmul_auto can plan).
    math(EXPR spad_half "${spad_rows} / 2")
    math(EXPR acc_half "${NPU_ACC_ROWS} / 2")
    math(EXPR ab_rows "2 * ${NPU_D}")
    if(spad_half LESS ab_rows OR acc_half LESS NPU_D)
        message(FATAL_ERROR "NPU: SPAD/ACC capacity cannot hold one double-buffered DIM=${NPU_D} tile "
                            "(SPAD half ${spad_half} rows, need ${ab_rows}; ACC half ${acc_half} rows, need ${NPU_D})")
    endif()
endfunction()
