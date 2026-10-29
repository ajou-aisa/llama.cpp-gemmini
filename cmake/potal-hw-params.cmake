# PoTal hardware parameter contract.
#
# The header produced by PoTal RTL elaboration (gemmini_potal*_params.h) is the
# single source of truth for the hardware geometry. This module validates it at
# configure time and exposes the values under POTAL_HW_* names. It never edits
# the source header, never falls back to another header and never infers a
# missing value.
#
# Inputs (cache/variables): POTAL_HW_PARAMS, POTAL_EXPECTED_DIM, POTAL_HW_CONFIG.
# Outputs in <out_dir>:
#   original-potal-params.h  byte-exact copy of the selected header
#   potal_hw_params.h        POTAL_HW_* adapter (no generic Gemmini names)
#   potal_build_manifest.txt provenance record
# Parent-scope results: POTAL_HW_GENERATED_DIR, POTAL_HW_DIM.

# Exactly one "#define NAME <decimal>" line; anything else is a hard error.
# `content` is the header with every line wrapped as "\n<line>\n" (see below),
# so each regex sees whole lines and adjacent duplicates are both counted.
function(_potal_hw_int content name out)
    string(REGEX MATCHALL "\n[ \t]*#[ \t]*define[ \t]+${name}([ \t\r][^\n]*)?\n" lines "${content}")
    list(LENGTH lines count)
    if(count EQUAL 0)
        message(FATAL_ERROR "PoTal: POTAL_HW_PARAMS does not define required parameter ${name}")
    endif()
    if(NOT count EQUAL 1)
        message(FATAL_ERROR "PoTal: POTAL_HW_PARAMS defines ${name} ${count} times")
    endif()
    if(content MATCHES "\n[ \t]*#[ \t]*undef[ \t]+${name}[ \t\r\n]")
        message(FATAL_ERROR "PoTal: POTAL_HW_PARAMS undefines ${name}")
    endif()
    if(NOT lines MATCHES "^\n[ \t]*#[ \t]*define[ \t]+${name}[ \t]+([0-9]+)[ \t\r]*\n$")
        message(FATAL_ERROR "PoTal: ${name} must be a plain decimal integer: '${lines}'")
    endif()
    set(value "${CMAKE_MATCH_1}")
    string(LENGTH "${value}" digits)
    if(digits GREATER 9 OR value MATCHES "^0[0-9]")
        message(FATAL_ERROR "PoTal: ${name}=${value} is out of the accepted range")
    endif()
    set(${out} "${value}" PARENT_SCOPE)
endfunction()

# Exactly one "typedef <type> NAME;" and it must be the supported type.
# `;` is masked as <SEMICOLON> in `content` so matches stay single list items.
function(_potal_hw_typedef content name expected)
    string(REGEX MATCHALL "typedef[ \t]+[A-Za-z0-9_ \t]+[ \t]${name}[ \t]*<SEMICOLON>" decls "${content}")
    list(LENGTH decls count)
    if(NOT count EQUAL 1)
        message(FATAL_ERROR "PoTal: POTAL_HW_PARAMS must declare ${name} exactly once (found ${count})")
    endif()
    if(NOT decls MATCHES "^typedef[ \t]+${expected}[ \t]+${name}[ \t]*<SEMICOLON>$")
        string(REPLACE "<SEMICOLON>" ";" decls "${decls}")
        message(FATAL_ERROR
            "PoTal: unsupported precision '${decls}'; this API supports A8W8 with "
            "typedef ${expected} ${name}")
    endif()
endfunction()

function(_potal_hw_pow2 name value)
    math(EXPR low "${value} & (${value} - 1)")
    if(value LESS 1 OR NOT low EQUAL 0)
        message(FATAL_ERROR "PoTal: ${name}=${value} must be a positive power of two")
    endif()
endfunction()

function(_potal_git_head dir out)
    set(head "unknown")
    find_package(Git QUIET)
    if(GIT_EXECUTABLE)
        execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${dir}" rev-parse HEAD
            RESULT_VARIABLE rc OUTPUT_VARIABLE sha ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
        if(rc EQUAL 0)
            execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${dir}" status --porcelain --untracked-files=no
                OUTPUT_VARIABLE dirty ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
            set(head "${sha}")
            if(dirty)
                string(APPEND head " (tracked files modified)")
            endif()
        endif()
    endif()
    set(${out} "${head}" PARENT_SCOPE)
endfunction()

function(potal_configure_hw_params out_dir)
    if(NOT DEFINED POTAL_HW_PARAMS OR POTAL_HW_PARAMS STREQUAL "")
        message(FATAL_ERROR "PoTal: POTAL_HW_PARAMS is required (absolute path to the RTL-generated header)")
    endif()
    if(NOT IS_ABSOLUTE "${POTAL_HW_PARAMS}")
        message(FATAL_ERROR "PoTal: POTAL_HW_PARAMS must be an absolute path, got '${POTAL_HW_PARAMS}'")
    endif()
    if(NOT EXISTS "${POTAL_HW_PARAMS}" OR IS_DIRECTORY "${POTAL_HW_PARAMS}")
        message(FATAL_ERROR "PoTal: POTAL_HW_PARAMS is not a readable file: ${POTAL_HW_PARAMS}")
    endif()
    if(NOT DEFINED POTAL_EXPECTED_DIM OR NOT POTAL_EXPECTED_DIM MATCHES "^(16|32|64)$")
        message(FATAL_ERROR "PoTal: POTAL_EXPECTED_DIM must be 16, 32, or 64, got '${POTAL_EXPECTED_DIM}'")
    endif()

    get_filename_component(source "${POTAL_HW_PARAMS}" REALPATH)
    # file(READ) aborts configure if the file cannot be read.
    file(READ "${source}" content)
    file(SHA256 "${source}" sha256)
    if(DEFINED POTAL_TARGET_PARAMS)
        if(NOT IS_ABSOLUTE "${POTAL_TARGET_PARAMS}" OR NOT EXISTS "${POTAL_TARGET_PARAMS}" OR
           IS_DIRECTORY "${POTAL_TARGET_PARAMS}")
            message(FATAL_ERROR "PoTal: POTAL_TARGET_PARAMS must name the target's generated parameter header")
        endif()
        file(SHA256 "${POTAL_TARGET_PARAMS}" target_sha256)
        if(NOT sha256 STREQUAL target_sha256)
            message(FATAL_ERROR "PoTal: target parameter mismatch: selected ${source} does not match ${POTAL_TARGET_PARAMS}")
        endif()
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${POTAL_TARGET_PARAMS}")
    endif()
    # Wrap every line as "\n<line>\n" and mask `;` (a CMake list separator).
    string(REPLACE "\n" "\n\n" content "\n${content}\n")
    string(REPLACE ";" "<SEMICOLON>" content "${content}")

    if(NOT content MATCHES "\n[ \t]*#[ \t]*ifndef[ \t]+GEMMINI_PARAMS_H[ \t\r]*\n\n[ \t]*#[ \t]*define[ \t]+GEMMINI_PARAMS_H[ \t\r]*\n")
        message(FATAL_ERROR "PoTal: ${source} is not a generated Gemmini/PoTal parameter header (GEMMINI_PARAMS_H guard missing)")
    endif()

    foreach(name IN ITEMS XCUSTOM_ACC DIM ADDR_LEN BANK_NUM BANK_ROWS ACC_ROWS MAX_BYTES
            POTAL_SCALE_ENTRIES POTAL_INDEX_ENTRIES)
        _potal_hw_int("${content}" ${name} ${name})
    endforeach()
    _potal_hw_typedef("${content}" elem_t int8_t)
    _potal_hw_typedef("${content}" acc_t int32_t)

    # PoTal RTL decodes scale and loop commands on RoCC custom-3 only.
    if(NOT XCUSTOM_ACC EQUAL 3)
        message(FATAL_ERROR "PoTal: XCUSTOM_ACC=${XCUSTOM_ACC}; PoTal hardware contract requires custom-3")
    endif()
    if(NOT DIM MATCHES "^(16|32|64)$")
        message(FATAL_ERROR "PoTal: header DIM=${DIM}; supported DIM is 16, 32, or 64")
    endif()
    if(NOT DIM EQUAL POTAL_EXPECTED_DIM)
        message(FATAL_ERROR "PoTal: DIM mismatch: header reports ${DIM}, POTAL_EXPECTED_DIM=${POTAL_EXPECTED_DIM}")
    endif()
    if(DEFINED GGML_GEMMINI_DIM AND NOT "${GGML_GEMMINI_DIM}" STREQUAL "${DIM}")
        message(FATAL_ERROR "PoTal: DIM mismatch: header reports ${DIM}, GGML_GEMMINI_DIM=${GGML_GEMMINI_DIM}")
    endif()
    if(DEFINED POTAL_TARGET_DIM)
        if(NOT POTAL_TARGET_DIM MATCHES "^(16|32)$" OR NOT "${POTAL_TARGET_DIM}" STREQUAL "${DIM}")
            message(FATAL_ERROR "PoTal: target DIM mismatch: header reports ${DIM}, POTAL_TARGET_DIM=${POTAL_TARGET_DIM}")
        endif()
    endif()
    if(POTAL_HW_CONFIG MATCHES "^FireSimQuadRocketSinglePoTalD(16|32)Config$")
        if(NOT "${CMAKE_MATCH_1}" STREQUAL "${DIM}")
            message(FATAL_ERROR "PoTal: config DIM mismatch: ${POTAL_HW_CONFIG}, header reports ${DIM}")
        endif()
    endif()
    # LocalAddr and PRELOAD/COMPUTE rs fields are 32 bits wide.
    if(NOT ADDR_LEN EQUAL 32)
        message(FATAL_ERROR "PoTal: ADDR_LEN=${ADDR_LEN}; supported value is 32")
    endif()
    _potal_hw_pow2(BANK_NUM ${BANK_NUM})
    _potal_hw_pow2(BANK_ROWS ${BANK_ROWS})
    _potal_hw_pow2(ACC_ROWS ${ACC_ROWS})
    math(EXPR max_bytes_rem "${MAX_BYTES} % ${DIM}")
    if(MAX_BYTES LESS DIM OR NOT max_bytes_rem EQUAL 0)
        message(FATAL_ERROR "PoTal: MAX_BYTES=${MAX_BYTES} must be a positive multiple of DIM=${DIM} (int8 rows)")
    endif()
    # LoopMatmul splits SPAD and ACC between two concurrent loops; each half
    # must hold at least one A/B tile pair and one C tile.
    math(EXPR spad_half "${BANK_NUM} * ${BANK_ROWS} / 2")
    math(EXPR acc_half "${ACC_ROWS} / 2")
    math(EXPR ab_tile_rows "2 * ${DIM}")
    if(spad_half LESS ab_tile_rows OR acc_half LESS DIM)
        message(FATAL_ERROR "PoTal: SPAD/ACC capacity too small for DIM=${DIM}")
    endif()
    # potal.h encodes scale keys with an 8-bit slot (POTAL_SCALE_ENTRIES 256);
    # the H2 Index Table holds whole DIM-wide K passes (ExecuteController
    # requires at least one).
    if(NOT POTAL_SCALE_ENTRIES EQUAL 256)
        message(FATAL_ERROR "PoTal: POTAL_SCALE_ENTRIES=${POTAL_SCALE_ENTRIES}; potal.h supports 256 scale entries")
    endif()
    if(POTAL_INDEX_ENTRIES LESS DIM)
        message(FATAL_ERROR "PoTal: POTAL_INDEX_ENTRIES=${POTAL_INDEX_ENTRIES} must hold at least one DIM=${DIM} K pass")
    endif()

    file(MAKE_DIRECTORY "${out_dir}")
    # configure_file records the source as a configure dependency: editing the
    # header re-runs this validation on the next build.
    configure_file("${source}" "${out_dir}/original-potal-params.h" COPYONLY)

    set(adapter "/* Generated by cmake/potal-hw-params.cmake. Do not edit. */
/* Source: ${source} */
/* SHA-256: ${sha256} */
#ifndef POTAL_HW_PARAMS_H
#define POTAL_HW_PARAMS_H

#define POTAL_HW_XCUSTOM   ${XCUSTOM_ACC}
#define POTAL_HW_DIM       ${DIM}
#define POTAL_HW_ADDR_LEN  ${ADDR_LEN}
#define POTAL_HW_BANK_NUM  ${BANK_NUM}
#define POTAL_HW_BANK_ROWS ${BANK_ROWS}
#define POTAL_HW_ACC_ROWS  ${ACC_ROWS}
#define POTAL_HW_MAX_BYTES ${MAX_BYTES}
#define POTAL_HW_SCALE_ENTRIES ${POTAL_SCALE_ENTRIES}
#define POTAL_HW_INDEX_ENTRIES ${POTAL_INDEX_ENTRIES}
#define POTAL_HW_PARAMS_SHA256 \"${sha256}\"

#endif /* POTAL_HW_PARAMS_H */
")
    # Rewrite only on change so unchanged reconfigures do not force rebuilds.
    file(WRITE "${out_dir}/potal_hw_params.h.tmp" "${adapter}")
    configure_file("${out_dir}/potal_hw_params.h.tmp" "${out_dir}/potal_hw_params.h" COPYONLY)

    get_filename_component(source_dir "${source}" DIRECTORY)
    _potal_git_head("${PROJECT_SOURCE_DIR}" software_head)
    _potal_git_head("${source_dir}" hardware_head)
    if(POTAL_HW_CONFIG)
        set(hw_config "${POTAL_HW_CONFIG}")
    else()
        set(hw_config "unspecified")
    endif()
    file(WRITE "${out_dir}/potal_build_manifest.txt"
"# PoTal API build manifest (recorded at configure time).
# The header SHA-256 identifies the parameter header only, not the full RTL;
# record the RTL revision and Chipyard config separately.
potal_hw_params=${POTAL_HW_PARAMS}
potal_hw_params_realpath=${source}
potal_hw_params_sha256=${sha256}
potal_expected_dim=${POTAL_EXPECTED_DIM}
potal_target_dim=${POTAL_TARGET_DIM}
potal_target_params=${POTAL_TARGET_PARAMS}
target_params_sha256=${target_sha256}
ggml_gemmini_dim=${GGML_GEMMINI_DIM}
dim=${DIM}
xcustom_acc=${XCUSTOM_ACC}
addr_len=${ADDR_LEN}
bank_num=${BANK_NUM}
bank_rows=${BANK_ROWS}
acc_rows=${ACC_ROWS}
max_bytes=${MAX_BYTES}
scale_entries=${POTAL_SCALE_ENTRIES}
index_entries=${POTAL_INDEX_ENTRIES}
precision=A8W8 elem_t=int8_t acc_t=int32_t
hw_config=${hw_config}
hw_params_repo_head=${hardware_head}
software_head=${software_head}
cmake_build_type=${CMAKE_BUILD_TYPE}
cmake_system_processor=${CMAKE_SYSTEM_PROCESSOR}
cmake_toolchain_file=${CMAKE_TOOLCHAIN_FILE}
cmake_c_compiler=${CMAKE_C_COMPILER}
potal_api_only=${POTAL_API_ONLY}
")

    message(STATUS "PoTal hardware parameters: DIM=${DIM} XCUSTOM=${XCUSTOM_ACC} "
        "BANK_NUM=${BANK_NUM} BANK_ROWS=${BANK_ROWS} ACC_ROWS=${ACC_ROWS} MAX_BYTES=${MAX_BYTES} "
        "SCALE_ENTRIES=${POTAL_SCALE_ENTRIES} INDEX_ENTRIES=${POTAL_INDEX_ENTRIES}")
    message(STATUS "PoTal hardware header: ${source} sha256=${sha256}")
    message(STATUS "PoTal build manifest: ${out_dir}/potal_build_manifest.txt")

    set(POTAL_HW_GENERATED_DIR "${out_dir}" PARENT_SCOPE)
    set(POTAL_HW_DIM "${DIM}" PARENT_SCOPE)
endfunction()
