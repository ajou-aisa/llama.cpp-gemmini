if(NOT DEFINED TEST_SCRIPT OR NOT EXISTS "${TEST_SCRIPT}")
    message(FATAL_ERROR "TEST_SCRIPT must point to build-arm64.sh")
endif()
if(NOT DEFINED TEST_ROOT)
    message(FATAL_ERROR "TEST_ROOT is required")
endif()

file(REMOVE_RECURSE "${TEST_ROOT}")
file(MAKE_DIRECTORY "${TEST_ROOT}/bin")

file(WRITE "${TEST_ROOT}/bin/uname" [=[#!/bin/bash
printf '%s\n' Linux
]=])
file(WRITE "${TEST_ROOT}/bin/nproc" [=[#!/bin/bash
printf '%s\n' 7
]=])
file(WRITE "${TEST_ROOT}/bin/getconf" [=[#!/bin/bash
printf '%s\n' 7
]=])
file(WRITE "${TEST_ROOT}/bin/sysctl" [=[#!/bin/bash
printf '%s\n' 'sysctl must not be used on Linux' >&2
exit 97
]=])
file(WRITE "${TEST_ROOT}/bin/cmake" [=[#!/bin/bash
printf 'cmake:%s\n' "$*" >> "$CONTRACT_LOG"
]=])
execute_process(
    COMMAND chmod +x
        "${TEST_ROOT}/bin/uname"
        "${TEST_ROOT}/bin/nproc"
        "${TEST_ROOT}/bin/getconf"
        "${TEST_ROOT}/bin/sysctl"
        "${TEST_ROOT}/bin/cmake")

set(contract_log "${TEST_ROOT}/commands.log")
execute_process(
    COMMAND "${CMAKE_COMMAND}" -E env
        "PATH=${TEST_ROOT}/bin:$ENV{PATH}"
        "CONTRACT_LOG=${contract_log}"
        "BUILD_DIR=${TEST_ROOT}/build"
        "BUILD_JOBS="
        bash "${TEST_SCRIPT}"
    WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/.."
    RESULT_VARIABLE rc
    OUTPUT_VARIABLE stdout
    ERROR_VARIABLE stderr)
if(NOT rc EQUAL 0)
    file(REMOVE_RECURSE "${TEST_ROOT}")
    message(FATAL_ERROR
        "build-arm64.sh failed Linux CPU detection: ${stdout}\n${stderr}")
endif()

file(READ "${contract_log}" commands)
string(FIND "${commands}"
    "cmake:--build ${TEST_ROOT}/build --target llama-cli llama-perplexity llama-quantize -j7"
    build_at)
file(REMOVE_RECURSE "${TEST_ROOT}")
if(build_at EQUAL -1)
    message(FATAL_ERROR
        "build-arm64.sh did not use the detected Linux CPU count:\n${commands}")
endif()
