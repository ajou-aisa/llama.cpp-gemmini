# RISC-V 64 Linux cross toolchain (no machine-specific paths).
#
# Inputs, as -D cache values or environment variables:
#   RISCV_TOOLCHAIN_ROOT    directory containing bin/<prefix>gcc (empty: search PATH)
#   RISCV_TOOLCHAIN_PREFIX  e.g. riscv64-unknown-linux-gnu- (empty: try
#                           riscv64-unknown-linux-gnu-, then riscv64-linux-gnu-)
#   RISCV_SYSROOT           target sysroot (empty: the compiler's -print-sysroot)
# FireSim example: RISCV_TOOLCHAIN_ROOT=$RISCV (the conda riscv-tools prefix).
# Bare-metal (*-elf-) toolchains are rejected: llama.cpp needs a Linux libc.

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

foreach(_rv_var IN ITEMS RISCV_TOOLCHAIN_ROOT RISCV_TOOLCHAIN_PREFIX RISCV_SYSROOT)
    if(NOT DEFINED ${_rv_var} AND DEFINED ENV{${_rv_var}})
        set(${_rv_var} "$ENV{${_rv_var}}")
    endif()
endforeach()
# try_compile projects re-read this file; forward the resolved inputs.
list(APPEND CMAKE_TRY_COMPILE_PLATFORM_VARIABLES RISCV_TOOLCHAIN_ROOT RISCV_TOOLCHAIN_PREFIX RISCV_SYSROOT)

if(RISCV_TOOLCHAIN_PREFIX MATCHES "elf-?$")
    message(FATAL_ERROR "riscv64-linux toolchain: RISCV_TOOLCHAIN_PREFIX='${RISCV_TOOLCHAIN_PREFIX}' is a bare-metal toolchain")
endif()
if(RISCV_TOOLCHAIN_PREFIX)
    set(_rv_prefixes "${RISCV_TOOLCHAIN_PREFIX}")
else()
    set(_rv_prefixes riscv64-unknown-linux-gnu- riscv64-linux-gnu-)
endif()
if(RISCV_TOOLCHAIN_ROOT)
    set(_rv_hints HINTS "${RISCV_TOOLCHAIN_ROOT}/bin" NO_DEFAULT_PATH)
endif()
unset(_rv_cc CACHE)
foreach(_rv_prefix IN LISTS _rv_prefixes)
    find_program(_rv_cc NAMES "${_rv_prefix}gcc" ${_rv_hints})
    if(_rv_cc)
        set(RISCV_TOOLCHAIN_PREFIX "${_rv_prefix}")
        break()
    endif()
endforeach()
if(NOT _rv_cc)
    message(FATAL_ERROR "riscv64-linux toolchain: no ${_rv_prefixes}gcc found "
                        "(set RISCV_TOOLCHAIN_ROOT or RISCV_TOOLCHAIN_PREFIX)")
endif()
get_filename_component(_rv_bin "${_rv_cc}" DIRECTORY)
set(CMAKE_C_COMPILER "${_rv_cc}")
set(CMAKE_CXX_COMPILER "${_rv_bin}/${RISCV_TOOLCHAIN_PREFIX}g++")

execute_process(COMMAND "${CMAKE_C_COMPILER}" -dumpmachine
    OUTPUT_VARIABLE _rv_machine OUTPUT_STRIP_TRAILING_WHITESPACE RESULT_VARIABLE _rv_rc)
if(NOT _rv_rc EQUAL 0 OR NOT _rv_machine MATCHES "^riscv64-.*linux")
    message(FATAL_ERROR "riscv64-linux toolchain: ${CMAKE_C_COMPILER} targets '${_rv_machine}', not riscv64 Linux")
endif()

if(NOT RISCV_SYSROOT)
    execute_process(COMMAND "${CMAKE_C_COMPILER}" -print-sysroot
        OUTPUT_VARIABLE RISCV_SYSROOT OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
endif()
if(RISCV_SYSROOT)
    set(CMAKE_SYSROOT "${RISCV_SYSROOT}")
    set(CMAKE_FIND_ROOT_PATH "${RISCV_SYSROOT}")
endif()
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
