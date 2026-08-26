cmake_minimum_required(VERSION 3.14) # for add_link_options and implicit target directories.
project("llama.cpp" C CXX)
include(CheckIncludeFileCXX)

#set(CMAKE_WARN_DEPRECATED YES)
set(CMAKE_WARN_UNUSED_CLI YES)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT XCODE AND NOT MSVC AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Add path to modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")

# Gemmini include path (needed for gemmini_params.h in ggml-gemmini-args.h)
get_filename_component(GEMMINI_SW_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../RISC-V-DynDNN-gemmini-include" ABSOLUTE)
if (EXISTS "${GEMMINI_SW_PATH}/gemmini.h" OR EXISTS "${GEMMINI_SW_PATH}/include/gemmini.h")
    include_directories(${GEMMINI_SW_PATH} ${GEMMINI_SW_PATH}/include)
else()
    unset(GEMMINI_SW_PATH)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Ensure runtime shared libs are found from the executable directory in the rootfs.
if (UNIX AND NOT APPLE)
    set(CMAKE_BUILD_RPATH_USE_ORIGIN ON)
    set(CMAKE_BUILD_RPATH "$ORIGIN")
    set(CMAKE_INSTALL_RPATH "$ORIGIN")
endif()

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(LLAMA_STANDALONE ON)

    include(git-vars)

    # configure project version
    # TODO
else()
    set(LLAMA_STANDALONE OFF)
endif()

option(LLAMA_USE_SYSTEM_GGML "Use system libggml" OFF)

if (EMSCRIPTEN)
    set(BUILD_SHARED_LIBS_DEFAULT OFF)

    option(LLAMA_WASM_SINGLE_FILE "llama: embed WASM inside the generated llama.js" ON)
else()
    if (MINGW)
        set(BUILD_SHARED_LIBS_DEFAULT OFF)
    else()
        set(BUILD_SHARED_LIBS_DEFAULT ON)
    endif()
endif()

option(BUILD_SHARED_LIBS "build shared libraries" ${BUILD_SHARED_LIBS_DEFAULT})

if (WIN32)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
endif()

if (MSVC)
    add_compile_options("$<$<COMPILE_LANGUAGE:C>:/utf-8>")
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/utf-8>")
    add_compile_options("$<$<COMPILE_LANGUAGE:C>:/bigobj>")
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/bigobj>")
endif()

#
# option list
#

# debug
option(LLAMA_ALL_WARNINGS           "llama: enable all compiler warnings"                   ON)
option(LLAMA_ALL_WARNINGS_3RD_PARTY "llama: enable all compiler warnings in 3rd party libs" OFF)

# build
option(LLAMA_FATAL_WARNINGS "llama: enable -Werror flag" OFF)

# sanitizers
option(LLAMA_SANITIZE_THREAD    "llama: enable thread sanitizer"    OFF)
option(LLAMA_SANITIZE_ADDRESS   "llama: enable address sanitizer"   OFF)
option(LLAMA_SANITIZE_UNDEFINED "llama: enable undefined sanitizer" OFF)

# utils
option(LLAMA_BUILD_COMMON "llama: build common utils library" ${LLAMA_STANDALONE})

# extra artifacts
option(LLAMA_BUILD_TESTS    "llama: build tests"          ${LLAMA_STANDALONE})
option(LLAMA_BUILD_TOOLS    "llama: build tools"          ${LLAMA_STANDALONE})
option(LLAMA_BUILD_EXAMPLES "llama: build examples"       ${LLAMA_STANDALONE})
option(LLAMA_BUILD_SERVER   "llama: build server example" ${LLAMA_STANDALONE})

# 3rd party libs
option(LLAMA_CURL       "llama: use libcurl to download model from an URL" ON)
option(LLAMA_LLGUIDANCE "llama-common: include LLGuidance library for structured output in common utils" OFF)

# Required for relocatable CMake package
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/build-info.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/common.cmake)

# override ggml options
set(GGML_ALL_WARNINGS   ${LLAMA_ALL_WARNINGS})
set(GGML_FATAL_WARNINGS ${LLAMA_FATAL_WARNINGS})

# change the default for these ggml options
if (NOT DEFINED GGML_LLAMAFILE)
    set(GGML_LLAMAFILE_DEFAULT ON)
endif()

if (NOT DEFINED GGML_CUDA_GRAPHS)
    set(GGML_CUDA_GRAPHS_DEFAULT ON)
endif()

# transition helpers
function (llama_option_depr TYPE OLD NEW)
    if (${OLD})
        message(${TYPE} "${OLD} is deprecated and will be removed in the future.\nUse ${NEW} instead\n")
        set(${NEW} ON PARENT_SCOPE)
    endif()
endfunction()

llama_option_depr(FATAL_ERROR LLAMA_CUBLAS              GGML_CUDA)
llama_option_depr(WARNING     LLAMA_CUDA                GGML_CUDA)
llama_option_depr(WARNING     LLAMA_KOMPUTE             GGML_KOMPUTE)
llama_option_depr(WARNING     LLAMA_METAL               GGML_METAL)
llama_option_depr(WARNING     LLAMA_METAL_EMBED_LIBRARY GGML_METAL_EMBED_LIBRARY)
llama_option_depr(WARNING     LLAMA_NATIVE              GGML_NATIVE)
llama_option_depr(WARNING     LLAMA_RPC                 GGML_RPC)
llama_option_depr(WARNING     LLAMA_SYCL                GGML_SYCL)
llama_option_depr(WARNING     LLAMA_SYCL_F16            GGML_SYCL_F16)
llama_option_depr(WARNING     LLAMA_CANN                GGML_CANN)

if (NOT MSVC)
    if (LLAMA_SANITIZE_THREAD)
        message(STATUS "Using -fsanitize=thread")

        add_compile_options(-fsanitize=thread)
        link_libraries     (-fsanitize=thread)
    endif()

    if (LLAMA_SANITIZE_ADDRESS)
        message(STATUS "Using -fsanitize=address")

        add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
        link_libraries     (-fsanitize=address)
    endif()

    if (LLAMA_SANITIZE_UNDEFINED)
        message(STATUS "Using -fsanitize=undefined")

        add_compile_options(-fsanitize=undefined)
        link_libraries     (-fsanitize=undefined)
    endif()
endif()

#
# 3rd-party
#

if (LLAMA_USE_SYSTEM_GGML)
    message(STATUS "Using system-provided libggml, skipping ggml build")
    find_package(ggml REQUIRED)
    add_library(ggml ALIAS ggml::ggml)
endif()

# Gemmini compile-time switches (source of truth: root cache + build scripts).
set(GGML_GEMMINI_PRINT_TILE "0" CACHE STRING "Gemmini PRINT_TILE macro (0|1)")
set_property(CACHE GGML_GEMMINI_PRINT_TILE PROPERTY STRINGS 0 1)
set(GGML_GEMMINI_OPTION "CPU" CACHE STRING "Gemmini OPTION macro (CPU|OS|WS)")
set_property(CACHE GGML_GEMMINI_OPTION PROPERTY STRINGS CPU OS WS)
string(TOUPPER "${GGML_GEMMINI_OPTION}" _GGML_GEMMINI_OPTION_UPPER)
set(_GGML_GEMMINI_OPTION_VALUES CPU OS WS)
list(FIND _GGML_GEMMINI_OPTION_VALUES "${_GGML_GEMMINI_OPTION_UPPER}" _GGML_GEMMINI_OPTION_INDEX)
if(_GGML_GEMMINI_OPTION_INDEX EQUAL -1)
    message(FATAL_ERROR
        "GGML_GEMMINI_OPTION must be one of CPU|OS|WS, got '${GGML_GEMMINI_OPTION}'")
endif()
set(GGML_GEMMINI_OPTION "${_GGML_GEMMINI_OPTION_UPPER}" CACHE STRING
    "Gemmini OPTION macro (CPU|OS|WS)" FORCE)
set_property(CACHE GGML_GEMMINI_OPTION PROPERTY STRINGS CPU OS WS)
set(GGML_GEMMINI_FULL_C "1" CACHE STRING "Gemmini FULL_C macro (0|1)")
set_property(CACHE GGML_GEMMINI_FULL_C PROPERTY STRINGS 0 1)
set(GGML_GEMMINI_LOW_D "0" CACHE STRING "Gemmini LOW_D macro (0|1)")
set_property(CACHE GGML_GEMMINI_LOW_D PROPERTY STRINGS 0 1)
set(GGML_GEMMINI_TRANSPOSE_B "1" CACHE STRING "Gemmini TRANSPOSE_B macro (0|1)")
set_property(CACHE GGML_GEMMINI_TRANSPOSE_B PROPERTY STRINGS 0 1)
set(GGML_GEMMINI_WS_DEBUG "0" CACHE STRING "Gemmini WS debug logs (0|1)")
set_property(CACHE GGML_GEMMINI_WS_DEBUG PROPERTY STRINGS 0 1)
set(GGML_GEMMINI_WS_LOOP_CYCLE "0" CACHE STRING "Gemmini WS loop hardware cycle profiling (0|1)")
set_property(CACHE GGML_GEMMINI_WS_LOOP_CYCLE PROPERTY STRINGS 0 1)
set(GGML_GEMMINI_DISABLE_WS_REUSE "0" CACHE STRING "Disable WS A/B reuse for debugging (0|1)")
set_property(CACHE GGML_GEMMINI_DISABLE_WS_REUSE PROPERTY STRINGS 0 1)

# ---------------------------------------------------------------------------
# Adding a new compile-time option
# ---------------------------------------------------------------------------
# 1. Choose a macro name (must match the C++ header, e.g. GGML_GEMMINI_*).
# 2. Add a set(CACHE ...) + set_property(STRINGS ...) block below.
# 3. Validate the string input and map it to an integer value.
# 4. Append the integer to GGML_GEMMINI_COMPILE_DEFS so it reaches #ifdef.
# 5. In the C++ header add the enum entry, #elif branch, and static_assert.
# 6. In the dispatch site (e.g. quantize.cpp) add a case / if constexpr branch.
# ---------------------------------------------------------------------------
set(GGML_GEMMINI_COMPUTE_TYPE "INT" CACHE STRING "Gemmini compute type: INT or FLOAT")
set_property(CACHE GGML_GEMMINI_COMPUTE_TYPE PROPERTY STRINGS INT FLOAT)
string(TOUPPER "${GGML_GEMMINI_COMPUTE_TYPE}" _GGML_GEMMINI_COMPUTE_TYPE_UPPER)
if(NOT _GGML_GEMMINI_COMPUTE_TYPE_UPPER STREQUAL "INT" AND
   NOT _GGML_GEMMINI_COMPUTE_TYPE_UPPER STREQUAL "FLOAT")
    message(FATAL_ERROR "GGML_GEMMINI_COMPUTE_TYPE must be one of: INT, FLOAT")
endif()
if(_GGML_GEMMINI_COMPUTE_TYPE_UPPER STREQUAL "INT")
    set(_GGML_GEMMINI_COMPUTE_TYPE_VALUE 0)
else()
    set(_GGML_GEMMINI_COMPUTE_TYPE_VALUE 1)
endif()

set(GGML_GEMMINI_DEQUANT_FP_TEST OFF CACHE BOOL "Validate FLOAT with quantized/dequantized activations")
if(GGML_GEMMINI_DEQUANT_FP_TEST AND NOT _GGML_GEMMINI_COMPUTE_TYPE_UPPER STREQUAL "FLOAT")
    message(FATAL_ERROR "GGML_GEMMINI_DEQUANT_FP_TEST requires GGML_GEMMINI_COMPUTE_TYPE=FLOAT")
endif()
if(GGML_GEMMINI_DEQUANT_FP_TEST)
    set(_GGML_GEMMINI_DEQUANT_FP_TEST_VALUE 1)
else()
    set(_GGML_GEMMINI_DEQUANT_FP_TEST_VALUE 0)
endif()

set(GGML_GEMMINI_ACTIVATION_QUANT "EXSIA" CACHE STRING "Gemmini activation quant type")
set_property(CACHE GGML_GEMMINI_ACTIVATION_QUANT PROPERTY STRINGS EXSIA TENSOR TOKEN BLOCK STRIPE)
string(TOUPPER "${GGML_GEMMINI_ACTIVATION_QUANT}" _GGML_GEMMINI_ACTIVATION_QUANT_UPPER)
if(NOT _GGML_GEMMINI_ACTIVATION_QUANT_UPPER STREQUAL "EXSIA" AND
   NOT _GGML_GEMMINI_ACTIVATION_QUANT_UPPER STREQUAL "TENSOR" AND
   NOT _GGML_GEMMINI_ACTIVATION_QUANT_UPPER STREQUAL "TOKEN" AND
   NOT _GGML_GEMMINI_ACTIVATION_QUANT_UPPER STREQUAL "BLOCK" AND
   NOT _GGML_GEMMINI_ACTIVATION_QUANT_UPPER STREQUAL "STRIPE")
    message(FATAL_ERROR "GGML_GEMMINI_ACTIVATION_QUANT must be EXSIA, TENSOR, TOKEN, BLOCK, or STRIPE, got '${GGML_GEMMINI_ACTIVATION_QUANT}'")
endif()
if(_GGML_GEMMINI_ACTIVATION_QUANT_UPPER STREQUAL "EXSIA")
    set(_GGML_GEMMINI_ACTIVATION_QUANT_VALUE 0)
elseif(_GGML_GEMMINI_ACTIVATION_QUANT_UPPER STREQUAL "TENSOR")
    set(_GGML_GEMMINI_ACTIVATION_QUANT_VALUE 1)
elseif(_GGML_GEMMINI_ACTIVATION_QUANT_UPPER STREQUAL "TOKEN")
    set(_GGML_GEMMINI_ACTIVATION_QUANT_VALUE 2)
elseif(_GGML_GEMMINI_ACTIVATION_QUANT_UPPER STREQUAL "BLOCK")
    set(_GGML_GEMMINI_ACTIVATION_QUANT_VALUE 3)
else()
    set(_GGML_GEMMINI_ACTIVATION_QUANT_VALUE 4)
endif()

set(GGML_GEMMINI_BLOCK_SIZE "32" CACHE STRING "Gemmini block size")
set_property(CACHE GGML_GEMMINI_BLOCK_SIZE PROPERTY STRINGS 32 64 128)
if(NOT GGML_GEMMINI_BLOCK_SIZE MATCHES "^(32|64|128)$")
    message(FATAL_ERROR "GGML_GEMMINI_BLOCK_SIZE must be one of: 32, 64, 128; got '${GGML_GEMMINI_BLOCK_SIZE}'")
endif()
set(_GGML_GEMMINI_BLOCK_SIZE_VALUE "${GGML_GEMMINI_BLOCK_SIZE}")

set(GGML_GEMMINI_ACTIVATION_BITS "8" CACHE STRING "Gemmini activation quantization bit width")
set_property(CACHE GGML_GEMMINI_ACTIVATION_BITS PROPERTY STRINGS 4 8 16)
if(NOT GGML_GEMMINI_ACTIVATION_BITS MATCHES "^(4|8|16)$")
    message(FATAL_ERROR "GGML_GEMMINI_ACTIVATION_BITS must be one of: 4, 8, 16; got '${GGML_GEMMINI_ACTIVATION_BITS}'")
endif()
set(_GGML_GEMMINI_ACTIVATION_BITS_VALUE "${GGML_GEMMINI_ACTIVATION_BITS}")

set(GGML_GEMMINI_WEIGHT_BITS "8" CACHE STRING "Gemmini native weight bit width")
set_property(CACHE GGML_GEMMINI_WEIGHT_BITS PROPERTY STRINGS 4 8 16)
if(NOT GGML_GEMMINI_WEIGHT_BITS MATCHES "^(4|8|16)$")
    message(FATAL_ERROR "GGML_GEMMINI_WEIGHT_BITS must be one of: 4, 8, 16; got '${GGML_GEMMINI_WEIGHT_BITS}'")
endif()
set(_GGML_GEMMINI_WEIGHT_BITS_VALUE "${GGML_GEMMINI_WEIGHT_BITS}")

set(GGML_GEMMINI_EXECUTION_BACKEND "HARDWARE" CACHE STRING
    "Gemmini execution backend: HARDWARE or IM2P_SIM")
set_property(CACHE GGML_GEMMINI_EXECUTION_BACKEND PROPERTY STRINGS HARDWARE IM2P_SIM)
if(NOT GGML_GEMMINI_EXECUTION_BACKEND STREQUAL "HARDWARE" AND
   NOT GGML_GEMMINI_EXECUTION_BACKEND STREQUAL "IM2P_SIM")
    message(FATAL_ERROR
        "GGML_GEMMINI_EXECUTION_BACKEND must be exactly HARDWARE or IM2P_SIM, got '${GGML_GEMMINI_EXECUTION_BACKEND}'")
endif()
if(_GGML_GEMMINI_OPTION_UPPER STREQUAL "OS" OR
   (_GGML_GEMMINI_OPTION_UPPER STREQUAL "CPU" AND
    GGML_GEMMINI_EXECUTION_BACKEND STREQUAL "IM2P_SIM"))
    message(FATAL_ERROR
        "Illegal Gemmini option/backend combination: ${_GGML_GEMMINI_OPTION_UPPER}+${GGML_GEMMINI_EXECUTION_BACKEND}; legal combinations are CPU+HARDWARE, WS+HARDWARE, WS+IM2P_SIM")
endif()
if(NOT GGML_GEMMINI_ACTIVATION_BITS STREQUAL
       "${GGML_GEMMINI_WEIGHT_BITS}")
    message(FATAL_ERROR
        "Gemmini requires matched activation and weight widths")
endif()
set(IM2P_SIM_ROOT "" CACHE PATH "Path to the IM2P.sim source and isolated build artifacts")
if(DEFINED IM2P_DIM AND NOT DEFINED GGML_GEMMINI_DIM)
    set(GGML_GEMMINI_DIM "${IM2P_DIM}" CACHE STRING
        "Gemmini array dimension: 16, 32, or 64")
    message(DEPRECATION "IM2P_DIM is deprecated; use GGML_GEMMINI_DIM")
endif()
set(GGML_GEMMINI_DIM "16" CACHE STRING
    "Gemmini array dimension: 16, 32, or 64")
set_property(CACHE GGML_GEMMINI_DIM PROPERTY STRINGS 16 32 64)
if(NOT GGML_GEMMINI_DIM MATCHES "^(16|32|64)$")
    message(FATAL_ERROR
        "GGML_GEMMINI_DIM must be 16, 32, or 64, got '${GGML_GEMMINI_DIM}'")
endif()

set(GGML_GEMMINI_GENERATED_CONFIG_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(MAKE_DIRECTORY "${GGML_GEMMINI_GENERATED_CONFIG_DIR}")
if(GGML_GEMMINI)
    if(NOT GEMMINI_SW_PATH)
        message(FATAL_ERROR "GGML_GEMMINI requires the Gemmini include tree")
    endif()
    set(_GGML_GEMMINI_PARAMS_SOURCE "${GEMMINI_SW_PATH}/gemmini_params.h")
    if(NOT EXISTS "${_GGML_GEMMINI_PARAMS_SOURCE}")
        message(FATAL_ERROR
            "Missing Gemmini parameter header: ${_GGML_GEMMINI_PARAMS_SOURCE}")
    endif()
    file(READ "${_GGML_GEMMINI_PARAMS_SOURCE}" _GGML_GEMMINI_PARAMS_CONTENT)
    string(REGEX MATCH "#define[ \t]+DIM[ \t]+([0-9]+)"
        _GGML_GEMMINI_PHYSICAL_DIM_MATCH "${_GGML_GEMMINI_PARAMS_CONTENT}")
    if(NOT _GGML_GEMMINI_PHYSICAL_DIM_MATCH)
        message(FATAL_ERROR
            "Gemmini parameter header does not define a numeric DIM")
    endif()
    set(_GGML_GEMMINI_PHYSICAL_DIM "${CMAKE_MATCH_1}")
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "riscv")
        if(NOT _GGML_GEMMINI_PHYSICAL_DIM STREQUAL "${GGML_GEMMINI_DIM}")
            message(FATAL_ERROR
                "Physical Gemmini DIM mismatch: hardware header reports ${_GGML_GEMMINI_PHYSICAL_DIM}, build requests ${GGML_GEMMINI_DIM}")
        endif()
    else()
        string(REGEX REPLACE "#define[ \t]+DIM[ \t]+[0-9]+"
            "#define DIM ${GGML_GEMMINI_DIM}" _GGML_GEMMINI_PARAMS_CONTENT
            "${_GGML_GEMMINI_PARAMS_CONTENT}")
        if(GGML_GEMMINI_ACTIVATION_BITS EQUAL 16)
            string(REGEX REPLACE
                "#define[ \t]+MAX_BLOCK_LEN[ \t]+\\(MAX_BYTES/\\(DIM\\*1\\)\\)"
                "#define MAX_BLOCK_LEN (MAX_BYTES/(DIM*2))"
                _GGML_GEMMINI_PARAMS_CONTENT
                "${_GGML_GEMMINI_PARAMS_CONTENT}")
            string(REGEX REPLACE "typedef[ \t]+int8_t[ \t]+elem_t;"
                "typedef int16_t elem_t;"
                _GGML_GEMMINI_PARAMS_CONTENT
                "${_GGML_GEMMINI_PARAMS_CONTENT}")
            string(REGEX REPLACE
                "static const elem_t elem_t_max = 127;"
                "static const elem_t elem_t_max = 32767;"
                _GGML_GEMMINI_PARAMS_CONTENT
                "${_GGML_GEMMINI_PARAMS_CONTENT}")
            string(REGEX REPLACE
                "static const elem_t elem_t_min = -128;"
                "static const elem_t elem_t_min = -32768;"
                _GGML_GEMMINI_PARAMS_CONTENT
                "${_GGML_GEMMINI_PARAMS_CONTENT}")
            string(REPLACE "INT8_MAX" "INT16_MAX"
                _GGML_GEMMINI_PARAMS_CONTENT
                "${_GGML_GEMMINI_PARAMS_CONTENT}")
            string(REPLACE "INT8_MIN" "INT16_MIN"
                _GGML_GEMMINI_PARAMS_CONTENT
                "${_GGML_GEMMINI_PARAMS_CONTENT}")
        endif()
    endif()
    file(WRITE "${GGML_GEMMINI_GENERATED_CONFIG_DIR}/gemmini_params.h"
        "${_GGML_GEMMINI_PARAMS_CONTENT}")
    include_directories(BEFORE "${GGML_GEMMINI_GENERATED_CONFIG_DIR}")
endif()

if(GGML_GEMMINI_EXECUTION_BACKEND STREQUAL "IM2P_SIM")
    if(NOT GGML_GEMMINI)
        message(FATAL_ERROR "GGML_GEMMINI_EXECUTION_BACKEND=IM2P_SIM requires GGML_GEMMINI=ON")
    endif()
    if(CMAKE_CROSSCOMPILING OR CMAKE_SYSTEM_PROCESSOR MATCHES "riscv")
        message(FATAL_ERROR "GGML_GEMMINI_EXECUTION_BACKEND=IM2P_SIM is unavailable for RISC-V/cross-compiling builds")
    endif()
    if(IM2P_SIM_ROOT STREQUAL "")
        message(FATAL_ERROR "IM2P_SIM_ROOT is required for GGML_GEMMINI_EXECUTION_BACKEND=IM2P_SIM")
    endif()
    get_filename_component(IM2P_SIM_ROOT "${IM2P_SIM_ROOT}" ABSOLUTE)
    if(NOT EXISTS "${IM2P_SIM_ROOT}/sim/include/im2p_sim.h" OR
       NOT EXISTS "${IM2P_SIM_ROOT}/frontend/include/im2p_gemmini_frontend.hpp")
        message(FATAL_ERROR
            "IM2P_SIM_ROOT='${IM2P_SIM_ROOT}' does not contain the IM2P simulator and frontend headers")
    endif()

    set(GGML_GEMMINI_IM2P_ARTIFACT_ID
        "a${GGML_GEMMINI_ACTIVATION_BITS}-w${GGML_GEMMINI_WEIGHT_BITS}-d${GGML_GEMMINI_DIM}")
    set(GGML_GEMMINI_IM2P_FRONTEND_ARCHIVE
        "${IM2P_SIM_ROOT}/build/lib/${GGML_GEMMINI_IM2P_ARTIFACT_ID}/libim2p_gemmini_frontend.a")
    set(GGML_GEMMINI_IM2P_SIM_ARCHIVE
        "${IM2P_SIM_ROOT}/build/cargo/${GGML_GEMMINI_IM2P_ARTIFACT_ID}/release/libim2p_sim.a")
    foreach(_GGML_GEMMINI_IM2P_ARCHIVE IN ITEMS
            GGML_GEMMINI_IM2P_FRONTEND_ARCHIVE GGML_GEMMINI_IM2P_SIM_ARCHIVE)
        if(NOT EXISTS "${${_GGML_GEMMINI_IM2P_ARCHIVE}}")
            message(FATAL_ERROR
                "Missing matching IM2P ${GGML_GEMMINI_IM2P_ARTIFACT_ID} archive: ${${_GGML_GEMMINI_IM2P_ARCHIVE}}")
        endif()
    endforeach()

    set(_GGML_GEMMINI_IM2P_PROBE_SOURCE "${CMAKE_CURRENT_BINARY_DIR}/im2p-build-contract-probe.cpp")
    file(WRITE "${_GGML_GEMMINI_IM2P_PROBE_SOURCE}" [=[
#include <cstdint>
#include <cstdio>
extern "C" {
std::uint32_t im2p_sim_abi_version(void);
std::uint32_t im2p_sim_activation_bits(void);
std::uint32_t im2p_sim_activation_storage_bytes(void);
std::uint32_t im2p_sim_weight_bits(void);
std::uint32_t im2p_sim_weight_storage_bytes(void);
std::uint32_t im2p_sim_dim(void);
}
int main() {
    std::printf("%u;%u;%u;%u;%u;%u", im2p_sim_abi_version(),
                im2p_sim_activation_bits(),
                im2p_sim_activation_storage_bytes(),
                im2p_sim_weight_bits(), im2p_sim_weight_storage_bytes(),
                im2p_sim_dim());
    return 0;
}
]=])
    unset(_GGML_GEMMINI_IM2P_PROBE_COMPILED CACHE)
    unset(_GGML_GEMMINI_IM2P_PROBE_RESULT CACHE)
    try_run(_GGML_GEMMINI_IM2P_PROBE_RESULT _GGML_GEMMINI_IM2P_PROBE_COMPILED
        "${CMAKE_CURRENT_BINARY_DIR}/im2p-build-contract-probe"
        "${_GGML_GEMMINI_IM2P_PROBE_SOURCE}"
        LINK_LIBRARIES "${GGML_GEMMINI_IM2P_SIM_ARCHIVE}"
        RUN_OUTPUT_VARIABLE _GGML_GEMMINI_IM2P_IDENTITY)
    if(NOT _GGML_GEMMINI_IM2P_PROBE_COMPILED)
        message(FATAL_ERROR
            "Unable to link the selected IM2P simulator archive for identity validation")
    endif()
    if(NOT _GGML_GEMMINI_IM2P_PROBE_RESULT EQUAL 0 OR
       NOT _GGML_GEMMINI_IM2P_IDENTITY MATCHES "^([0-9]+);([0-9]+);([0-9]+);([0-9]+);([0-9]+);([0-9]+)$")
        message(FATAL_ERROR "Malformed IM2P simulator identity '${_GGML_GEMMINI_IM2P_IDENTITY}'")
    endif()
    set(_GGML_GEMMINI_IM2P_ABI "${CMAKE_MATCH_1}")
    set(_GGML_GEMMINI_IM2P_ACTIVATION_BITS "${CMAKE_MATCH_2}")
    set(_GGML_GEMMINI_IM2P_ACTIVATION_STORAGE_BYTES "${CMAKE_MATCH_3}")
    set(_GGML_GEMMINI_IM2P_WEIGHT_BITS "${CMAKE_MATCH_4}")
    set(_GGML_GEMMINI_IM2P_WEIGHT_STORAGE_BYTES "${CMAKE_MATCH_5}")
    set(_GGML_GEMMINI_IM2P_DIM "${CMAKE_MATCH_6}")
    if(NOT _GGML_GEMMINI_IM2P_ABI STREQUAL "4")
        message(FATAL_ERROR
            "IM2P simulator ABI mismatch: llama requires 4, archive reports ${_GGML_GEMMINI_IM2P_ABI}")
    endif()
    if(NOT _GGML_GEMMINI_IM2P_ACTIVATION_BITS STREQUAL "${GGML_GEMMINI_ACTIVATION_BITS}")
        message(FATAL_ERROR
            "IM2P activation width mismatch: llama requests ${GGML_GEMMINI_ACTIVATION_BITS}, archive reports ${_GGML_GEMMINI_IM2P_ACTIVATION_BITS}")
    endif()
    if(NOT _GGML_GEMMINI_IM2P_WEIGHT_BITS STREQUAL "${GGML_GEMMINI_WEIGHT_BITS}")
        message(FATAL_ERROR
            "IM2P weight width mismatch: llama requests ${GGML_GEMMINI_WEIGHT_BITS}, archive reports ${_GGML_GEMMINI_IM2P_WEIGHT_BITS}")
    endif()
    if(GGML_GEMMINI_ACTIVATION_BITS STREQUAL "16")
        set(_GGML_GEMMINI_EXPECTED_ACTIVATION_STORAGE_BYTES 2)
    else()
        set(_GGML_GEMMINI_EXPECTED_ACTIVATION_STORAGE_BYTES 1)
    endif()
    if(GGML_GEMMINI_WEIGHT_BITS STREQUAL "16")
        set(_GGML_GEMMINI_EXPECTED_WEIGHT_STORAGE_BYTES 2)
    else()
        set(_GGML_GEMMINI_EXPECTED_WEIGHT_STORAGE_BYTES 1)
    endif()
    if(NOT _GGML_GEMMINI_IM2P_ACTIVATION_STORAGE_BYTES STREQUAL
       "${_GGML_GEMMINI_EXPECTED_ACTIVATION_STORAGE_BYTES}")
        message(FATAL_ERROR
            "IM2P activation storage mismatch: llama requires ${_GGML_GEMMINI_EXPECTED_ACTIVATION_STORAGE_BYTES}, archive reports ${_GGML_GEMMINI_IM2P_ACTIVATION_STORAGE_BYTES}")
    endif()
    if(NOT _GGML_GEMMINI_IM2P_WEIGHT_STORAGE_BYTES STREQUAL
       "${_GGML_GEMMINI_EXPECTED_WEIGHT_STORAGE_BYTES}")
        message(FATAL_ERROR
            "IM2P weight storage mismatch: llama requires ${_GGML_GEMMINI_EXPECTED_WEIGHT_STORAGE_BYTES}, archive reports ${_GGML_GEMMINI_IM2P_WEIGHT_STORAGE_BYTES}")
    endif()
    if(NOT _GGML_GEMMINI_IM2P_DIM STREQUAL "${GGML_GEMMINI_DIM}")
        message(FATAL_ERROR
            "IM2P DIM mismatch: llama requests ${GGML_GEMMINI_DIM}, archive reports ${_GGML_GEMMINI_IM2P_DIM}")
    endif()

    # The simulator queries above validate the Rust/RTL archive itself. Exercise
    # the public frontend with that same archive to catch a stale frontend copied
    # into a correctly named pair directory before any project target compiles.
    find_package(Threads REQUIRED)
    set(_GGML_GEMMINI_IM2P_PAIR_PROBE_SOURCE
        "${CMAKE_CURRENT_BINARY_DIR}/im2p-frontend-pair-probe.cpp")
    file(WRITE "${_GGML_GEMMINI_IM2P_PAIR_PROBE_SOURCE}" [=[
#include "im2p_gemmini_frontend.hpp"
#include "ggml-gemmini-args.h"
#include <cstdint>
int main() {
    if (im2p_execute_matmul_extended(nullptr, nullptr, nullptr) !=
        IM2P_INVALID_LAYOUT)
        return 9;
    if (im2p_begin_striped_matmul(nullptr, nullptr, nullptr) != IM2P_ERROR)
        return 10;
    if (im2p_publish_stripe(nullptr, nullptr) != IM2P_INVALID_LAYOUT)
        return 11;
    if (im2p::gemmini::compiled_activation_bits() !=
        @GGML_GEMMINI_ACTIVATION_BITS@)
        return 20;
    if (im2p::gemmini::compiled_weight_bits() !=
        @GGML_GEMMINI_WEIGHT_BITS@)
        return 30;
    if (im2p::gemmini::compiled_dim() != @GGML_GEMMINI_DIM@)
        return 40;
    ggml_gemmini_args_t args{};
    const auto *base = reinterpret_cast<const std::uint8_t *>(&args);
    const auto offset = [base](const auto *member) -> std::uint64_t {
        return static_cast<std::uint64_t>(
            reinterpret_cast<const std::uint8_t *>(member) - base);
    };
    const auto linked = im2p::gemmini::compiled_args_layout_fingerprint();
    return linked.size == sizeof(args) &&
                   linked.native_weight_bytes == offset(&args.native_weight_bytes) &&
                   linked.col_stride_f_out == offset(&args.col_stride_f_out) &&
                   linked.stride_f_out == offset(&args.stride_f_out) &&
                   linked.tile_i == offset(&args.tile_I)
               ? 0
               : 50;
}
]=])
    file(READ "${_GGML_GEMMINI_IM2P_PAIR_PROBE_SOURCE}"
        _GGML_GEMMINI_IM2P_PAIR_PROBE_CONTENT)
    string(CONFIGURE "${_GGML_GEMMINI_IM2P_PAIR_PROBE_CONTENT}"
        _GGML_GEMMINI_IM2P_PAIR_PROBE_CONTENT @ONLY)
    file(WRITE "${_GGML_GEMMINI_IM2P_PAIR_PROBE_SOURCE}"
        "${_GGML_GEMMINI_IM2P_PAIR_PROBE_CONTENT}")
    set(_GGML_GEMMINI_IM2P_PAIR_PROBE_INCLUDES
        "${CMAKE_CURRENT_BINARY_DIR}/generated"
        "${CMAKE_CURRENT_SOURCE_DIR}/ggml/src/ggml-gemmini"
        "${CMAKE_CURRENT_SOURCE_DIR}/ggml/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/ggml/src"
        "${CMAKE_CURRENT_SOURCE_DIR}/ggml/src/ggml-gemmini-utils/include"
        "${GEMMINI_SW_PATH}"
        "${GEMMINI_SW_PATH}/include"
        "${IM2P_SIM_ROOT}/frontend/include"
        "${IM2P_SIM_ROOT}/sim/include")
    unset(_GGML_GEMMINI_IM2P_PAIR_PROBE_COMPILED CACHE)
    unset(_GGML_GEMMINI_IM2P_PAIR_PROBE_RESULT CACHE)
    try_run(_GGML_GEMMINI_IM2P_PAIR_PROBE_RESULT
        _GGML_GEMMINI_IM2P_PAIR_PROBE_COMPILED
        "${CMAKE_CURRENT_BINARY_DIR}/im2p-frontend-pair-probe"
        "${_GGML_GEMMINI_IM2P_PAIR_PROBE_SOURCE}"
        CMAKE_FLAGS
            "-DCMAKE_CXX_STANDARD=20"
            "-DCMAKE_CXX_STANDARD_REQUIRED=ON"
            "-DINCLUDE_DIRECTORIES=${_GGML_GEMMINI_IM2P_PAIR_PROBE_INCLUDES}"
        COMPILE_DEFINITIONS
            "-DGGML_GEMMINI_BLOCK_SIZE=${GGML_GEMMINI_BLOCK_SIZE}"
            "-DGGML_GEMMINI_ACTIVATION_BITS=${GGML_GEMMINI_ACTIVATION_BITS}"
            "-DGGML_GEMMINI_WEIGHT_BITS=${GGML_GEMMINI_WEIGHT_BITS}"
        LINK_LIBRARIES
            "${GGML_GEMMINI_IM2P_FRONTEND_ARCHIVE}"
            "${GGML_GEMMINI_IM2P_SIM_ARCHIVE}"
            Threads::Threads
            ${CMAKE_DL_LIBS}
        COMPILE_OUTPUT_VARIABLE _GGML_GEMMINI_IM2P_PAIR_PROBE_BUILD_OUTPUT)
    if(NOT _GGML_GEMMINI_IM2P_PAIR_PROBE_COMPILED)
        if(_GGML_GEMMINI_IM2P_PAIR_PROBE_BUILD_OUTPUT MATCHES
           "im2p_poll_completed_extended")
            message(FATAL_ERROR
                "Incompatible IM2P pair: simulator archive is missing required symbol im2p_poll_completed_extended")
        endif()
        message(FATAL_ERROR
            "IM2P frontend args layout mismatch or incompatible pair: unable to link the selected frontend/simulator pair")
    endif()
    if(_GGML_GEMMINI_IM2P_PAIR_PROBE_RESULT EQUAL 50)
        message(FATAL_ERROR
            "IM2P frontend args layout mismatch for ${GGML_GEMMINI_IM2P_ARTIFACT_ID}; rebuild the selected frontend archive")
    elseif(NOT _GGML_GEMMINI_IM2P_PAIR_PROBE_RESULT EQUAL 0)
        message(FATAL_ERROR
            "IM2P frontend configuration mismatch for ${GGML_GEMMINI_IM2P_ARTIFACT_ID} (probe result ${_GGML_GEMMINI_IM2P_PAIR_PROBE_RESULT})")
    endif()
    message(STATUS "GGML Gemmini IM2P frontend archive: ${GGML_GEMMINI_IM2P_FRONTEND_ARCHIVE}")
    message(STATUS "GGML Gemmini IM2P simulator archive: ${GGML_GEMMINI_IM2P_SIM_ARCHIVE}")
endif()
set(GGML_GEMMINI_EXSIA_SIGMA "2" CACHE STRING "EXSIA outlier sigma threshold")
if(NOT "${GGML_GEMMINI_EXSIA_SIGMA}" MATCHES "^[1-9][0-9]*$")
    message(FATAL_ERROR "GGML_GEMMINI_EXSIA_SIGMA must be a positive integer, got '${GGML_GEMMINI_EXSIA_SIGMA}'")
endif()
string(LENGTH "${GGML_GEMMINI_EXSIA_SIGMA}" _GGML_GEMMINI_EXSIA_SIGMA_LENGTH)
if(_GGML_GEMMINI_EXSIA_SIGMA_LENGTH GREATER 10 OR
   (_GGML_GEMMINI_EXSIA_SIGMA_LENGTH EQUAL 10 AND
    "${GGML_GEMMINI_EXSIA_SIGMA}" STRGREATER "2147483647"))
    message(FATAL_ERROR
        "GGML_GEMMINI_EXSIA_SIGMA must be a positive integer no greater than 2147483647, got '${GGML_GEMMINI_EXSIA_SIGMA}'")
endif()

foreach(_GGML_GEMMINI_BOOL_VAR IN ITEMS
        GGML_GEMMINI_PRINT_TILE
        GGML_GEMMINI_FULL_C
        GGML_GEMMINI_LOW_D
        GGML_GEMMINI_TRANSPOSE_B
        GGML_GEMMINI_WS_DEBUG
        GGML_GEMMINI_WS_LOOP_CYCLE
        GGML_GEMMINI_DISABLE_WS_REUSE)
    if (NOT ${_GGML_GEMMINI_BOOL_VAR} MATCHES "^[01]$")
        message(FATAL_ERROR
            "${_GGML_GEMMINI_BOOL_VAR} must be 0 or 1, got '${${_GGML_GEMMINI_BOOL_VAR}}'")
    endif()
endforeach()

set(CYCLE_DETAIL "0" CACHE STRING "Enable EXSIA cycle-detail aggregate logs (0|1)")
set_property(CACHE CYCLE_DETAIL PROPERTY STRINGS 0 1)
if (NOT CYCLE_DETAIL MATCHES "^[01]$")
    message(FATAL_ERROR "CYCLE_DETAIL must be 0 or 1, got '${CYCLE_DETAIL}'")
endif()

set(GGML_GEMMINI_EXSIA_PROFILE_SCOPE "OFF" CACHE STRING "ExSIA profile scope (OFF, TIMELINE, STAGE)")
set_property(CACHE GGML_GEMMINI_EXSIA_PROFILE_SCOPE PROPERTY STRINGS OFF TIMELINE STAGE)
string(TOUPPER "${GGML_GEMMINI_EXSIA_PROFILE_SCOPE}" _GGML_GEMMINI_EXSIA_PROFILE_SCOPE_UPPER)
if(NOT _GGML_GEMMINI_EXSIA_PROFILE_SCOPE_UPPER STREQUAL "OFF" AND
   NOT _GGML_GEMMINI_EXSIA_PROFILE_SCOPE_UPPER STREQUAL "TIMELINE" AND
   NOT _GGML_GEMMINI_EXSIA_PROFILE_SCOPE_UPPER STREQUAL "STAGE")
    message(FATAL_ERROR "GGML_GEMMINI_EXSIA_PROFILE_SCOPE must be OFF, TIMELINE, or STAGE, got '${GGML_GEMMINI_EXSIA_PROFILE_SCOPE}'")
endif()
if(_GGML_GEMMINI_EXSIA_PROFILE_SCOPE_UPPER STREQUAL "OFF")
    set(GGML_GEMMINI_EXSIA_PROFILE_SCOPE_VALUE 0)
elseif(_GGML_GEMMINI_EXSIA_PROFILE_SCOPE_UPPER STREQUAL "TIMELINE")
    set(GGML_GEMMINI_EXSIA_PROFILE_SCOPE_VALUE 1)
else()
    set(GGML_GEMMINI_EXSIA_PROFILE_SCOPE_VALUE 2)
endif()

option(GGML_GEMMINI_ENABLE_OPENMP "Enable OpenMP for Gemmini CPU-side kernels" OFF)
# Balanced Radix-256 Residual Matrix Decomposition. OFF is a residual-compensation
# ablation only; it does not select any other compensation path.
option(GGML_GEMMINI_ENABLE_RMD "Enable Balanced Radix-256 residual compensation (RMD)" ON)
set(GGML_GEMMINI_DEFAULT_RMD_BACKEND "CPU" CACHE STRING
    "Default RMD backend: CPU or WS")
set_property(CACHE GGML_GEMMINI_DEFAULT_RMD_BACKEND PROPERTY STRINGS CPU WS)
if (NOT GGML_GEMMINI_DEFAULT_RMD_BACKEND STREQUAL "CPU" AND
    NOT GGML_GEMMINI_DEFAULT_RMD_BACKEND STREQUAL "WS")
    message(FATAL_ERROR
        "GGML_GEMMINI_DEFAULT_RMD_BACKEND must be exactly CPU or WS, got '${GGML_GEMMINI_DEFAULT_RMD_BACKEND}'")
endif()
if (GGML_GEMMINI_DEFAULT_RMD_BACKEND STREQUAL "CPU")
    set(_GGML_GEMMINI_DEFAULT_RMD_BACKEND_VALUE 0)
else()
    set(_GGML_GEMMINI_DEFAULT_RMD_BACKEND_VALUE 1)
endif()
# Public matmul invocation defaults: FULL / STRIPE_PIPELINE.
option(GGML_GEMMINI_ENABLE_STRIPE_MATMUL "Enable strict stripe matmul modes" ON)
option(GGML_GEMMINI_ENABLE_STRIPE_PIPELINE "Enable strict stripe pipeline mode" ON)
set(GGML_GEMMINI_DEFAULT_MATMUL_MODE "FULL" CACHE STRING
    "Default Gemmini matmul mode: FULL or STRIPE_PIPELINE")
set_property(CACHE GGML_GEMMINI_DEFAULT_MATMUL_MODE PROPERTY STRINGS
    FULL STRIPE_PIPELINE)
set(GGML_GEMMINI_DEFAULT_STRIPE_JOB_CAPACITY "2" CACHE STRING
    "Default stripe job capacity")
option(GGML_GEMMINI_ALLOW_RUNTIME_MATMUL_OVERRIDE
       "Allow GEMMINI_* environment overrides for matmul options" ON)

function(ggml_gemmini_require_bool _var)
    string(TOUPPER "${${_var}}" _GGML_GEMMINI_BOOL_VALUE)
    if (NOT _GGML_GEMMINI_BOOL_VALUE STREQUAL "ON" AND
        NOT _GGML_GEMMINI_BOOL_VALUE STREQUAL "OFF" AND
        NOT _GGML_GEMMINI_BOOL_VALUE STREQUAL "TRUE" AND
        NOT _GGML_GEMMINI_BOOL_VALUE STREQUAL "FALSE" AND
        NOT _GGML_GEMMINI_BOOL_VALUE STREQUAL "YES" AND
        NOT _GGML_GEMMINI_BOOL_VALUE STREQUAL "NO" AND
        NOT _GGML_GEMMINI_BOOL_VALUE STREQUAL "Y" AND
        NOT _GGML_GEMMINI_BOOL_VALUE STREQUAL "N" AND
        NOT _GGML_GEMMINI_BOOL_VALUE STREQUAL "1" AND
        NOT _GGML_GEMMINI_BOOL_VALUE STREQUAL "0")
        message(FATAL_ERROR "${_var} must be a boolean value, got '${${_var}}'")
    endif()
endfunction()

foreach(_GGML_GEMMINI_MATMUL_BOOL IN ITEMS
        GGML_GEMMINI_ENABLE_RMD
        GGML_GEMMINI_ENABLE_STRIPE_MATMUL
        GGML_GEMMINI_ENABLE_STRIPE_PIPELINE
        GGML_GEMMINI_ALLOW_RUNTIME_MATMUL_OVERRIDE)
    ggml_gemmini_require_bool(${_GGML_GEMMINI_MATMUL_BOOL})
endforeach()

if (GGML_GEMMINI_ENABLE_RMD)
    set(_GGML_GEMMINI_ENABLE_RMD_VALUE 1)
else()
    set(_GGML_GEMMINI_ENABLE_RMD_VALUE 0)
endif()

set(_GGML_GEMMINI_MATMUL_MODES FULL STRIPE_PIPELINE)
list(FIND _GGML_GEMMINI_MATMUL_MODES "${GGML_GEMMINI_DEFAULT_MATMUL_MODE}"
     _GGML_GEMMINI_DEFAULT_MATMUL_MODE_VALUE)
if (_GGML_GEMMINI_DEFAULT_MATMUL_MODE_VALUE EQUAL -1)
    message(FATAL_ERROR
        "GGML_GEMMINI_DEFAULT_MATMUL_MODE must be FULL or STRIPE_PIPELINE, got '${GGML_GEMMINI_DEFAULT_MATMUL_MODE}'")
endif()
if (GGML_GEMMINI_ENABLE_STRIPE_PIPELINE AND NOT GGML_GEMMINI_ENABLE_STRIPE_MATMUL)
    message(FATAL_ERROR "GGML_GEMMINI_ENABLE_STRIPE_PIPELINE requires GGML_GEMMINI_ENABLE_STRIPE_MATMUL=ON")
endif()
if (NOT GGML_GEMMINI_DEFAULT_MATMUL_MODE STREQUAL "FULL" AND
    NOT GGML_GEMMINI_ENABLE_STRIPE_MATMUL)
    message(FATAL_ERROR "Stripe matmul defaults require GGML_GEMMINI_ENABLE_STRIPE_MATMUL=ON")
endif()
if (GGML_GEMMINI_DEFAULT_MATMUL_MODE STREQUAL "STRIPE_PIPELINE" AND
    NOT GGML_GEMMINI_ENABLE_STRIPE_PIPELINE)
    message(FATAL_ERROR "STRIPE_PIPELINE default requires GGML_GEMMINI_ENABLE_STRIPE_PIPELINE=ON")
endif()
foreach(_GGML_GEMMINI_MATMUL_COUNT IN ITEMS
        GGML_GEMMINI_DEFAULT_STRIPE_JOB_CAPACITY)
    if (NOT ${_GGML_GEMMINI_MATMUL_COUNT} MATCHES "^[1-9][0-9]*$")
        message(FATAL_ERROR "${_GGML_GEMMINI_MATMUL_COUNT} must be a positive integer")
    endif()
    string(LENGTH "${${_GGML_GEMMINI_MATMUL_COUNT}}" _GGML_GEMMINI_MATMUL_COUNT_LENGTH)
    if (_GGML_GEMMINI_MATMUL_COUNT_LENGTH GREATER 10 OR
        (_GGML_GEMMINI_MATMUL_COUNT_LENGTH EQUAL 10 AND
         "${${_GGML_GEMMINI_MATMUL_COUNT}}" STRGREATER "4294967295"))
        message(FATAL_ERROR "${_GGML_GEMMINI_MATMUL_COUNT} exceeds uint32 range")
    endif()
endforeach()

foreach(_GGML_GEMMINI_MATMUL_BOOL IN ITEMS
        GGML_GEMMINI_ENABLE_STRIPE_MATMUL
        GGML_GEMMINI_ENABLE_STRIPE_PIPELINE
        GGML_GEMMINI_ALLOW_RUNTIME_MATMUL_OVERRIDE)
    if (${_GGML_GEMMINI_MATMUL_BOOL})
        set(_${_GGML_GEMMINI_MATMUL_BOOL}_VALUE 1)
    else()
        set(_${_GGML_GEMMINI_MATMUL_BOOL}_VALUE 0)
    endif()
endforeach()
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/ggml/src/ggml-gemmini/ggml-gemmini-matmul-config.hpp.in"
    "${GGML_GEMMINI_GENERATED_CONFIG_DIR}/ggml-gemmini-matmul-config.hpp"
    @ONLY)

# ExSIA default mode configures internal activation staging, not the public matmul mode.
set(GGML_GEMMINI_EXSIA_DEFAULT_MODE "SEQUENTIAL" CACHE STRING "ExSIA default mode")
set_property(CACHE GGML_GEMMINI_EXSIA_DEFAULT_MODE PROPERTY STRINGS SEQUENTIAL LOCAL_PARALLEL LOCAL_FOLDING_PIPELINE)
string(TOUPPER "${GGML_GEMMINI_EXSIA_DEFAULT_MODE}" _GGML_GEMMINI_EXSIA_DEFAULT_MODE_UPPER)
set(_GGML_GEMMINI_EXSIA_DEFAULT_MODES SEQUENTIAL LOCAL_PARALLEL LOCAL_FOLDING_PIPELINE)
list(FIND _GGML_GEMMINI_EXSIA_DEFAULT_MODES "${_GGML_GEMMINI_EXSIA_DEFAULT_MODE_UPPER}" _GGML_GEMMINI_EXSIA_DEFAULT_MODE_VALUE)
if (_GGML_GEMMINI_EXSIA_DEFAULT_MODE_VALUE EQUAL -1)
    message(FATAL_ERROR "GGML_GEMMINI_EXSIA_DEFAULT_MODE must be SEQUENTIAL, LOCAL_PARALLEL, or LOCAL_FOLDING_PIPELINE")
endif()

set(GGML_GEMMINI_EXSIA_LOCAL_WORKERS "4" CACHE STRING "ExSIA Local worker count")
if (NOT GGML_GEMMINI_EXSIA_LOCAL_WORKERS MATCHES "^[1-9][0-9]*$")
    message(FATAL_ERROR "GGML_GEMMINI_EXSIA_LOCAL_WORKERS must be a positive integer")
endif()
if (NOT (GGML_GEMMINI_EXSIA_LOCAL_WORKERS EQUAL 3 OR GGML_GEMMINI_EXSIA_LOCAL_WORKERS EQUAL 4))
    message(FATAL_ERROR "GGML_GEMMINI_EXSIA_LOCAL_WORKERS must be 3 or 4")
endif()
if (_GGML_GEMMINI_EXSIA_DEFAULT_MODE_VALUE GREATER 0 AND NOT GGML_GEMMINI_ENABLE_OPENMP)
    message(FATAL_ERROR "Parallel ExSIA default modes require GGML_GEMMINI_ENABLE_OPENMP=ON")
endif()
math(EXPR _GGML_GEMMINI_EXSIA_EXPECTED_OMP_THREADS
    "${GGML_GEMMINI_EXSIA_LOCAL_WORKERS} + 1")

message(STATUS "GGML Gemmini matmul defaults:"
    "\n  array dimension: ${GGML_GEMMINI_DIM}"
    "\n  stripe capability: ${GGML_GEMMINI_ENABLE_STRIPE_MATMUL}"
    "\n  stripe pipeline: ${GGML_GEMMINI_ENABLE_STRIPE_PIPELINE}"
    "\n  default mode: ${GGML_GEMMINI_DEFAULT_MATMUL_MODE}"
    "\n  residual compensation: RMD=${GGML_GEMMINI_ENABLE_RMD}"
    "\n  default RMD backend: ${GGML_GEMMINI_DEFAULT_RMD_BACKEND}"
    "\n  block size: ${GGML_GEMMINI_BLOCK_SIZE}"
    "\n  default stripe job capacity: ${GGML_GEMMINI_DEFAULT_STRIPE_JOB_CAPACITY}"
    "\n  runtime override: ${GGML_GEMMINI_ALLOW_RUNTIME_MATMUL_OVERRIDE}"
    "\n  ExSIA internal mode: ${_GGML_GEMMINI_EXSIA_DEFAULT_MODE_UPPER}"
    "\n  ExSIA workers/team: ${GGML_GEMMINI_EXSIA_LOCAL_WORKERS}/${_GGML_GEMMINI_EXSIA_EXPECTED_OMP_THREADS}")

set(GGML_GEMMINI_COMPILE_DEFS
    PRINT_TILE=${GGML_GEMMINI_PRINT_TILE}
    OPTION=${_GGML_GEMMINI_OPTION_UPPER}
    FULL_C=${GGML_GEMMINI_FULL_C}
    LOW_D=${GGML_GEMMINI_LOW_D}
    TRANSPOSE_B=${GGML_GEMMINI_TRANSPOSE_B}
    GEMMINI_WS_DEBUG=${GGML_GEMMINI_WS_DEBUG}
    GEMMINI_WS_LOOP_CYCLE=${GGML_GEMMINI_WS_LOOP_CYCLE}
    GEMMINI_DISABLE_WS_REUSE=${GGML_GEMMINI_DISABLE_WS_REUSE}
    GGML_GEMMINI_COMPUTE_TYPE=${_GGML_GEMMINI_COMPUTE_TYPE_VALUE}
    GGML_GEMMINI_DEQUANT_FP_TEST=${_GGML_GEMMINI_DEQUANT_FP_TEST_VALUE}
    GGML_GEMMINI_ACTIVATION_QUANT=${_GGML_GEMMINI_ACTIVATION_QUANT_VALUE}
    GGML_GEMMINI_BLOCK_SIZE=${_GGML_GEMMINI_BLOCK_SIZE_VALUE}
    GGML_GEMMINI_EXSIA_SIGMA=${GGML_GEMMINI_EXSIA_SIGMA}
    GGML_GEMMINI_ACTIVATION_BITS=${_GGML_GEMMINI_ACTIVATION_BITS_VALUE}
    GGML_GEMMINI_WEIGHT_BITS=${_GGML_GEMMINI_WEIGHT_BITS_VALUE}
    GGML_GEMMINI_CONFIGURED_DIM=${GGML_GEMMINI_DIM}
)
if(GGML_GEMMINI_EXECUTION_BACKEND STREQUAL "IM2P_SIM")
    list(APPEND GGML_GEMMINI_COMPILE_DEFS GGML_GEMMINI_EXECUTION_BACKEND_IM2P_SIM=1)
else()
    list(APPEND GGML_GEMMINI_COMPILE_DEFS GGML_GEMMINI_EXECUTION_BACKEND_HARDWARE=1)
endif()

# set log option
set(LOG_DEBUG "1" CACHE STRING "Enable debug logs (0|1)")
set_property(CACHE LOG_DEBUG PROPERTY STRINGS 0 1)
set(LOG_CYCLE "0" CACHE STRING "Enable cycle logs (0|1)")
set_property(CACHE LOG_CYCLE PROPERTY STRINGS 0 1)
set(LOG_DUMP "0" CACHE STRING "Enable tensor dump logs (0|1)")
set_property(CACHE LOG_DUMP PROPERTY STRINGS 0 1)
set(LOG_DUMP_SCALE "0" CACHE STRING "Enable scale dump logs (0|1)")
set_property(CACHE LOG_DUMP_SCALE PROPERTY STRINGS 0 1)
foreach(_GGML_GEMMINI_LOG_BOOL IN ITEMS LOG_DEBUG LOG_CYCLE LOG_DUMP LOG_DUMP_SCALE)
    if (NOT ${_GGML_GEMMINI_LOG_BOOL} MATCHES "^[01]$")
        message(FATAL_ERROR
            "${_GGML_GEMMINI_LOG_BOOL} must be 0 or 1, got '${${_GGML_GEMMINI_LOG_BOOL}}'")
    endif()
endforeach()

if (CYCLE_DETAIL AND NOT LOG_CYCLE)
    message(FATAL_ERROR "CYCLE_DETAIL=1 requires LOG_CYCLE=1")
endif()
if (NOT _GGML_GEMMINI_EXSIA_PROFILE_SCOPE_UPPER STREQUAL "OFF" AND NOT CYCLE_DETAIL)
    message(FATAL_ERROR "GGML_GEMMINI_EXSIA_PROFILE_SCOPE requires CYCLE_DETAIL=1")
endif()

set(GGML_GEMMINI_RUNTIME_COMPILE_DEFS
    LOG_DEBUG=${LOG_DEBUG}
    LOG_CYCLE=${LOG_CYCLE}
    LOG_DUMP=${LOG_DUMP}
    LOG_DUMP_SCALE=${LOG_DUMP_SCALE}
    GEMMINI_VERIFY_DEQ=1
    CYCLE_DETAIL=${CYCLE_DETAIL}
    GGML_GEMMINI_COMPUTE_TYPE=${_GGML_GEMMINI_COMPUTE_TYPE_VALUE}
    GGML_GEMMINI_DEQUANT_FP_TEST=${_GGML_GEMMINI_DEQUANT_FP_TEST_VALUE}
    GGML_GEMMINI_ACTIVATION_QUANT=${_GGML_GEMMINI_ACTIVATION_QUANT_VALUE}
    GGML_GEMMINI_EXSIA_DEFAULT_MODE_VALUE=${_GGML_GEMMINI_EXSIA_DEFAULT_MODE_VALUE}
    GGML_GEMMINI_EXSIA_LOCAL_WORKERS=${GGML_GEMMINI_EXSIA_LOCAL_WORKERS}
    GGML_GEMMINI_ENABLE_RMD=${_GGML_GEMMINI_ENABLE_RMD_VALUE}
    GGML_GEMMINI_ACTIVATION_BITS=${_GGML_GEMMINI_ACTIVATION_BITS_VALUE}
    GGML_GEMMINI_WEIGHT_BITS=${_GGML_GEMMINI_WEIGHT_BITS_VALUE}
)

list(APPEND GGML_GEMMINI_COMPILE_DEFS ${GGML_GEMMINI_RUNTIME_COMPILE_DEFS})
if (LLAMA_BUILD_TESTS AND GGML_GEMMINI)
    list(APPEND GGML_GEMMINI_COMPILE_DEFS GGML_GEMMINI_TEST_OBSERVER=1)
endif()
set(GGML_GEMMINI_UTILS_COMPILE_DEFS
    LOG_DEBUG=${LOG_DEBUG}
    LOG_CYCLE=${LOG_CYCLE}
    CYCLE_DETAIL=${CYCLE_DETAIL})

# ---- ggml build ----
if (NOT TARGET ggml AND NOT LLAMA_USE_SYSTEM_GGML)
    add_subdirectory(ggml)
    # ... otherwise assume ggml is added by a parent CMakeLists.txt
endif()

#
# build the library
#

add_subdirectory(src)

#
# utils, programs, examples and tests
#

if (NOT LLAMA_BUILD_COMMON)
    message(STATUS "LLAMA_BUILD_COMMON is OFF, disabling LLAMA_CURL")
    set(LLAMA_CURL OFF)
endif()

if (LLAMA_BUILD_COMMON)
    add_subdirectory(common)
endif()

if (LLAMA_BUILD_COMMON AND LLAMA_BUILD_TESTS AND NOT CMAKE_JS_VERSION)
    include(CTest)
    add_subdirectory(tests)
endif()

if (LLAMA_BUILD_COMMON AND LLAMA_BUILD_EXAMPLES)
    add_subdirectory(examples)
    add_subdirectory(pocs)
endif()

if (LLAMA_BUILD_COMMON AND LLAMA_BUILD_TOOLS)
    add_subdirectory(tools)
endif()

if (TARGET test-quantize-atomic-output)
    add_dependencies(test-quantize-atomic-output llama-gguf-split)
endif()

#
# install
#

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(LLAMA_BUILD_NUMBER        ${BUILD_NUMBER})
set(LLAMA_BUILD_COMMIT        ${BUILD_COMMIT})
set(LLAMA_INSTALL_VERSION 0.0.${BUILD_NUMBER})

set(LLAMA_INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_INCLUDEDIR} CACHE PATH "Location of header  files")
set(LLAMA_LIB_INSTALL_DIR     ${CMAKE_INSTALL_LIBDIR}     CACHE PATH "Location of library files")
set(LLAMA_BIN_INSTALL_DIR     ${CMAKE_INSTALL_BINDIR}     CACHE PATH "Location of binary  files")

set(LLAMA_PUBLIC_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/include/llama.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/llama-cpp.h)

set_target_properties(llama
    PROPERTIES
        PUBLIC_HEADER "${LLAMA_PUBLIC_HEADERS}")

install(TARGETS llama LIBRARY PUBLIC_HEADER)

configure_package_config_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/llama-config.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/llama-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/llama
    PATH_VARS LLAMA_INCLUDE_INSTALL_DIR
              LLAMA_LIB_INSTALL_DIR
              LLAMA_BIN_INSTALL_DIR )

write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/llama-version.cmake
    VERSION ${LLAMA_INSTALL_VERSION}
    COMPATIBILITY SameMajorVersion)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/llama-config.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/llama-version.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/llama)

install(
    FILES convert_hf_to_gguf.py
    PERMISSIONS
        OWNER_READ
        OWNER_WRITE
        OWNER_EXECUTE
        GROUP_READ
        GROUP_EXECUTE
        WORLD_READ
        WORLD_EXECUTE
    DESTINATION ${CMAKE_INSTALL_BINDIR})

configure_file(cmake/llama.pc.in
        "${CMAKE_CURRENT_BINARY_DIR}/llama.pc"
        @ONLY)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/llama.pc"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)
