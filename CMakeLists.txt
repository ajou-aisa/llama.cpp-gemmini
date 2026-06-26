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
set(GGML_GEMMINI_FULL_C "1" CACHE STRING "Gemmini FULL_C macro (0|1)")
set_property(CACHE GGML_GEMMINI_FULL_C PROPERTY STRINGS 0 1)
set(GGML_GEMMINI_LOW_D "0" CACHE STRING "Gemmini LOW_D macro (0|1)")
set_property(CACHE GGML_GEMMINI_LOW_D PROPERTY STRINGS 0 1)
set(GGML_GEMMINI_TRANSPOSE_B "1" CACHE STRING "Gemmini TRANSPOSE_B macro (0|1)")
set_property(CACHE GGML_GEMMINI_TRANSPOSE_B PROPERTY STRINGS 0 1)
set(GGML_GEMMINI_WS_DEBUG "0" CACHE STRING "Gemmini WS debug logs (0|1)")
set_property(CACHE GGML_GEMMINI_WS_DEBUG PROPERTY STRINGS 0 1)
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
set(GGML_GEMMINI_COMPUTE_TYPE "INT" CACHE STRING "Gemmini compute type: INT, FLOAT, or DEQUANT_FP_TEST")
set_property(CACHE GGML_GEMMINI_COMPUTE_TYPE PROPERTY STRINGS INT FLOAT DEQUANT_FP_TEST)
string(TOUPPER "${GGML_GEMMINI_COMPUTE_TYPE}" _GGML_GEMMINI_COMPUTE_TYPE_UPPER)
if(NOT _GGML_GEMMINI_COMPUTE_TYPE_UPPER STREQUAL "INT" AND
   NOT _GGML_GEMMINI_COMPUTE_TYPE_UPPER STREQUAL "FLOAT" AND
   NOT _GGML_GEMMINI_COMPUTE_TYPE_UPPER STREQUAL "DEQUANT_FP_TEST")
    message(FATAL_ERROR "GGML_GEMMINI_COMPUTE_TYPE must be one of: INT, FLOAT, DEQUANT_FP_TEST")
endif()
if(_GGML_GEMMINI_COMPUTE_TYPE_UPPER STREQUAL "INT")
    set(_GGML_GEMMINI_COMPUTE_TYPE_VALUE 0)
elseif(_GGML_GEMMINI_COMPUTE_TYPE_UPPER STREQUAL "FLOAT")
    set(_GGML_GEMMINI_COMPUTE_TYPE_VALUE 1)
else()
    set(_GGML_GEMMINI_COMPUTE_TYPE_VALUE 2)
endif()

set(GGML_GEMMINI_ACTIVATION_QUANT "EXSIA" CACHE STRING "Gemmini activation quant type")
set_property(CACHE GGML_GEMMINI_ACTIVATION_QUANT PROPERTY STRINGS EXSIA TENSOR)
string(TOUPPER "${GGML_GEMMINI_ACTIVATION_QUANT}" _GGML_GEMMINI_ACTIVATION_QUANT_UPPER)
if(NOT _GGML_GEMMINI_ACTIVATION_QUANT_UPPER STREQUAL "EXSIA" AND NOT _GGML_GEMMINI_ACTIVATION_QUANT_UPPER STREQUAL "TENSOR")
    message(FATAL_ERROR "GGML_GEMMINI_ACTIVATION_QUANT must be EXSIA or TENSOR, got '${GGML_GEMMINI_ACTIVATION_QUANT}'")
endif()
if(_GGML_GEMMINI_ACTIVATION_QUANT_UPPER STREQUAL "EXSIA")
    set(_GGML_GEMMINI_ACTIVATION_QUANT_VALUE 0)
else()
    set(_GGML_GEMMINI_ACTIVATION_QUANT_VALUE 1)
endif()

set(GGML_GEMMINI_BLOCK_SIZE "32" CACHE STRING "Gemmini block size")
set(_GGML_GEMMINI_BLOCK_SIZE_VALUE "${GGML_GEMMINI_BLOCK_SIZE}")

foreach(_GGML_GEMMINI_BOOL_VAR IN ITEMS
        GGML_GEMMINI_PRINT_TILE
        GGML_GEMMINI_FULL_C
        GGML_GEMMINI_LOW_D
        GGML_GEMMINI_TRANSPOSE_B
        GGML_GEMMINI_WS_DEBUG
        GGML_GEMMINI_DISABLE_WS_REUSE)
    if (NOT ${_GGML_GEMMINI_BOOL_VAR} MATCHES "^[01]$")
        message(FATAL_ERROR
            "${_GGML_GEMMINI_BOOL_VAR} must be 0 or 1, got '${${_GGML_GEMMINI_BOOL_VAR}}'")
    endif()
endforeach()

string(TOUPPER "${GGML_GEMMINI_OPTION}" _GGML_GEMMINI_OPTION_UPPER)
set(_GGML_GEMMINI_OPTION_VALUES CPU OS WS)
list(FIND _GGML_GEMMINI_OPTION_VALUES "${_GGML_GEMMINI_OPTION_UPPER}" _GGML_GEMMINI_OPTION_INDEX)
if (_GGML_GEMMINI_OPTION_INDEX EQUAL -1)
    message(FATAL_ERROR "GGML_GEMMINI_OPTION must be one of CPU|OS|WS, got '${GGML_GEMMINI_OPTION}'")
endif()

set(GGML_GEMMINI_COMPILE_DEFS
    PRINT_TILE=${GGML_GEMMINI_PRINT_TILE}
    OPTION=${_GGML_GEMMINI_OPTION_UPPER}
    FULL_C=${GGML_GEMMINI_FULL_C}
    LOW_D=${GGML_GEMMINI_LOW_D}
    TRANSPOSE_B=${GGML_GEMMINI_TRANSPOSE_B}
    GEMMINI_WS_DEBUG=${GGML_GEMMINI_WS_DEBUG}
    GEMMINI_DISABLE_WS_REUSE=${GGML_GEMMINI_DISABLE_WS_REUSE}
    GGML_GEMMINI_COMPUTE_TYPE=${_GGML_GEMMINI_COMPUTE_TYPE_VALUE}
    GGML_GEMMINI_ACTIVATION_QUANT=${_GGML_GEMMINI_ACTIVATION_QUANT_VALUE}
    GGML_GEMMINI_BLOCK_SIZE=${_GGML_GEMMINI_BLOCK_SIZE_VALUE}
)

# set log option
set(LOG_DEBUG "1" CACHE STRING "Enable debug logs (0|1)")
set_property(CACHE LOG_DEBUG PROPERTY STRINGS 0 1)
set(LOG_CYCLE "1" CACHE STRING "Enable cycle logs (0|1)")
set_property(CACHE LOG_CYCLE PROPERTY STRINGS 0 1)
set(LOG_DUMP "0" CACHE STRING "Enable tensor dump logs (0|1)")
set_property(CACHE LOG_DUMP PROPERTY STRINGS 0 1)
set(LOG_DUMP_SCALE "0" CACHE STRING "Enable scale dump logs (0|1)")
set_property(CACHE LOG_DUMP_SCALE PROPERTY STRINGS 0 1)
set(ERROR_COMPENSATION "0" CACHE STRING "Enable activation error compensation (0|1)")
set_property(CACHE ERROR_COMPENSATION PROPERTY STRINGS 0 1)

set(GGML_GEMMINI_RUNTIME_COMPILE_DEFS
    LOG_DEBUG=${LOG_DEBUG}
    LOG_CYCLE=${LOG_CYCLE}
    LOG_DUMP=${LOG_DUMP}
    LOG_DUMP_SCALE=${LOG_DUMP_SCALE}
    ERROR_COMPENSATION=${ERROR_COMPENSATION}
    GEMMINI_VERIFY_DEQ=1
    GGML_GEMMINI_COMPUTE_TYPE=${_GGML_GEMMINI_COMPUTE_TYPE_VALUE}
    GGML_GEMMINI_ACTIVATION_QUANT=${_GGML_GEMMINI_ACTIVATION_QUANT_VALUE}
)

list(APPEND GGML_GEMMINI_COMPILE_DEFS ${GGML_GEMMINI_RUNTIME_COMPILE_DEFS})
set(GGML_GEMMINI_UTILS_COMPILE_DEFS ${GGML_GEMMINI_RUNTIME_COMPILE_DEFS})

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
