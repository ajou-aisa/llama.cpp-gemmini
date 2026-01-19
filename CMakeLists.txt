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

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

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

# ---- Research (AISA) integration ----
option(LLAMA_WITH_ORCA "Build and link ORCA (research) library" ON)

if (LLAMA_WITH_ORCA)
    # llama.cpp-gemmini 와 orca(연구)가 모두 workspace/ 디렉터리라는 가정
    get_filename_component(ORCA_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../orca" ABSOLUTE)

    if (EXISTS "${ORCA_SRC_DIR}/CMakeLists.txt")
        # 외부 소스 디렉터리를 add_subdirectory 할 때는 build dir를 반드시 따로 준다
        add_subdirectory("${ORCA_SRC_DIR}" "${CMAKE_BINARY_DIR}/orca-build")
    else()
        message(FATAL_ERROR "ORCA repo not found: ${ORCA_SRC_DIR}")
    endif()
endif()

# set log option
set(_ggml_orca_defs
    LOG_DEBUG=1
    LOG_CYCLE=1
    ACTIVATION_BLOCK_SCALE=1
    ACTIVATION_TENSOR_SCALE=0
)

# ---- ggml build (must come AFTER orca) ----
if (NOT TARGET ggml AND NOT LLAMA_USE_SYSTEM_GGML)
    add_subdirectory(ggml)
    # ... otherwise assume ggml is added by a parent CMakeLists.txt
endif()

# Apply the same logging macros to ORCA itself so its compiled code matches ggml consumers
if (TARGET orca-log)
    target_compile_definitions(orca-log PUBLIC ${_ggml_orca_defs})
endif()
if (TARGET orca-capi)
    target_compile_definitions(orca-capi PRIVATE ${_ggml_orca_defs})
endif()

# ---- Link ORCA to ggml targets that include <orca/...> ----
if (LLAMA_WITH_ORCA)
    if (TARGET ggml-cpu AND TARGET orca)
        # ggml-cpu는 C API(log/layer)만 필요하므로 ggml 의존이 없는 orca-capi에 연결
        target_link_libraries(ggml-cpu PUBLIC orca-capi)

        # C 타겟이지만 C++ 라이브러리를 링크하므로 링크 단계는 C++로
        set_target_properties(ggml-cpu PROPERTIES LINKER_LANGUAGE CXX)
    endif()

    # gemmini도 orca를 쓰면 동일하게
    if (TARGET ggml-gemmini AND TARGET orca)
        target_link_libraries(ggml-gemmini PUBLIC orca-capi)
        # ggml-gemmini는 C++이므로 LINKER_LANGUAGE는 보통 불필요
    endif()
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
