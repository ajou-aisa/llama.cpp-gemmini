cmake_minimum_required(VERSION 3.16)

get_filename_component(repo "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
if(NOT DEFINED BUILD_DIR)
    message(FATAL_ERROR "Pass -DBUILD_DIR=<configured native build directory>")
endif()
get_filename_component(build "${BUILD_DIR}" ABSOLUTE BASE_DIR "${repo}")
if(NOT EXISTS "${build}/CMakeCache.txt")
    message(FATAL_ERROR "No configured build at ${build}")
endif()
load_cache("${build}" READ_WITH_PREFIX layout_
    CMAKE_CXX_COMPILER CMAKE_CXX_FLAGS GEMMINI_SW_PATH
    GGML_GEMMINI_ACTIVATION_BITS GGML_GEMMINI_WEIGHT_BITS
    GGML_GEMMINI_BLOCK_SIZE)
separate_arguments(compiler_flags UNIX_COMMAND "${layout_CMAKE_CXX_FLAGS}")
set(probe "${build}/rmd-native-layout-probe")
file(WRITE "${probe}.cpp" [=[
#include "ggml-gemmini-args.h"
#include <cstdio>

int main() {
    ggml_gemmini_args_t args;
    const auto *base = reinterpret_cast<const unsigned char *>(&args);
    const auto offset = [base](const auto *member) {
        return static_cast<size_t>(reinterpret_cast<const unsigned char *>(member) - base);
    };
    std::printf("compiler=%s\n", __VERSION__);
#if defined(_LIBCPP_VERSION)
    std::printf("stdlib=libc++ version=%d\n", _LIBCPP_VERSION);
#elif defined(__GLIBCXX__)
    std::printf("stdlib=libstdc++ version=%ld\n", static_cast<long>(__GLIBCXX__));
#else
    std::printf("stdlib=unidentified\n");
#endif
    std::printf("pointer=%zu string=%zu optional_run_id=%zu\n",
        sizeof(void *), sizeof(std::string), sizeof(std::optional<uint64_t>));
    std::printf("exsia_meta=%zu activation_meta=%zu args=%zu\n",
        sizeof(act::exsia::Meta), sizeof(act::Meta), sizeof(args));
    std::printf("native_weight_bytes=%zu col_stride_f_out=%zu stride_f_out=%zu tile_I=%zu\n",
        offset(&args.native_weight_bytes), offset(&args.col_stride_f_out),
        offset(&args.stride_f_out), offset(&args.tile_I));
}
]=])
execute_process(COMMAND "${layout_CMAKE_CXX_COMPILER}" ${compiler_flags} -std=c++17
    "-DGGML_GEMMINI_ACTIVATION_BITS=${layout_GGML_GEMMINI_ACTIVATION_BITS}"
    "-DGGML_GEMMINI_WEIGHT_BITS=${layout_GGML_GEMMINI_WEIGHT_BITS}"
    "-DGGML_GEMMINI_BLOCK_SIZE=${layout_GGML_GEMMINI_BLOCK_SIZE}"
    "-I${build}/generated" "-I${repo}/ggml/src/ggml-gemmini"
    "-I${repo}/ggml/src" "-I${repo}/ggml/include"
    "-I${repo}/ggml/src/ggml-gemmini-utils/include"
    "-I${layout_GEMMINI_SW_PATH}" "-I${layout_GEMMINI_SW_PATH}/include"
    "${probe}.cpp" -o "${probe}"
    RESULT_VARIABLE compiled)
if(NOT compiled EQUAL 0)
    message(FATAL_ERROR "Native layout probe compilation failed: ${compiled}")
endif()
execute_process(COMMAND "${probe}" RESULT_VARIABLE ran)
if(NOT ran EQUAL 0)
    message(FATAL_ERROR "Native layout probe execution failed: ${ran}")
endif()
message(STATUS "Printed this compiler's header layout; this does not validate a linked IM2P archive.")
