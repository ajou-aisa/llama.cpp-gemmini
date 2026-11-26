# Compiler and sysroot: cmake/toolchains/riscv64-linux.cmake (set
# RISCV_TOOLCHAIN_ROOT, e.g. the FireSim riscv-tools prefix).
include(${CMAKE_CURRENT_LIST_DIR}/toolchains/riscv64-linux.cmake)

set( target riscv64 )

set( CMAKE_C_COMPILER_TARGET   ${target} )
set( CMAKE_CXX_COMPILER_TARGET ${target} )

set( warn_c_flags "-Wno-format -Wno-unused-variable -Wno-unused-function" )

set( CMAKE_C_FLAGS_INIT   "${warn_c_flags}" )
set( CMAKE_CXX_FLAGS_INIT "${warn_c_flags}" )

set( LLAMA_CURL OFF )
set( GGML_OPENMP OFF )
