set( CMAKE_SYSTEM_NAME Linux )
set( CMAKE_SYSTEM_PROCESSOR riscv64 )

set( target riscv64 )

set(CMAKE_C_COMPILER /home/alveo/firesim/.conda-env/riscv-tools/bin/riscv64-unknown-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER /home/alveo/firesim/.conda-env/riscv-tools/bin/riscv64-unknown-linux-gnu-g++)

set(CMAKE_SYSROOT /home/alveo/firesim/.conda-env/riscv-tools/sysroot)
set(CMAKE_FIND_ROOT_PATH ${CMAKE_SYSROOT})

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

set( CMAKE_C_COMPILER_TARGET   ${target} )
set( CMAKE_CXX_COMPILER_TARGET ${target} )

set( warn_c_flags "-Wno-format -Wno-unused-variable -Wno-unused-function" )

set( CMAKE_C_FLAGS_INIT   "${warn_c_flags}" )
set( CMAKE_CXX_FLAGS_INIT "${warn_c_flags}" )

set( LLAMA_CURL OFF )
set( GGML_OPENMP OFF )
