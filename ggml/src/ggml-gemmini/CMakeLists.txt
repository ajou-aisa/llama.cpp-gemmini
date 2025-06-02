message(STATUS "GEMMINI found")

set(RISCV_TOOL_PATH $ENV{RISCV})
set(GEMMINI_SW_PATH $ENV{GEM_HOME})

ggml_add_backend_library(ggml-gemmini
                         ggml-gemmini.cpp
                        )

target_compile_options(ggml-gemmini PRIVATE
  -std=gnu99
  -fno-common
  -fno-builtin-printf
  -march=rv64gc
  -Wa,-march=rv64gc
  -fno-tree-loop-distribute-patterns
  -fno-common
  -ffast-math
)

target_include_directories(ggml-gemmini PRIVATE
  ${RISCV_TOOL_PATH}/sysroot/include
  ${GEMMINI_SW_PATH}/gemmini-rocc-tests
)
target_link_libraries     (ggml-gemmini PRIVATE
  # ${RISCV_TOOL_PATH}/sysroot/lib/libm.so
)
