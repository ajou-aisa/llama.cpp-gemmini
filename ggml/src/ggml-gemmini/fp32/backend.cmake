message(STATUS "Gemmini FP32 DIM8 WS backend (opt-in)")
if(NOT EXISTS "${GEMMINI_SW_PATH}/gemmini.h")
    message(FATAL_ERROR "FP32 requires the FP32 DIM8 SDK overlay")
endif()
ggml_add_backend_library(ggml-gemmini
    fp32/backend.cpp fp32/exec.cpp ggml-gemmini-buffer.cpp)
target_compile_features(ggml-gemmini PRIVATE cxx_std_17)
target_compile_definitions(ggml-gemmini PUBLIC GGML_GEMMINI_FP32=1)
target_compile_options(ggml-gemmini PRIVATE -fno-fast-math -ffp-contract=off)
target_include_directories(ggml-gemmini PUBLIC
    "${CMAKE_CURRENT_LIST_DIR}" "${CMAKE_CURRENT_LIST_DIR}/.."
    "${CMAKE_CURRENT_LIST_DIR}/../../../include"
    "${GEMMINI_SW_PATH}")
target_link_libraries(ggml-gemmini PRIVATE Threads::Threads)
if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "riscv")
    target_sources(ggml-gemmini PRIVATE fp32/rocc-model.cpp)
    target_compile_options(ggml-gemmini PRIVATE -march=native)
endif()

if(LLAMA_BUILD_TESTS)
    add_executable(gemmini-fp32-block-test fp32/tests/block.cpp)
    add_executable(gemmini-fp32-attention-test fp32/tests/attention.cpp)
    foreach(t gemmini-fp32-block-test gemmini-fp32-attention-test)
        target_link_libraries(${t} PRIVATE ggml-gemmini ggml-base)
        target_compile_options(${t} PRIVATE -fno-fast-math -ffp-contract=off)
    endforeach()
    cmake_language(DEFER DIRECTORY "${CMAKE_SOURCE_DIR}" CALL
        add_test NAME gemmini-fp32-block-test COMMAND "$<TARGET_FILE:gemmini-fp32-block-test>")
    cmake_language(DEFER DIRECTORY "${CMAKE_SOURCE_DIR}" CALL
        add_test NAME gemmini-fp32-attention-test COMMAND "$<TARGET_FILE:gemmini-fp32-attention-test>")
    add_executable(gemmini-fp32-e2e-runner fp32/tests/runner.cpp)
    target_link_libraries(gemmini-fp32-e2e-runner PRIVATE llama ggml-gemmini ggml-cpu)
    target_compile_options(gemmini-fp32-e2e-runner PRIVATE -fno-fast-math -ffp-contract=off)
    add_executable(golden_replay_fp32 fp32/tests/replay.cpp)
    target_link_libraries(golden_replay_fp32 PRIVATE ggml-base)
    target_compile_options(golden_replay_fp32 PRIVATE -fno-fast-math -ffp-contract=off)
    if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "riscv")
        target_compile_options(golden_replay_fp32 PRIVATE -march=native)
        add_executable(gemmini-fp32-rtl-vector-test fp32/tests/rtl-vectors.cpp)
        target_link_libraries(gemmini-fp32-rtl-vector-test PRIVATE ggml-gemmini ggml-base)
        target_compile_options(gemmini-fp32-rtl-vector-test PRIVATE
            -fno-fast-math -ffp-contract=off -march=native)
    endif()
endif()
