cmake_minimum_required(VERSION 3.14)
project(gemmini_task7_cpu_work LANGUAGES CXX)

enable_testing()

if(NOT DEFINED REPOSITORY_ROOT)
    message(FATAL_ERROR "REPOSITORY_ROOT is required")
endif()

add_executable(test-gemmini-matmul-cpu-work
    ../test-gemmini-matmul-cpu-work.cpp
    ${REPOSITORY_ROOT}/ggml/src/ggml-gemmini/ggml-gemmini-matmul-cpu-work.cpp)
target_compile_features(test-gemmini-matmul-cpu-work PRIVATE cxx_std_17)
target_include_directories(test-gemmini-matmul-cpu-work PRIVATE
    ${REPOSITORY_ROOT}/ggml/src/ggml-gemmini)

add_test(NAME test-gemmini-matmul-cpu-work
    COMMAND test-gemmini-matmul-cpu-work)
add_test(NAME test-gemmini-matmul-cpu-boundaries
    COMMAND ${CMAKE_COMMAND}
        -DMATMUL_SOURCE=${REPOSITORY_ROOT}/ggml/src/ggml-gemmini/ggml-gemmini-matmul.cpp
        -P ${REPOSITORY_ROOT}/tests/test-gemmini-matmul-cpu-boundaries.cmake)
