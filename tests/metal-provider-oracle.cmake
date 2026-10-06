# Optional executed CPU_FUNCTIONAL oracle; the default build needs no sibling checkout.
# Configure with -DGGML_METAL_QUANTIZED_REFERENCE_ROOT=/path/to/IM2P.sim
# plus -DGGML_METAL_QUANTIZED=ON -DGGML_BACKEND_DL=OFF -DLLAMA_BUILD_TESTS=ON.
# Build target test-metal-provider-oracles, then run:
# ctest --test-dir <build> -R '^test-metal-provider-oracle-' --output-on-failure
set(GGML_METAL_QUANTIZED_REFERENCE_ROOT "" CACHE PATH
    "IM2P.sim checkout for optional Metal comparisons with the pinned CPU_FUNCTIONAL provider")

if (GGML_METAL_QUANTIZED_REFERENCE_ROOT STREQUAL "")
    return()
endif()
if (NOT GGML_METAL_QUANTIZED OR NOT TARGET ggml-metal)
    message(FATAL_ERROR "GGML_METAL_QUANTIZED_REFERENCE_ROOT requires GGML_METAL_QUANTIZED=ON and the Metal backend")
endif()
if (GGML_BACKEND_DL)
    message(FATAL_ERROR "Metal provider oracle tests link backend symbols directly; use GGML_BACKEND_DL=OFF")
endif()

set(_metal_oracle_reference "${GGML_METAL_QUANTIZED_REFERENCE_ROOT}")
set(_metal_oracle_commit "bc6168c5ab3cc47edc716ff2c23402bd09867b8d")
set(_metal_oracle_required
    frontend/src/im2p_cpu_functional.cpp
    frontend/src/im2p_cpu_functional_compute.cpp
    frontend/src/im2p_cpu_functional_internal.hpp
    frontend/include/im2p_cpu_functional.hpp
    sim/include/im2p_sim.h
    sim/include/im2p_geometry.h
    sim/include/im2p_compact_runs.h)
foreach(_metal_oracle_file IN LISTS _metal_oracle_required)
    if (NOT EXISTS "${_metal_oracle_reference}/${_metal_oracle_file}")
        message(FATAL_ERROR "Metal provider oracle reference is missing ${_metal_oracle_file} under ${_metal_oracle_reference}")
    endif()
endforeach()
find_package(Git REQUIRED)
execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${_metal_oracle_reference}" rev-parse HEAD
    RESULT_VARIABLE _metal_oracle_git_result OUTPUT_VARIABLE _metal_oracle_actual_commit
    ERROR_VARIABLE _metal_oracle_git_error OUTPUT_STRIP_TRAILING_WHITESPACE)
if (NOT _metal_oracle_git_result EQUAL 0 OR NOT _metal_oracle_actual_commit STREQUAL _metal_oracle_commit)
    message(FATAL_ERROR "Metal provider oracle requires IM2P.sim commit ${_metal_oracle_commit}; got '${_metal_oracle_actual_commit}'. ${_metal_oracle_git_error}")
endif()
execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${_metal_oracle_reference}" diff --quiet HEAD -- ${_metal_oracle_required}
    RESULT_VARIABLE _metal_oracle_git_result)
if (NOT _metal_oracle_git_result EQUAL 0)
    message(FATAL_ERROR "Metal provider oracle source/header files differ from pinned IM2P.sim commit ${_metal_oracle_commit}")
endif()

get_filename_component(_metal_oracle_project "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
set(_metal_oracle_helper "${_metal_oracle_project}/ggml/src/ggml-gemmini/quants/common/hp1_scu.hpp")
file(SHA256 "${_metal_oracle_helper}" _metal_oracle_helper_sha256)
message(STATUS "Metal CPU_FUNCTIONAL oracle: IM2P.sim ${_metal_oracle_commit}, hp1_scu.hpp SHA256 ${_metal_oracle_helper_sha256}")
add_custom_target(test-metal-provider-oracles)
foreach(_metal_oracle_bits IN ITEMS 4 8)
    foreach(_metal_oracle_dim IN ITEMS 16 32 64)
        set(_metal_oracle_target "test-metal-provider-oracle-a${_metal_oracle_bits}-d${_metal_oracle_dim}")
        # Pinned reference's fixed 256 KiB packed scratchpad / 64 KiB INT32 accumulator profiles.
        math(EXPR _metal_oracle_bank_rows "524288 / ${_metal_oracle_bits} / ${_metal_oracle_dim}")
        math(EXPR _metal_oracle_accumulator_rows "16384 / ${_metal_oracle_dim}")
        add_executable(${_metal_oracle_target}
            "${CMAKE_CURRENT_LIST_DIR}/test-metal-provider-oracle.cpp"
            "${_metal_oracle_reference}/frontend/src/im2p_cpu_functional.cpp"
            "${_metal_oracle_reference}/frontend/src/im2p_cpu_functional_compute.cpp")
        target_compile_features(${_metal_oracle_target} PRIVATE cxx_std_20)
        target_compile_options(${_metal_oracle_target} PRIVATE -O2 -fno-fast-math -ffp-contract=off)
        target_compile_definitions(${_metal_oracle_target} PRIVATE
            IM2P_ACTIVATION_BITS=${_metal_oracle_bits}
            IM2P_WEIGHT_BITS=${_metal_oracle_bits}
            IM2P_DIM=${_metal_oracle_dim}
            IM2P_ACCUMULATOR_ROWS=${_metal_oracle_accumulator_rows}
            IM2P_PARTIAL_BITS=32
            IM2P_GEMMINI_BANK_COUNT=4
            IM2P_GEMMINI_BANK_ROWS=${_metal_oracle_bank_rows})
        target_include_directories(${_metal_oracle_target} PRIVATE
            "${_metal_oracle_reference}/frontend/include"
            "${_metal_oracle_reference}/sim/include"
            "${_metal_oracle_project}/ggml/src/ggml-gemmini")
        target_link_libraries(${_metal_oracle_target} PRIVATE ggml-metal ggml-base)
        add_dependencies(test-metal-provider-oracles ${_metal_oracle_target})
        add_test(NAME ${_metal_oracle_target} COMMAND $<TARGET_FILE:${_metal_oracle_target}>)
        set_tests_properties(${_metal_oracle_target} PROPERTIES LABELS "metal;provider-oracle")
    endforeach()
endforeach()
