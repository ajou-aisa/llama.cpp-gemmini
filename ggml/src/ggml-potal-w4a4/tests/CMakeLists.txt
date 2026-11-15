# W4A4 test selection for the ggml-potal-w4a4 fork. Included (not add_subdirectory) at the END of
# ../CMakeLists.txt, after every add_test, with: include(tests/CMakeLists.txt)
# The fork builds only W4A4 (GGML_GEMMINI_ACTIVATION_BITS=4/WEIGHT_BITS=4), so tests whose fixtures or
# expectations are A8W8 (Q8_HP1 weights, INT8 operands/staging, radix-256 limbs, balanced W8 PV, the
# 256 KiB W8 scratchpad) are kept as targets but DISABLED in ctest; the W4 golden replaces their numerics.

set(POTAL_W4A4_W8_ONLY_TESTS
    potal-api-test
    potal-stripe-api-test
    potal-attention-test
    potal-ggml-unit-test
    potal-log-test
    potal-tiling-unit-test
    potal-residual-unit-test
    potal-stripe-pipeline-test
    potal-hw-timeline-test
    potal-paired-plan-test
    potal-paired-unit-test
    potal-paired-auto-test
    potal-paired-execution-test
    potal-e4b1-test
    potal-e4b2-test
    potal-e4c1-test
    potal-resident-test
    potal-resident-model-test)

if (GGML_GEMMINI_ACTIVATION_BITS STREQUAL "4" AND GGML_GEMMINI_WEIGHT_BITS STREQUAL "4")
    foreach(_w8_test ${POTAL_W4A4_W8_ONLY_TESTS})
        if (TEST ${_w8_test})
            set_tests_properties(${_w8_test} PROPERTIES DISABLED TRUE LABELS "w8-only")
        endif()
    endforeach()
endif()

# Native golden: the fork's W4 dense (paired K-tile on the H2 emulator), QK and PV lowering against
# the independent golden of the existing ggml-gemmini 4-bit algorithm (tests/support/w4a4-golden.hpp).
add_executable(potal-w4a4-golden-test ${CMAKE_CURRENT_LIST_DIR}/potal-w4a4-golden-test.cpp ${POTAL_BACKEND_SOURCES})
potal_add_backend_settings(potal-w4a4-golden-test)
target_include_directories(potal-w4a4-golden-test PRIVATE ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(potal-w4a4-golden-test PRIVATE ggml)
if (NOT CMAKE_CROSSCOMPILING)
    add_test(NAME potal-w4a4-golden-test COMMAND potal-w4a4-golden-test)
    set_tests_properties(potal-w4a4-golden-test PROPERTIES PASS_REGULAR_EXPRESSION "W4A4_GOLDEN PASS cases=")
    add_test(NAME potal-w4a4-counters-test COMMAND potal-w4a4-golden-test dense-random)
    set_tests_properties(potal-w4a4-counters-test PROPERTIES PASS_REGULAR_EXPRESSION "W4A4_COUNTERS dense-random PASS")
endif()
