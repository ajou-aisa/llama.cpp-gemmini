set(TARGET llama-perplexity)
add_executable(${TARGET} perplexity.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
if(GGML_GEMMINI_METAL_CPU_EXACT)
    target_link_libraries(${TARGET} PRIVATE ggml-gemmini)
    target_include_directories(${TARGET} PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src/ggml-metal)
    target_compile_definitions(${TARGET} PRIVATE PERPLEXITY_METAL_CPU_EXACT=1
        EVALUATION_EXACT_COMPUTE="${GGML_GEMMINI_COMPUTE_TYPE}"
        EVALUATION_EXACT_ACTIVATION="${GGML_GEMMINI_ACTIVATION_QUANT}"
        EVALUATION_EXACT_BITS=${GGML_GEMMINI_WEIGHT_BITS}
        EVALUATION_EXACT_DIM=${GGML_GEMMINI_DIM})
endif()
if(GGML_METAL_QUANTIZED)
    target_include_directories(${TARGET} PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src/ggml-metal)
    target_compile_definitions(${TARGET} PRIVATE
        ${GGML_GEMMINI_RUNTIME_COMPILE_DEFS}
        PERPLEXITY_METAL_QUANTIZED=1
        GGML_GEMMINI_DIM=${GGML_GEMMINI_DIM}
        EVALUATION_ACTIVATION_MODE="${GGML_GEMMINI_ACTIVATION_QUANT}")
    if(LLAMA_BUILD_TESTS AND GGML_GEMMINI_ACTIVATION_QUANT STREQUAL "BLOCK")
        add_executable(test-metal-ppl-coverage test-metal-coverage.cpp)
        target_link_libraries(test-metal-ppl-coverage PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
        target_compile_features(test-metal-ppl-coverage PRIVATE cxx_std_17)
        target_include_directories(test-metal-ppl-coverage PRIVATE $<TARGET_PROPERTY:${TARGET},INCLUDE_DIRECTORIES>)
        target_compile_definitions(test-metal-ppl-coverage PRIVATE $<TARGET_PROPERTY:${TARGET},COMPILE_DEFINITIONS>)
        add_test(NAME test-metal-ppl-coverage COMMAND test-metal-ppl-coverage)
        set_tests_properties(test-metal-ppl-coverage PROPERTIES SKIP_RETURN_CODE 77)
    endif()
endif()
