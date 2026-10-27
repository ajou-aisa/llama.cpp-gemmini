set(TARGET llama-perplexity)
add_executable(${TARGET} perplexity.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
if (GGML_GEMMINI_CUDA_RTN)
    target_link_libraries(${TARGET} PRIVATE ggml-gemmini ggml-cuda-rtn)
endif()
if (GGML_GEMMINI_CUDA_CPU_EXACT)
    target_link_libraries(${TARGET} PRIVATE ggml-gemmini)
    target_compile_definitions(${TARGET} PRIVATE PERPLEXITY_CUDA_CPU_EXACT=1
        EVALUATION_EXACT_COMPUTE="${GGML_GEMMINI_COMPUTE_TYPE}"
        EVALUATION_EXACT_ACTIVATION="${GGML_GEMMINI_ACTIVATION_QUANT}"
        EVALUATION_EXACT_BITS=${GGML_GEMMINI_WEIGHT_BITS}
        EVALUATION_EXACT_DIM=${GGML_GEMMINI_DIM})
endif()
