# Accuracy evaluation only. Existing public options remain until Phase 2D.
if (GGML_GEMMINI_CUDA_RTN OR GGML_GEMMINI_CUDA_RTN_ATTENTION)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    add_library(ggml-cuda-rtn STATIC ${CMAKE_CURRENT_LIST_DIR}/cuda-rtn/ggml-cuda-rtn.cu)
    set_target_properties(ggml-cuda-rtn PROPERTIES POSITION_INDEPENDENT_CODE ON CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
    target_compile_options(ggml-cuda-rtn PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--ftz=false;--fmad=false;--prec-div=true>)
    target_include_directories(ggml-cuda-rtn PRIVATE "${CMAKE_CURRENT_LIST_DIR}/../../../include" "${CMAKE_CURRENT_LIST_DIR}/../..")
    target_link_libraries(ggml-cuda-rtn PUBLIC CUDA::cudart CUDA::cublas ggml-base)
    target_link_libraries(ggml-gemmini PRIVATE ggml-cuda-rtn)
    if (GGML_GEMMINI_CUDA_RTN)
        target_compile_definitions(ggml-gemmini PUBLIC GGML_GEMMINI_CUDA_RTN=1)
    else()
        target_compile_definitions(ggml-gemmini PUBLIC GGML_GEMMINI_CUDA_RTN_ATTENTION=1)
    endif()
endif()
if (GGML_GEMMINI_CUDA_CPU_EXACT)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    add_library(ggml-gemmini-cuda-cpu-exact STATIC
        ${CMAKE_CURRENT_LIST_DIR}/cuda-exact/ggml-cuda-cpu-exact.cu
        ${CMAKE_CURRENT_LIST_DIR}/cuda-exact/ggml-cuda-hp1-scu.cu
        ${CMAKE_CURRENT_LIST_DIR}/cuda-exact/ggml-cuda-potal-attention.cu)
    set_target_properties(ggml-gemmini-cuda-cpu-exact PROPERTIES
        POSITION_INDEPENDENT_CODE ON CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
    target_compile_options(ggml-gemmini-cuda-cpu-exact PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--ftz=false;--fmad=false;--prec-div=true;--prec-sqrt=true>)
    target_include_directories(ggml-gemmini-cuda-cpu-exact PRIVATE "${CMAKE_CURRENT_LIST_DIR}/../../../include" "${CMAKE_CURRENT_LIST_DIR}/../..")
    target_link_libraries(ggml-gemmini-cuda-cpu-exact PUBLIC CUDA::cudart CUDA::cublas ggml-base)
    target_link_libraries(ggml-gemmini PRIVATE ggml-gemmini-cuda-cpu-exact)
    target_compile_definitions(ggml-gemmini PUBLIC GGML_GEMMINI_CUDA_CPU_EXACT=1)
endif()
