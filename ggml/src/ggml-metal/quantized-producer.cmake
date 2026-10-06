set(_GGML_METAL_QUANTIZED_PRODUCER_DIR "${CMAKE_CURRENT_LIST_DIR}")

function(ggml_metal_add_quantized_producer target)
    if(NOT GGML_GEMMINI_ACTIVATION_BITS MATCHES "^(4|8)$" OR
       NOT GGML_GEMMINI_ACTIVATION_BITS STREQUAL GGML_GEMMINI_WEIGHT_BITS OR
       NOT GGML_GEMMINI_DIM MATCHES "^(16|32|64)$" OR
       NOT GGML_GEMMINI_BLOCK_SIZE STREQUAL "32" OR
       NOT GGML_GEMMINI_COMPUTE_TYPE STREQUAL "INT" OR
       NOT GGML_GEMMINI_ACTIVATION_QUANT MATCHES "^(BLOCK|EXSIA)$")
        message(FATAL_ERROR "Metal quantized matmul requires matched A4W4/A8W8, DIM16/32/64, K32, INT and BLOCK/EXSIA")
    endif()
    if(CYCLE_SIM OR NOT GGML_GEMMINI_EXECUTION_BACKEND STREQUAL "HARDWARE")
        message(FATAL_ERROR "Metal CPU quantization producer must not depend on a simulator or FPGA provider")
    endif()
    if(NOT TARGET ggml-metal-quantized-producer)
        set(_producer "${_GGML_METAL_QUANTIZED_PRODUCER_DIR}")
        get_filename_component(_gemmini "${_producer}/../ggml-gemmini" ABSOLUTE)
        set(_headers "${GEMMINI_SW_PATH}")
        if(NOT _headers)
            get_filename_component(_headers "${_producer}/../../../../RISC-V-DynDNN-gemmini-include" ABSOLUTE)
        endif()
        if(NOT EXISTS "${_headers}/gemmini.h" OR NOT EXISTS "${_headers}/gemmini_params.h")
            message(FATAL_ERROR "Metal CPU producer requires the Gemmini geometry/parameter headers at GEMMINI_SW_PATH")
        endif()
        set(_generated "${CMAKE_CURRENT_BINARY_DIR}/quantized-producer-generated")
        file(MAKE_DIRECTORY "${_generated}")
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_headers}/gemmini_params.h")
        file(READ "${_headers}/gemmini_params.h" _params)
        if(NOT _params MATCHES "#define[ \t]+DIM[ \t]+[0-9]+")
            message(FATAL_ERROR "Gemmini parameter header lacks a numeric DIM")
        endif()
        string(REGEX REPLACE "#define[ \t]+DIM[ \t]+[0-9]+" "#define DIM ${GGML_GEMMINI_DIM}" _params "${_params}")
        file(GENERATE OUTPUT "${_generated}/gemmini_params.h" CONTENT "${_params}")

        add_library(ggml-metal-quantized-producer STATIC
            "${_producer}/ggml-metal-quantized-producer.cpp"
            "${_gemmini}/quants/act/dispatch.cpp"
            "${_gemmini}/quants/act/block/block.cpp"
            "${_gemmini}/quants/act/exsia/exsia.cpp"
            "${_gemmini}/quants/act/tensor/tensor.cpp"
            "${_gemmini}/quants/act/token/token.cpp"
            "${_gemmini}/quants/act/stripe/stripe.cpp"
            "${_gemmini}/quants/common/fp16_util.cpp"
            "${_gemmini}/quants/common/dequant.cpp"
            "${_gemmini}/quants/common/weight_reader.cpp"
            "${_gemmini}/residual/rmd/rmd-builder.cpp"
            "${_gemmini}/residual/rmd/rmd-bitmap-builder.cpp"
            "${_gemmini}/residual/rmd/rmd-run-aware.cpp"
            "${_gemmini}/residual/rmd/rmd-compose.cpp")
        set_target_properties(ggml-metal-quantized-producer PROPERTIES POSITION_INDEPENDENT_CODE ON)
        target_compile_features(ggml-metal-quantized-producer PRIVATE cxx_std_17)
        target_compile_options(ggml-metal-quantized-producer PRIVATE -fno-fast-math -ffp-contract=off)
        target_compile_definitions(ggml-metal-quantized-producer PRIVATE
            ${GGML_GEMMINI_COMPILE_DEFS}
            GGML_BACKEND_BUILD GGML_BACKEND_SHARED
            EXSIA_VALIDATION=0
            GGML_GEMMINI_EXSIA_PROFILE_SCOPE_VALUE=${GGML_GEMMINI_EXSIA_PROFILE_SCOPE_VALUE})
        target_include_directories(ggml-metal-quantized-producer BEFORE PRIVATE "${_generated}")
        target_include_directories(ggml-metal-quantized-producer PRIVATE
            "${GGML_GEMMINI_GENERATED_CONFIG_DIR}" "${_gemmini}" "${_headers}"
            "${_producer}/.." "${_producer}/../../include"
            "${_producer}/../ggml-gemmini-utils/include")
        target_link_libraries(ggml-metal-quantized-producer PRIVATE ggml-base ggml-gemmini-utils)
        if(GGML_GEMMINI_ENABLE_OPENMP)
            find_package(OpenMP REQUIRED COMPONENTS CXX)
            target_compile_definitions(ggml-metal-quantized-producer PRIVATE GGML_GEMMINI_HAS_OPENMP)
            target_link_libraries(ggml-metal-quantized-producer PRIVATE OpenMP::OpenMP_CXX)
        endif()
    endif()
    target_link_libraries(${target} PRIVATE ggml-metal-quantized-producer)
endfunction()
