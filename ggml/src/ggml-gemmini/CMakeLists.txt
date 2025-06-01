message(STATUS "GEMMINI found")

ggml_add_backend_library(ggml-gemmini
                         ggml-gemmini.cpp
                        )

# target_compile_options(ggml-gemmini PRIVATE ${BLAS_LINKER_FLAGS})

# target_link_libraries     (ggml-blas PRIVATE ${BLAS_LIBRARIES})
# target_include_directories(ggml-blas PRIVATE ${BLAS_INCLUDE_DIRS})
