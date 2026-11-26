add_executable(llama-cycle-summary main.cpp)
target_link_libraries(llama-cycle-summary PRIVATE ggml-npu-profiling)
target_compile_features(llama-cycle-summary PRIVATE cxx_std_17)
install(TARGETS llama-cycle-summary RUNTIME)
