# Read-only GGUF metadata inspection. No backend initialization or model eval.
add_executable(llama-model-inventory main.cpp)
target_compile_features(llama-model-inventory PRIVATE cxx_std_17)
target_link_libraries(llama-model-inventory PRIVATE ggml-base)
