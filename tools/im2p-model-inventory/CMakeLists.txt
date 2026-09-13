# Read-only GGUF metadata inspection. No backend initialization or model eval.
add_executable(llama-im2p-model-inventory main.cpp)
target_compile_features(llama-im2p-model-inventory PRIVATE cxx_std_17)
target_link_libraries(llama-im2p-model-inventory PRIVATE ggml-base)
