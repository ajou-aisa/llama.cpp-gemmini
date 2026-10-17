set(TARGET llama-cli)
add_executable(${TARGET} main.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE common llama ggml-gemmini-utils ${CMAKE_THREAD_LIBS_INIT})
target_compile_definitions(${TARGET} PRIVATE
    CYCLE_SIM=${CYCLE_SIM}
    LOG_DEBUG=${LOG_DEBUG}
    LOG_CYCLE=${LOG_CYCLE})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../ggml/src/ggml-gemmini)
