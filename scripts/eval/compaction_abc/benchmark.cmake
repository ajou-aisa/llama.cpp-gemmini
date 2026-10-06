add_executable(compaction-abc "${ABC_SOURCE}/bench.cpp")
target_compile_features(compaction-abc PRIVATE cxx_std_20)
target_compile_options(compaction-abc PRIVATE -O3 -Wall -Wextra -Werror)
target_compile_definitions(compaction-abc PRIVATE ${GGML_GEMMINI_COMPILE_DEFS} LOG_CYCLE=0 CYCLE_DETAIL=0)
target_include_directories(compaction-abc PRIVATE
    "${CMAKE_SOURCE_DIR}/ggml/src/ggml-gemmini"
    "${CMAKE_SOURCE_DIR}/ggml/src"
    "${CMAKE_SOURCE_DIR}/ggml/include"
    "${CMAKE_SOURCE_DIR}/ggml/src/ggml-gemmini-utils/include"
    "${CMAKE_BINARY_DIR}/generated"
    "${GEMMINI_SW_PATH}"
    "${ABC_CYCLE_INCLUDE}")
target_link_libraries(compaction-abc PRIVATE ggml-gemmini "${ABC_CYCLE_LIBRARY}")

add_executable(compaction-cycle "${ABC_SOURCE}/cycle-runner.cpp")
target_compile_features(compaction-cycle PRIVATE cxx_std_20)
target_compile_options(compaction-cycle PRIVATE -O3 -Wall -Wextra -Werror)
target_include_directories(compaction-cycle PRIVATE "${ABC_CYCLE_INCLUDE}")
target_link_libraries(compaction-cycle PRIVATE "${ABC_CYCLE_LIBRARY}")

add_executable(compaction-cached "${ABC_SOURCE}/cached.cpp")
target_compile_features(compaction-cached PRIVATE cxx_std_20)
target_compile_options(compaction-cached PRIVATE -O3 -Wall -Wextra -Werror)
