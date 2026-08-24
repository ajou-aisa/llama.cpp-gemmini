set(TARGET llama-cli)
add_executable(${TARGET} main.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE common llama ggml-gemmini-utils ${CMAKE_THREAD_LIBS_INIT})
target_compile_definitions(${TARGET} PRIVATE
    LOG_DEBUG=${LOG_DEBUG}
    LOG_CYCLE=${LOG_CYCLE})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if (NOT DEFINED GEMMINI_SW_PATH OR GEMMINI_SW_PATH STREQUAL "")
    if (DEFINED ENV{GEM_HOME} AND NOT "$ENV{GEM_HOME}" STREQUAL "")
        set(GEMMINI_SW_PATH $ENV{GEM_HOME})
    else()
        get_filename_component(LLAMA_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../" ABSOLUTE)
        set(GEMMINI_SW_PATH "${LLAMA_ROOT}/../RISC-V-DynDNN-gemmini-include")
    endif()
endif()
if (NOT EXISTS "${GEMMINI_SW_PATH}/gemmini.h"
    AND NOT EXISTS "${GEMMINI_SW_PATH}/include/gemmini.h"
    AND NOT EXISTS "${GEMMINI_SW_PATH}/gemmini-rocc-tests/include/gemmini.h")
    get_filename_component(LLAMA_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../" ABSOLUTE)
    set(GEMMINI_SW_PATH "${LLAMA_ROOT}/../RISC-V-DynDNN-gemmini-include")
endif()

target_include_directories(${TARGET} PRIVATE
    ${GEMMINI_SW_PATH}
    ${GEMMINI_SW_PATH}/include
    ${GEMMINI_SW_PATH}/gemmini-rocc-tests
    ${GEMMINI_SW_PATH}/gemmini-rocc-tests/include
    ${GEMMINI_SW_PATH}/rocc-software/src
    ${CMAKE_CURRENT_SOURCE_DIR}/../../ggml/src/ggml-gemmini)
