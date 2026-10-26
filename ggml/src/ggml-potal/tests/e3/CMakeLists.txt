# Opt-in acceptance executables only; no production source or default changes.
set(_e3_sources ${POTAL_BACKEND_SOURCES})
list(REMOVE_ITEM _e3_sources ops.cpp)
set(_e3_absolute_sources)
foreach(_source IN LISTS _e3_sources)
    if (IS_ABSOLUTE "${_source}")
        list(APPEND _e3_absolute_sources "${_source}")
    else()
        list(APPEND _e3_absolute_sources "${CMAKE_CURRENT_SOURCE_DIR}/${_source}")
    endif()
endforeach()

add_executable(potal-e3-context-test
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/potal-e3-context.c
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/potal-e3-context-test.cpp)
target_include_directories(potal-e3-context-test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include ${POTAL_HW_GENERATED_DIR})
set_target_properties(potal-e3-context-test PROPERTIES
    C_STANDARD 11 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF)
target_compile_features(potal-e3-context-test PRIVATE cxx_std_17)
if (CMAKE_SYSTEM_PROCESSOR MATCHES "^riscv")
    target_sources(potal-e3-context-test PRIVATE ${POTAL_BACKEND_SOURCES}
        matmul/tiling.cpp matmul/stripe-api.cpp)
    target_include_directories(potal-e3-context-test PRIVATE ${POTAL_PRIVATE_INCLUDES})
    target_compile_definitions(potal-e3-context-test PRIVATE ${POTAL_LOG_DEFINITIONS})
    target_link_libraries(potal-e3-context-test PRIVATE ggml ggml-gemmini-utils Threads::Threads)
endif()

add_executable(potal-e3-prototype-test
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/potal-e3-prototype-test.cpp ${POTAL_BACKEND_SOURCES})
potal_add_backend_settings(potal-e3-prototype-test)
target_include_directories(potal-e3-prototype-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_link_libraries(potal-e3-prototype-test PRIVATE ggml)
add_executable(potal-e3-kbench
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/potal-e3-kbench.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/potal-e3-backend.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ggml-potal.cpp ${_e3_absolute_sources})
potal_add_backend_settings(potal-e3-kbench)
target_include_directories(potal-e3-kbench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_link_libraries(potal-e3-kbench PRIVATE ggml)

add_executable(potal-e3-window-diag
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/potal-e3-window-diag.cpp ${POTAL_BACKEND_SOURCES})
potal_add_backend_settings(potal-e3-window-diag)
target_include_directories(potal-e3-window-diag PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_link_libraries(potal-e3-window-diag PRIVATE ggml)
foreach(_api scale_write scale_release index_config index_write loop_ws_issue
             acc_readback_issue fence session_close)
    target_link_options(potal-e3-window-diag PRIVATE "-Wl,--wrap=potal_paired_${_api}")
endforeach()
