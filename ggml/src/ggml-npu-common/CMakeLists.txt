# CPU-only consumers need profiling without numerical software or an NPU API.
add_subdirectory(profiling)

if (NOT GGML_GEMMINI AND NOT GGML_POTAL)
    return()
endif()

set(GGML_NPU_COMMON_SHARED_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/quantization/activation/dispatch.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/quantization/activation/quantize.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/quantization/activation/block/block.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/quantization/activation/exsia/exsia.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/quantization/activation/exsia/exsia-profile.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/quantization/activation/exsia/local.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/quantization/activation/exsia/folding.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/quantization/activation/tensor/tensor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/quantization/activation/stripe/stripe.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/quantization/activation/token/token.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/quantization/common/dequant.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/quantization/common/weight_reader.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/residual/direct/direct-executor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/residual/rmd/rmd-builder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/residual/rmd/rmd-bitmap-builder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/residual/rmd/rmd-compose.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/matmul/options.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/profiling/telemetry.cpp
    PARENT_SCOPE)

# Tests exercise different quantizers, worker counts and instrumentation ABIs.
# Group identical compilation configurations, then share their source objects.
# This runs after callers have finished assigning their compile properties.
function(npu_common_collect_targets directory)
    get_property(targets DIRECTORY "${directory}" PROPERTY BUILDSYSTEM_TARGETS)
    set_property(GLOBAL APPEND PROPERTY NPU_COMMON_CONSUMERS ${targets})
    get_property(children DIRECTORY "${directory}" PROPERTY SUBDIRECTORIES)
    foreach(child IN LISTS children)
        npu_common_collect_targets("${child}")
    endforeach()
endfunction()

function(npu_common_extract_target consumer)
    get_target_property(kind ${consumer} TYPE)
    if (kind STREQUAL "OBJECT_LIBRARY")
        # Existing strict compile-only probes are their own instrumented variants.
        return()
    endif()
    get_target_property(directory ${consumer} SOURCE_DIR)
    get_target_property(sources ${consumer} SOURCES)
    if (NOT sources)
        return()
    endif()
    set(common)
    set(retained)
    set(source_properties)
    foreach(source IN LISTS sources)
        get_filename_component(absolute "${source}" ABSOLUTE BASE_DIR "${directory}")
        if (absolute MATCHES "/ggml-npu-common/.*\\.(cpp|c)$" AND
            NOT absolute MATCHES "/profiling/src/")
            list(APPEND common "${absolute}")
            foreach(property IN ITEMS COMPILE_OPTIONS COMPILE_DEFINITIONS COMPILE_FLAGS)
                get_source_file_property(value "${absolute}" DIRECTORY "${directory}" ${property})
                if (value)
                    string(APPEND source_properties "${absolute}:${property}=${value}\n")
                endif()
            endforeach()
        else()
            list(APPEND retained "${source}")
        endif()
    endforeach()
    if (NOT common)
        return()
    endif()

    set(signature "${source_properties}")
    foreach(property IN ITEMS COMPILE_DEFINITIONS COMPILE_OPTIONS COMPILE_FEATURES
            CXX_STANDARD CXX_EXTENSIONS CXX_VISIBILITY_PRESET VISIBILITY_INLINES_HIDDEN)
        get_target_property(value ${consumer} ${property})
        if (value STREQUAL "value-NOTFOUND")
            set(value "")
        endif()
        if (ARGV1 STREQUAL "production" AND property STREQUAL "COMPILE_DEFINITIONS")
            list(FILTER value EXCLUDE REGEX "^GGML_GEMMINI_(TESTING|TEST_OBSERVER|DIRECT_METRICS_TESTING)(=|$)")
        endif()
        set(saved_${property} "${value}")
        string(APPEND signature "${property}=${value}\n")
    endforeach()
    get_target_property(links ${consumer} LINK_LIBRARIES)
    set(parallel_library)
    if ("OpenMP::OpenMP_CXX" IN_LIST links)
        find_package(OpenMP REQUIRED COMPONENTS CXX)
        set(parallel_library OpenMP::OpenMP_CXX)
    endif()
    string(APPEND signature "parallel=${parallel_library}")
    string(SHA256 identity "${signature}")
    get_property(library GLOBAL PROPERTY "NPU_COMMON_VARIANT_${identity}")
    if (NOT library)
        if (ARGV1 STREQUAL "production")
            set(library ggml-npu-common)
        else()
            string(SUBSTRING "${identity}" 0 12 short_id)
            set(library "ggml-npu-common-test-${short_id}")
        endif()
        add_library(${library} STATIC)
        set_property(GLOBAL PROPERTY "NPU_COMMON_VARIANT_${identity}" "${library}")
        set_target_properties(${library} PROPERTIES POSITION_INDEPENDENT_CODE ON)
        foreach(property IN ITEMS COMPILE_DEFINITIONS COMPILE_OPTIONS COMPILE_FEATURES
                CXX_STANDARD CXX_EXTENSIONS CXX_VISIBILITY_PRESET VISIBILITY_INLINES_HIDDEN)
            if (NOT saved_${property} STREQUAL "")
                set_property(TARGET ${library} PROPERTY ${property} "${saved_${property}}")
            endif()
        endforeach()
        target_compile_features(${library} PRIVATE cxx_std_17)
        target_include_directories(${library} PRIVATE
            "${CMAKE_CURRENT_FUNCTION_LIST_DIR}"
            "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/.."
            "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../../include"
            "${NPU_GENERATED_DIR}"
            "${GGML_GEMMINI_GENERATED_CONFIG_DIR}")
        target_link_libraries(${library} PRIVATE ggml-base ggml-npu-profiling
            Threads::Threads ${parallel_library})
        if (library STREQUAL "ggml-npu-common")
            install(TARGETS ${library} ARCHIVE)
        endif()
    endif()

    get_target_property(owned ${library} SOURCES)
    foreach(source IN LISTS common)
        if (NOT source IN_LIST owned)
            target_sources(${library} PRIVATE "${source}")
            foreach(property IN ITEMS COMPILE_OPTIONS COMPILE_DEFINITIONS COMPILE_FLAGS)
                get_source_file_property(value "${source}" DIRECTORY "${directory}" ${property})
                if (value)
                    set_source_files_properties("${source}" TARGET_DIRECTORY ${library}
                        PROPERTIES ${property} "${value}")
                endif()
            endforeach()
            list(APPEND owned "${source}")
        endif()
    endforeach()
    if (ARGV1 STREQUAL "production")
        return()
    endif()
    set_property(TARGET ${consumer} PROPERTY SOURCES "${retained}")
    if (consumer STREQUAL "ggml-gemmini" OR consumer STREQUAL "ggml-potal")
        target_link_libraries(${consumer} PUBLIC ${library})
    else()
        target_link_libraries(${consumer} PRIVATE ${library})
    endif()
endfunction()

function(npu_common_finish)
    find_package(Threads REQUIRED)
    npu_common_collect_targets("${CMAKE_SOURCE_DIR}")
    get_property(consumers GLOBAL PROPERTY NPU_COMMON_CONSUMERS)
    # Production owns the canonical target; test variants cannot replace it.
    foreach(backend IN ITEMS ggml-gemmini ggml-potal)
        if (TARGET ${backend})
            npu_common_extract_target(${backend} production)
            npu_common_extract_target(${backend})
        endif()
    endforeach()
    foreach(consumer IN LISTS consumers)
        if (NOT consumer STREQUAL "ggml-gemmini" AND NOT consumer STREQUAL "ggml-potal")
            npu_common_extract_target(${consumer})
        endif()
    endforeach()
endfunction()

cmake_language(DEFER DIRECTORY "${CMAKE_SOURCE_DIR}" CALL npu_common_finish)
