if(CYCLE_SIM)
    if(IM2P_SIM_ROOT STREQUAL "")
        get_filename_component(IM2P_SIM_ROOT "${_GGML_GEMMINI_SOURCE_DIR}/../IM2P.sim" ABSOLUTE)
    endif()
    get_filename_component(IM2P_SIM_ROOT "${IM2P_SIM_ROOT}" REALPATH)
    if(NOT EXISTS "${IM2P_SIM_ROOT}/frontend/src/im2p_cpu_functional.cpp")
        message(FATAL_ERROR "CYCLE_SIM requires the IM2P CPU-functional frontend sources")
    endif()
    if(NOT GGML_GEMMINI_ACTIVATION_BITS MATCHES "^(4|8)$" OR
       NOT GGML_GEMMINI_BLOCK_SIZE STREQUAL "32" OR
       (GGML_GEMMINI_EXECUTION_BACKEND STREQUAL "IM2P_SIM" AND
        NOT IM2P_SIM_IMPLEMENTATION STREQUAL "GEMMINI_HP1"))
        message(FATAL_ERROR "CYCLE_SIM requires an A4W4/A8W8 block32 GEMMINI_HP1 target")
    endif()
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    set(_CYCLE_SIM_GENERATED "${_GGML_GEMMINI_TOP_BINARY_DIR}/cycle-sim-generated")
    set(_CYCLE_SIM_GENERATE
        "${Python3_EXECUTABLE}" "${_GGML_GEMMINI_SOURCE_DIR}/scripts/cycle-sim-build-info.py"
        --sim "${IM2P_SIM_ROOT}" --llama "${_GGML_GEMMINI_SOURCE_DIR}"
        --bits "${GGML_GEMMINI_ACTIVATION_BITS}" --dim "${GGML_GEMMINI_DIM}"
        --out "${_CYCLE_SIM_GENERATED}")
    execute_process(COMMAND ${_CYCLE_SIM_GENERATE}
        RESULT_VARIABLE _CYCLE_SIM_GENERATE_RESULT)
    if(NOT _CYCLE_SIM_GENERATE_RESULT EQUAL 0)
        message(FATAL_ERROR "CYCLE_SIM hardware contract generation failed")
    endif()
    include("${_CYCLE_SIM_GENERATED}/cycle-sim-target.cmake")
    if(GGML_GEMMINI)
        set(_CYCLE_SIM_PARAM_BANK_NUM "${IM2P_CYCLE_SIM_BANK_COUNT}")
        set(_CYCLE_SIM_PARAM_BANK_ROWS "${IM2P_CYCLE_SIM_BANK_ROWS}")
        set(_CYCLE_SIM_PARAM_ACC_ROWS "${IM2P_CYCLE_SIM_ACCUMULATOR_ROWS}")
        foreach(_macro IN ITEMS BANK_NUM BANK_ROWS ACC_ROWS)
            set(_pattern "#define[ \t]+${_macro}[ \t]+[0-9]+")
            string(REGEX MATCHALL "${_pattern}" _matches "${_GGML_GEMMINI_PARAMS_CONTENT}")
            list(LENGTH _matches _count)
            if(NOT _count EQUAL 1)
                message(FATAL_ERROR "CYCLE_SIM requires one numeric ${_macro} in the selected parameter header")
            endif()
            string(REGEX REPLACE "${_pattern}" "#define ${_macro} ${_CYCLE_SIM_PARAM_${_macro}}"
                _GGML_GEMMINI_PARAMS_CONTENT "${_GGML_GEMMINI_PARAMS_CONTENT}")
        endforeach()
        file(WRITE "${GGML_GEMMINI_GENERATED_CONFIG_DIR}/gemmini_params.h"
            "${_GGML_GEMMINI_PARAMS_CONTENT}")
    endif()
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
        "${_GGML_GEMMINI_SOURCE_DIR}/scripts/cycle-sim-build-info.py"
        ${IM2P_CYCLE_SIM_CONFIGURE_INPUTS})
elseif(GGML_GEMMINI_EXECUTION_BACKEND STREQUAL "IM2P_SIM")
    if(IM2P_SIM_IMPLEMENTATION STREQUAL "GEMMINI_HP1")
        set(_GGML_GEMMINI_IM2P_EXPECTED_IMPLEMENTATION
            "gemmini-hp1-integrated-v1")
    else()
        set(_GGML_GEMMINI_IM2P_EXPECTED_IMPLEMENTATION "legacy-bsv-v1")
    endif()
    if(NOT GGML_GEMMINI)
        message(FATAL_ERROR "GGML_GEMMINI_EXECUTION_BACKEND=IM2P_SIM requires GGML_GEMMINI=ON")
    endif()
    if(CMAKE_CROSSCOMPILING OR CMAKE_SYSTEM_PROCESSOR MATCHES "riscv")
        message(FATAL_ERROR "GGML_GEMMINI_EXECUTION_BACKEND=IM2P_SIM is unavailable for RISC-V/cross-compiling builds")
    endif()
    if(IM2P_SIM_ROOT STREQUAL "")
        message(FATAL_ERROR "IM2P_SIM_ROOT is required for GGML_GEMMINI_EXECUTION_BACKEND=IM2P_SIM")
    endif()
    get_filename_component(IM2P_SIM_ROOT "${IM2P_SIM_ROOT}" REALPATH)
    if(NOT EXISTS "${IM2P_SIM_ROOT}/sim/include/im2p_sim.h" OR
       NOT EXISTS "${IM2P_SIM_ROOT}/frontend/include/im2p_gemmini_frontend.hpp")
        message(FATAL_ERROR
            "IM2P_SIM_ROOT='${IM2P_SIM_ROOT}' does not contain the IM2P simulator and frontend headers")
    endif()

    set(GGML_GEMMINI_IM2P_ARTIFACT_ID
        "a${GGML_GEMMINI_ACTIVATION_BITS}-w${GGML_GEMMINI_WEIGHT_BITS}-d${GGML_GEMMINI_DIM}")
    # Keep provenance experiments outside the production selected generation.
    # The existing cache verifier and containment checks remain mandatory.
    if(IM2P_SIM_BUILD_DIR STREQUAL "")
        set(_GGML_GEMMINI_IM2P_BUILD_DIR "${IM2P_SIM_ROOT}/build")
    else()
        get_filename_component(_GGML_GEMMINI_IM2P_BUILD_DIR
            "${IM2P_SIM_BUILD_DIR}" ABSOLUTE)
    endif()
    set(_GGML_GEMMINI_IM2P_CURRENT
        "${_GGML_GEMMINI_IM2P_BUILD_DIR}/selected/${IM2P_SIM_IMPLEMENTATION}/${GGML_GEMMINI_IM2P_ARTIFACT_ID}/current")
    if(NOT IS_SYMLINK "${_GGML_GEMMINI_IM2P_CURRENT}")
        message(FATAL_ERROR
            "Missing atomic IM2P selected generation: ${_GGML_GEMMINI_IM2P_CURRENT}")
    endif()
    get_filename_component(GGML_GEMMINI_IM2P_GENERATION
        "${_GGML_GEMMINI_IM2P_CURRENT}" REALPATH)
    get_filename_component(_GGML_GEMMINI_IM2P_GENERATION_ROOT
        "${_GGML_GEMMINI_IM2P_BUILD_DIR}/selected/${IM2P_SIM_IMPLEMENTATION}/${GGML_GEMMINI_IM2P_ARTIFACT_ID}/generations"
        REALPATH)
    string(FIND "${GGML_GEMMINI_IM2P_GENERATION}/"
        "${_GGML_GEMMINI_IM2P_GENERATION_ROOT}/"
        _GGML_GEMMINI_IM2P_GENERATION_PREFIX)
    if(NOT _GGML_GEMMINI_IM2P_GENERATION_PREFIX EQUAL 0)
        message(FATAL_ERROR
            "IM2P selected generation escapes its private root: ${GGML_GEMMINI_IM2P_GENERATION}")
    endif()
    set(GGML_GEMMINI_IM2P_MANIFEST
        "${GGML_GEMMINI_IM2P_GENERATION}/real-lib.json")
    set(_GGML_GEMMINI_IM2P_CACHE_VERIFIER
        "${IM2P_SIM_ROOT}/scripts/real_lib_cache.py")
    if(NOT EXISTS "${_GGML_GEMMINI_IM2P_CACHE_VERIFIER}")
        message(FATAL_ERROR
            "Missing IM2P cache verifier: ${_GGML_GEMMINI_IM2P_CACHE_VERIFIER}")
    endif()
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    execute_process(
        COMMAND "${Python3_EXECUTABLE}"
            "${_GGML_GEMMINI_IM2P_CACHE_VERIFIER}" verify
            --manifest "${GGML_GEMMINI_IM2P_MANIFEST}"
            --expected-identity "${GGML_GEMMINI_IM2P_ARTIFACT_ID}"
            --expected-implementation "${IM2P_SIM_IMPLEMENTATION}"
            --expected-block-size "${GGML_GEMMINI_BLOCK_SIZE}"
            --expected-platform "${CMAKE_SYSTEM_NAME}"
            --expected-platform-release "${CMAKE_SYSTEM_VERSION}"
            --expected-arch "${CMAKE_SYSTEM_PROCESSOR}"
            --artifact-kind selected
        RESULT_VARIABLE _GGML_GEMMINI_IM2P_MANIFEST_RESULT
        OUTPUT_VARIABLE _GGML_GEMMINI_IM2P_MANIFEST_OUTPUT
        ERROR_VARIABLE _GGML_GEMMINI_IM2P_MANIFEST_ERROR)
    if(NOT _GGML_GEMMINI_IM2P_MANIFEST_RESULT EQUAL 0)
        message(FATAL_ERROR
            "IM2P cache manifest verification failed for ${GGML_GEMMINI_IM2P_ARTIFACT_ID}:\n${_GGML_GEMMINI_IM2P_MANIFEST_OUTPUT}${_GGML_GEMMINI_IM2P_MANIFEST_ERROR}")
    endif()
    if(IM2P_SIM_IMPLEMENTATION STREQUAL "GEMMINI_HP1")
        file(READ "${GGML_GEMMINI_IM2P_MANIFEST}" _GGML_GEMMINI_IM2P_MANIFEST_JSON)
        string(JSON _GGML_GEMMINI_IM2P_CONTRACT GET
            "${_GGML_GEMMINI_IM2P_MANIFEST_JSON}" build_config hardware_lowering_contract)
        foreach(_pair IN ITEMS "BANK_NUM=bank_count" "BANK_ROWS=bank_rows"
                               "ACC_ROWS=accumulator_rows")
            string(REPLACE "=" ";" _parts "${_pair}")
            list(GET _parts 0 _macro)
            list(GET _parts 1 _field)
            string(JSON _resolved GET "${_GGML_GEMMINI_IM2P_CONTRACT}" facts memory "${_field}")
            string(REGEX MATCHALL "#define[ \t]+${_macro}[ \t]+[0-9]+"
                _matches "${_GGML_GEMMINI_PARAMS_CONTENT}")
            list(LENGTH _matches _count)
            if(NOT _count EQUAL 1 OR NOT _resolved MATCHES "^[1-9][0-9]*$")
                message(FATAL_ERROR "Selected IM2P hardware memory contract lacks ${_macro}")
            endif()
            string(REGEX REPLACE "#define[ \t]+${_macro}[ \t]+[0-9]+"
                "#define ${_macro} ${_resolved}" _GGML_GEMMINI_PARAMS_CONTENT
                "${_GGML_GEMMINI_PARAMS_CONTENT}")
        endforeach()
        file(WRITE "${GGML_GEMMINI_GENERATED_CONFIG_DIR}/gemmini_params.h"
            "${_GGML_GEMMINI_PARAMS_CONTENT}")
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
            "${GGML_GEMMINI_IM2P_MANIFEST}")
    endif()
    set(GGML_GEMMINI_IM2P_FRONTEND_ARCHIVE
        "${GGML_GEMMINI_IM2P_GENERATION}/libim2p_gemmini_frontend.a")
    set(GGML_GEMMINI_IM2P_SIM_ARCHIVE
        "${GGML_GEMMINI_IM2P_GENERATION}/libim2p_sim.a")
    foreach(_GGML_GEMMINI_IM2P_ARCHIVE IN ITEMS
            GGML_GEMMINI_IM2P_FRONTEND_ARCHIVE GGML_GEMMINI_IM2P_SIM_ARCHIVE)
        if(NOT EXISTS "${${_GGML_GEMMINI_IM2P_ARCHIVE}}")
            message(FATAL_ERROR
                "Missing matching IM2P ${GGML_GEMMINI_IM2P_ARTIFACT_ID} archive: ${${_GGML_GEMMINI_IM2P_ARCHIVE}}")
        endif()
    endforeach()

    # Validate the public frontend and simulator identity in one native-link
    # probe. GEMMINI_HP1 keeps identity entry points in the same static-library
    # object as the Verilator runtime bridge, so probing libim2p_sim.a alone can
    # accidentally pull in the runtime without its required native link flags.
    find_package(Threads REQUIRED)
    set(_GGML_GEMMINI_IM2P_PAIR_PROBE_SOURCE
        "${_GGML_GEMMINI_BINARY_DIR}/im2p-frontend-pair-probe.cpp")
    file(WRITE "${_GGML_GEMMINI_IM2P_PAIR_PROBE_SOURCE}" [=[
#include "im2p_gemmini_frontend.hpp"
#include "im2p_sim.h"
#include "ggml-gemmini-args.h"
#include <cstdint>
#include <cstdio>
#include <cstring>
static_assert(IM2P_ABI_VERSION == 5, "SCU requires typed ABI5");
int main() {
    const char *revision = im2p_compiled_numerical_semantics_revision();
    const char *implementation = im2p_sim_implementation();
    if (revision == nullptr ||
        std::strcmp(revision, IM2P_SCU_NUMERICAL_REVISION) != 0)
        return 51;
    if (implementation == nullptr)
        return 52;
    std::printf("%s;%u;%u;%u;%u;%u;%u", implementation, im2p_sim_abi_version(),
                im2p_sim_activation_bits(),
                im2p_sim_activation_storage_bytes(),
                im2p_sim_weight_bits(), im2p_sim_weight_storage_bytes(),
                im2p_sim_dim());
    if (im2p_execute_matmul_extended(nullptr, nullptr, nullptr) !=
        IM2P_INVALID_LAYOUT)
        return 9;
    if (im2p_begin_striped_matmul(nullptr, nullptr, nullptr) != IM2P_ERROR)
        return 10;
    if (im2p_publish_stripe(nullptr, nullptr) != IM2P_INVALID_LAYOUT)
        return 11;
    if (im2p::gemmini::compiled_activation_bits() !=
        @GGML_GEMMINI_ACTIVATION_BITS@)
        return 20;
    if (im2p::gemmini::compiled_weight_bits() !=
        @GGML_GEMMINI_WEIGHT_BITS@)
        return 30;
    if (im2p::gemmini::compiled_dim() != @GGML_GEMMINI_DIM@)
        return 40;
    ggml_gemmini_args_t args{};
    const auto *base = reinterpret_cast<const std::uint8_t *>(&args);
    const auto offset = [base](const auto *member) -> std::uint64_t {
        return static_cast<std::uint64_t>(
            reinterpret_cast<const std::uint8_t *>(member) - base);
    };
    const auto linked = im2p::gemmini::compiled_args_layout_fingerprint();
    return linked.size == sizeof(args) &&
                   linked.native_weight_bytes == offset(&args.native_weight_bytes) &&
                   linked.col_stride_f_out == offset(&args.col_stride_f_out) &&
                   linked.stride_f_out == offset(&args.stride_f_out) &&
                   linked.tile_i == offset(&args.tile_I)
               ? 0
               : 50;
}
]=])
    file(READ "${_GGML_GEMMINI_IM2P_PAIR_PROBE_SOURCE}"
        _GGML_GEMMINI_IM2P_PAIR_PROBE_CONTENT)
    string(CONFIGURE "${_GGML_GEMMINI_IM2P_PAIR_PROBE_CONTENT}"
        _GGML_GEMMINI_IM2P_PAIR_PROBE_CONTENT @ONLY)
    file(WRITE "${_GGML_GEMMINI_IM2P_PAIR_PROBE_SOURCE}"
        "${_GGML_GEMMINI_IM2P_PAIR_PROBE_CONTENT}")
    set(_GGML_GEMMINI_IM2P_PAIR_PROBE_INCLUDES
        "${_GGML_GEMMINI_BINARY_DIR}/generated"
        "${_GGML_GEMMINI_SOURCE_DIR}/ggml/src/ggml-gemmini"
        "${_GGML_GEMMINI_SOURCE_DIR}/ggml/include"
        "${_GGML_GEMMINI_SOURCE_DIR}/ggml/src"
        "${_GGML_GEMMINI_SOURCE_DIR}/ggml/src/ggml-gemmini-utils/include"
        "${_GGML_GEMMINI_SOURCE_DIR}/common"
        "${GEMMINI_SW_PATH}"
        "${GEMMINI_SW_PATH}/include"
        "${IM2P_SIM_ROOT}/frontend/include"
        "${IM2P_SIM_ROOT}/sim/include")
    unset(_GGML_GEMMINI_IM2P_PAIR_PROBE_COMPILED CACHE)
    unset(_GGML_GEMMINI_IM2P_PAIR_PROBE_RESULT CACHE)
    try_run(_GGML_GEMMINI_IM2P_PAIR_PROBE_RESULT
        _GGML_GEMMINI_IM2P_PAIR_PROBE_COMPILED
        "${_GGML_GEMMINI_BINARY_DIR}/im2p-frontend-pair-probe"
        SOURCES
            "${_GGML_GEMMINI_IM2P_PAIR_PROBE_SOURCE}"
            "${_GGML_GEMMINI_SOURCE_DIR}/ggml/src/ggml-gemmini-utils/src/optrace.cpp"
            "${_GGML_GEMMINI_SOURCE_DIR}/ggml/src/ggml-gemmini-utils/src/debug.cpp"
            "${_GGML_GEMMINI_SOURCE_DIR}/ggml/src/ggml-gemmini-utils/src/cycle.cpp"
            "${_GGML_GEMMINI_SOURCE_DIR}/ggml/src/ggml-gemmini-utils/src/performance.cpp"
            "${_GGML_GEMMINI_SOURCE_DIR}/ggml/src/ggml-gemmini-utils/src/trace-context.cpp"
        CMAKE_FLAGS
            "-DCMAKE_CXX_STANDARD=20"
            "-DCMAKE_CXX_STANDARD_REQUIRED=ON"
            "-DINCLUDE_DIRECTORIES=${_GGML_GEMMINI_IM2P_PAIR_PROBE_INCLUDES}"
        COMPILE_DEFINITIONS
            "-DGGML_GEMMINI_BLOCK_SIZE=${GGML_GEMMINI_BLOCK_SIZE}"
            "-DGGML_GEMMINI_ACTIVATION_BITS=${GGML_GEMMINI_ACTIVATION_BITS}"
            "-DGGML_GEMMINI_WEIGHT_BITS=${GGML_GEMMINI_WEIGHT_BITS}"
        LINK_LIBRARIES
            "${GGML_GEMMINI_IM2P_FRONTEND_ARCHIVE}"
            "${GGML_GEMMINI_IM2P_SIM_ARCHIVE}"
            Threads::Threads
            ${CMAKE_DL_LIBS}
        RUN_OUTPUT_VARIABLE _GGML_GEMMINI_IM2P_IDENTITY
        COMPILE_OUTPUT_VARIABLE _GGML_GEMMINI_IM2P_PAIR_PROBE_BUILD_OUTPUT)
    if(NOT _GGML_GEMMINI_IM2P_PAIR_PROBE_COMPILED)
        if(_GGML_GEMMINI_IM2P_PAIR_PROBE_BUILD_OUTPUT MATCHES
           "im2p_poll_completed_extended")
            message(FATAL_ERROR
                "Incompatible IM2P pair: simulator archive is missing required symbol im2p_poll_completed_extended")
        endif()
        message(FATAL_ERROR
            "Unable to link the selected IM2P frontend/simulator pair:\n${_GGML_GEMMINI_IM2P_PAIR_PROBE_BUILD_OUTPUT}")
    endif()
    if(_GGML_GEMMINI_IM2P_PAIR_PROBE_RESULT EQUAL 51)
        message(FATAL_ERROR "IM2P SCU numerical revision mismatch: header and archive differ")
    endif()
    if(_GGML_GEMMINI_IM2P_PAIR_PROBE_RESULT EQUAL 52)
        message(FATAL_ERROR "IM2P simulator implementation identity is missing")
    endif()
    if(NOT _GGML_GEMMINI_IM2P_IDENTITY MATCHES
       "^([^;]+);([0-9]+);([0-9]+);([0-9]+);([0-9]+);([0-9]+);([0-9]+)$")
        message(FATAL_ERROR "Malformed IM2P simulator identity '${_GGML_GEMMINI_IM2P_IDENTITY}'")
    endif()
    set(_GGML_GEMMINI_IM2P_IMPLEMENTATION "${CMAKE_MATCH_1}")
    set(_GGML_GEMMINI_IM2P_ABI "${CMAKE_MATCH_2}")
    set(_GGML_GEMMINI_IM2P_ACTIVATION_BITS "${CMAKE_MATCH_3}")
    set(_GGML_GEMMINI_IM2P_ACTIVATION_STORAGE_BYTES "${CMAKE_MATCH_4}")
    set(_GGML_GEMMINI_IM2P_WEIGHT_BITS "${CMAKE_MATCH_5}")
    set(_GGML_GEMMINI_IM2P_WEIGHT_STORAGE_BYTES "${CMAKE_MATCH_6}")
    set(_GGML_GEMMINI_IM2P_DIM "${CMAKE_MATCH_7}")
    if(NOT _GGML_GEMMINI_IM2P_IMPLEMENTATION STREQUAL
       "${_GGML_GEMMINI_IM2P_EXPECTED_IMPLEMENTATION}")
        message(FATAL_ERROR
            "IM2P simulator implementation mismatch: llama requires ${_GGML_GEMMINI_IM2P_EXPECTED_IMPLEMENTATION}, archive reports ${_GGML_GEMMINI_IM2P_IMPLEMENTATION}")
    endif()
    if(NOT _GGML_GEMMINI_IM2P_ABI STREQUAL "5")
        message(FATAL_ERROR
            "IM2P simulator ABI mismatch: llama requires 5, archive reports ${_GGML_GEMMINI_IM2P_ABI}")
    endif()
    if(NOT _GGML_GEMMINI_IM2P_ACTIVATION_BITS STREQUAL "${GGML_GEMMINI_ACTIVATION_BITS}")
        message(FATAL_ERROR
            "IM2P activation width mismatch: llama requests ${GGML_GEMMINI_ACTIVATION_BITS}, archive reports ${_GGML_GEMMINI_IM2P_ACTIVATION_BITS}")
    endif()
    if(NOT _GGML_GEMMINI_IM2P_WEIGHT_BITS STREQUAL "${GGML_GEMMINI_WEIGHT_BITS}")
        message(FATAL_ERROR
            "IM2P weight width mismatch: llama requests ${GGML_GEMMINI_WEIGHT_BITS}, archive reports ${_GGML_GEMMINI_IM2P_WEIGHT_BITS}")
    endif()
    if(GGML_GEMMINI_ACTIVATION_BITS STREQUAL "16")
        set(_GGML_GEMMINI_EXPECTED_ACTIVATION_STORAGE_BYTES 2)
    else()
        set(_GGML_GEMMINI_EXPECTED_ACTIVATION_STORAGE_BYTES 1)
    endif()
    if(GGML_GEMMINI_WEIGHT_BITS STREQUAL "16")
        set(_GGML_GEMMINI_EXPECTED_WEIGHT_STORAGE_BYTES 2)
    else()
        set(_GGML_GEMMINI_EXPECTED_WEIGHT_STORAGE_BYTES 1)
    endif()
    if(NOT _GGML_GEMMINI_IM2P_ACTIVATION_STORAGE_BYTES STREQUAL
       "${_GGML_GEMMINI_EXPECTED_ACTIVATION_STORAGE_BYTES}")
        message(FATAL_ERROR
            "IM2P activation storage mismatch: llama requires ${_GGML_GEMMINI_EXPECTED_ACTIVATION_STORAGE_BYTES}, archive reports ${_GGML_GEMMINI_IM2P_ACTIVATION_STORAGE_BYTES}")
    endif()
    if(NOT _GGML_GEMMINI_IM2P_WEIGHT_STORAGE_BYTES STREQUAL
       "${_GGML_GEMMINI_EXPECTED_WEIGHT_STORAGE_BYTES}")
        message(FATAL_ERROR
            "IM2P weight storage mismatch: llama requires ${_GGML_GEMMINI_EXPECTED_WEIGHT_STORAGE_BYTES}, archive reports ${_GGML_GEMMINI_IM2P_WEIGHT_STORAGE_BYTES}")
    endif()
    if(NOT _GGML_GEMMINI_IM2P_DIM STREQUAL "${GGML_GEMMINI_DIM}")
        message(FATAL_ERROR
            "IM2P DIM mismatch: llama requests ${GGML_GEMMINI_DIM}, archive reports ${_GGML_GEMMINI_IM2P_DIM}")
    endif()
    if(_GGML_GEMMINI_IM2P_PAIR_PROBE_RESULT EQUAL 20)
        message(FATAL_ERROR "IM2P frontend activation width mismatch")
    endif()
    if(_GGML_GEMMINI_IM2P_PAIR_PROBE_RESULT EQUAL 30)
        message(FATAL_ERROR "IM2P frontend weight width mismatch")
    endif()
    if(_GGML_GEMMINI_IM2P_PAIR_PROBE_RESULT EQUAL 40)
        message(FATAL_ERROR "IM2P frontend DIM mismatch")
    endif()
    if(_GGML_GEMMINI_IM2P_PAIR_PROBE_RESULT EQUAL 50)
        message(FATAL_ERROR
            "IM2P frontend args layout mismatch for ${GGML_GEMMINI_IM2P_ARTIFACT_ID}; rebuild the selected frontend archive")
    endif()
    if(NOT _GGML_GEMMINI_IM2P_PAIR_PROBE_RESULT EQUAL 0)
        message(FATAL_ERROR
            "IM2P frontend configuration mismatch for ${GGML_GEMMINI_IM2P_ARTIFACT_ID} (probe result ${_GGML_GEMMINI_IM2P_PAIR_PROBE_RESULT})")
    endif()
    message(STATUS "GGML Gemmini IM2P frontend archive: ${GGML_GEMMINI_IM2P_FRONTEND_ARCHIVE}")
    message(STATUS "GGML Gemmini IM2P simulator archive: ${GGML_GEMMINI_IM2P_SIM_ARCHIVE}")
endif()
