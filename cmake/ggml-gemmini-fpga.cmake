# The normal llama build owns the SCU dependency. No experiment parent target
# may inject a different frontend or simulator after this check.
if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "FPGA_UART currently supports Linux hosts only")
endif()
if(IM2P_FPGA_ARCH STREQUAL "GEMMINI_HP1")
    set(GGML_GEMMINI_FPGA_SOURCE_DIR "${IM2P_SIM_ROOT}/fpga/gemmini_hp1/host")
else()
    set(GGML_GEMMINI_FPGA_SOURCE_DIR "${IM2P_SIM_ROOT}/fpga/scu_block_scale" CACHE PATH
        "Selected IFR3 SCU adapter source directory")
endif()
set(_fpga_required
        "${IM2P_SIM_ROOT}/sim/include/im2p_sim.h"
        "${IM2P_SIM_ROOT}/frontend/include/im2p_gemmini_frontend.hpp"
        "${IM2P_SIM_ROOT}/frontend/src/im2p_gemmini_frontend.cpp"
        "${GGML_GEMMINI_FPGA_SOURCE_DIR}/ggml-gemmini-fpga.cpp"
        "${GGML_GEMMINI_FPGA_SOURCE_DIR}/ggml-gemmini-fpga.hpp"
        "${GGML_GEMMINI_FPGA_SOURCE_DIR}/uart.cpp"
        "${GGML_GEMMINI_FPGA_SOURCE_DIR}/uart.hpp")
if(NOT IM2P_FPGA_ARCH STREQUAL "GEMMINI_HP1")
    list(APPEND _fpga_required
        "${GGML_GEMMINI_FPGA_SOURCE_DIR}/rtl_plugin.hpp"
        "${GGML_GEMMINI_FPGA_SOURCE_DIR}/window_uart.hpp"
        "${GGML_GEMMINI_FPGA_SOURCE_DIR}/window_uart.cpp"
        "${GGML_GEMMINI_FPGA_SOURCE_DIR}/window_protocol.hpp")
endif()
foreach(_required IN LISTS _fpga_required)
    if(NOT EXISTS "${_required}" OR IS_DIRECTORY "${_required}")
        message(FATAL_ERROR "FPGA_UART missing selected input: ${_required}")
    endif()
endforeach()
# Resolve once. Later retargeting of selected/current cannot change this build.
get_filename_component(IM2P_SIM_ROOT "${IM2P_SIM_ROOT}" REALPATH)
get_filename_component(GGML_GEMMINI_FPGA_SOURCE_DIR "${GGML_GEMMINI_FPGA_SOURCE_DIR}" REALPATH)
find_package(Threads REQUIRED)
set(_fpga_probe "${CMAKE_CURRENT_BINARY_DIR}/fpga-abi-probe.cpp")
file(WRITE "${_fpga_probe}" [=[
#include "im2p_sim.h"
#include <bit>
#include <cstdint>
static_assert(IM2P_ABI_VERSION == 5 && IM2P_OUTPUT_SCU_FINAL == 2);
static_assert(IM2P_VECTOR_UNSIGNED_MULTIPLY == 4);
static_assert(IM2P_VECTOR_LEFT_SHIFT == 5);
static_assert(sizeof(*im2p_matmul_desc_t{}.scales) == 4);
static_assert(std::bit_cast<std::uint32_t>(1.0f) == 0x3f800000u);
int main() { return 0; }
]=])
set(_fpga_probe_flags "-DCMAKE_CXX_STANDARD=20" "-DCMAKE_CXX_STANDARD_REQUIRED=ON"
    "-DINCLUDE_DIRECTORIES=${IM2P_SIM_ROOT}/sim/include")
unset(_fpga_probe_compiled CACHE)
unset(_fpga_probe_result CACHE)
try_compile(_fpga_probe_compiled "${CMAKE_CURRENT_BINARY_DIR}/fpga-abi-probe"
    "${_fpga_probe}" CMAKE_FLAGS ${_fpga_probe_flags}
    OUTPUT_VARIABLE _fpga_probe_build)
set(GGML_GEMMINI_FPGA_ABI_RUNTIME "NOT_RUN_physical_identity_requires_CAP")
if(NOT _fpga_probe_compiled)
    message(FATAL_ERROR "FPGA_UART C++20/header ABI compile probe failed:\n${_fpga_probe_build}")
endif()

# The cache key records source/ABI/artifact/compiler/target and linkage mode.
# Reconfiguration refreshes the identity and rebuilds affected targets in place.
if(IM2P_FPGA_ARCH STREQUAL "GEMMINI_HP1")
    if(GGML_GEMMINI_ACTIVATION_BITS STREQUAL "4")
        set(_fpga_packing signed-int4-low-nibble-first)
    else()
        set(_fpga_packing signed-int8)
    endif()
    set(_fpga_identity
        "ABI5;GEMMINI_HP1;HP1_ONLY;${IM2P_GEMMINI_PROFILE_ID};A${GGML_GEMMINI_ACTIVATION_BITS}/W${GGML_GEMMINI_WEIGHT_BITS}/D${GGML_GEMMINI_DIM};ACC32;block32;${_fpga_packing};hp1-fragment-sat32-v1;RMD_OFF\n")
else()
    set(_fpga_identity "ABI5;IFR3;signed-scu-sat-v2;H1;domain2;IFR4;RTL_PLUGIN1;scu_final_integer;H1:op4,HP1:op5;domain2;explicit_main_external:domain1;A8/W8/D16;block32;RMD_${GGML_GEMMINI_ENABLE_RMD}\n")
endif()
set(_fpga_inputs
        "${IM2P_SIM_ROOT}/sim/include/im2p_sim.h"
        "${IM2P_SIM_ROOT}/frontend/include/im2p_gemmini_frontend.hpp"
        "${IM2P_SIM_ROOT}/frontend/src/im2p_gemmini_frontend.cpp"
        "${GGML_GEMMINI_FPGA_SOURCE_DIR}/ggml-gemmini-fpga.hpp"
        "${GGML_GEMMINI_FPGA_SOURCE_DIR}/ggml-gemmini-fpga.cpp"
        "${GGML_GEMMINI_FPGA_SOURCE_DIR}/uart.hpp"
        "${GGML_GEMMINI_FPGA_SOURCE_DIR}/uart.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt"
        "${CMAKE_CURRENT_LIST_FILE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/ggml/CMakeLists.txt"
        "${CMAKE_CURRENT_SOURCE_DIR}/ggml/src/CMakeLists.txt"
        "${CMAKE_CURRENT_SOURCE_DIR}/ggml/src/ggml-quants.c"
        "${CMAKE_CURRENT_SOURCE_DIR}/ggml/src/ggml-quants.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/ggml/src/ggml-common.h"
        "${GGML_GEMMINI_GENERATED_CONFIG_DIR}/ggml-gemmini-matmul-config.hpp")
if(IM2P_FPGA_ARCH STREQUAL "GEMMINI_HP1")
    list(APPEND _fpga_inputs "${IM2P_GEMMINI_RESOLVED_PROFILE}")
else()
    list(APPEND _fpga_inputs
        "${GGML_GEMMINI_FPGA_SOURCE_DIR}/rtl_plugin.hpp"
        "${GGML_GEMMINI_FPGA_SOURCE_DIR}/window_uart.hpp"
        "${GGML_GEMMINI_FPGA_SOURCE_DIR}/window_uart.cpp"
        "${GGML_GEMMINI_FPGA_SOURCE_DIR}/window_protocol.hpp")
endif()
# Include transitive metadata/quantizer headers and backend control flow, not
# just the frontend's two direct headers. Whole CLI/source sealing is separate.
foreach(_tree IN ITEMS
        "${CMAKE_CURRENT_SOURCE_DIR}/ggml/src/ggml-gemmini"
        "${CMAKE_CURRENT_SOURCE_DIR}/ggml/src/ggml-gemmini-utils"
        "${CMAKE_CURRENT_SOURCE_DIR}/ggml/include"
        "${GEMMINI_SW_PATH}")
    file(GLOB_RECURSE _tree_inputs CONFIGURE_DEPENDS LIST_DIRECTORIES false
        "${_tree}/*.h" "${_tree}/*.hpp" "${_tree}/*.c" "${_tree}/*.cpp"
        "${_tree}/*.in" "${_tree}/CMakeLists.txt")
    list(APPEND _fpga_inputs ${_tree_inputs})
endforeach()
if(CMAKE_TOOLCHAIN_FILE)
    list(APPEND _fpga_inputs "${CMAKE_TOOLCHAIN_FILE}")
endif()
list(REMOVE_DUPLICATES _fpga_inputs)
list(SORT _fpga_inputs)
foreach(_input IN LISTS _fpga_inputs)
    file(SHA256 "${_input}" _sha)
    string(APPEND _fpga_identity "${_sha} ${_input}\n")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_input}")
endforeach()
set(_fpga_dl OFF)
if(GGML_BACKEND_DL)
    set(_fpga_dl ON)
endif()
string(APPEND _fpga_identity
    "compiler=${CMAKE_C_COMPILER};${CMAKE_C_COMPILER_ID};${CMAKE_C_COMPILER_VERSION};${CMAKE_CXX_COMPILER};${CMAKE_CXX_COMPILER_ID};${CMAKE_CXX_COMPILER_VERSION}\n"
    "target=${CMAKE_SYSTEM_NAME};${CMAKE_SYSTEM_PROCESSOR};${CMAKE_SYSROOT};${CMAKE_C_COMPILER_TARGET};${CMAKE_CXX_COMPILER_TARGET};${CMAKE_SIZEOF_VOID_P}\n"
    "build=${CMAKE_GENERATOR};${CMAKE_BUILD_TYPE};${CMAKE_CONFIGURATION_TYPES};${CMAKE_POSITION_INDEPENDENT_CODE};${CMAKE_INTERPROCEDURAL_OPTIMIZATION}\n"
    "definitions=${GGML_GEMMINI_COMPILE_DEFS};${GGML_GEMMINI_UTILS_COMPILE_DEFS}\n"
    "parallel=${GGML_GEMMINI_ENABLE_OPENMP};${GGML_GEMMINI_EXSIA_PROFILE_SCOPE};${GGML_GEMMINI_EXSIA_LOCAL_WORKERS}\n"
    "mode=${GGML_GEMMINI_DEFAULT_MATMUL_MODE};${GGML_GEMMINI_EXSIA_DEFAULT_MODE};${GGML_GEMMINI_ENABLE_STRIPE_MATMUL};${GGML_GEMMINI_ENABLE_STRIPE_PIPELINE};${GGML_GEMMINI_ALLOW_RUNTIME_MATMUL_OVERRIDE}\n"
    "shared=${BUILD_SHARED_LIBS};dl=${_fpga_dl};frontend_pic=ON;frontend_flags=-O2,-fno-fast-math\n")
set(_fpga_configs DEBUG RELEASE RELWITHDEBINFO MINSIZEREL ${CMAKE_CONFIGURATION_TYPES})
if(CMAKE_BUILD_TYPE)
    list(APPEND _fpga_configs "${CMAKE_BUILD_TYPE}")
endif()
list(TRANSFORM _fpga_configs TOUPPER)
list(REMOVE_DUPLICATES _fpga_configs)
list(SORT _fpga_configs)
foreach(_kind C CXX EXE_LINKER SHARED_LINKER MODULE_LINKER STATIC_LINKER)
    string(APPEND _fpga_identity "CMAKE_${_kind}_FLAGS=${CMAKE_${_kind}_FLAGS}\n")
    foreach(_config IN LISTS _fpga_configs)
        string(APPEND _fpga_identity "CMAKE_${_kind}_FLAGS_${_config}=${CMAKE_${_kind}_FLAGS_${_config}}\n")
    endforeach()
endforeach()
string(SHA256 _fpga_fingerprint "${_fpga_identity}")
if(DEFINED GGML_GEMMINI_FPGA_CONFIG_FINGERPRINT AND
   NOT GGML_GEMMINI_FPGA_CONFIG_FINGERPRINT STREQUAL _fpga_fingerprint)
    message(STATUS "FPGA_UART input/toolchain/linkage changed; regenerating the existing build")
endif()
set(GGML_GEMMINI_FPGA_CONFIG_FINGERPRINT "${_fpga_fingerprint}" CACHE INTERNAL "SCU build identity")
set(GGML_GEMMINI_FPGA_BUILD_ID "${_fpga_fingerprint}")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/fpga-build-contract.txt"
    "${_fpga_identity}fingerprint=${_fpga_fingerprint}\nabi_runtime=${GGML_GEMMINI_FPGA_ABI_RUNTIME}\n")
if(IM2P_FPGA_ARCH STREQUAL "GEMMINI_HP1")
    message(STATUS
        "GEMMINI backend=FPGA_UART architecture=GEMMINI_HP1 profile=${IM2P_GEMMINI_PROFILE_ID} HP1-only ACC32 block32 RMD=OFF ABI_RUNTIME=${GGML_GEMMINI_FPGA_ABI_RUNTIME}")
else()
    message(STATUS "GEMMINI backend=FPGA_UART ABI5 IFR3 H1 domain2; IFR4/rtl native H1 op4, HP1 op5, SCU final domain2; explicit main_external domain1; RMD=${GGML_GEMMINI_ENABLE_RMD} (SCU residual merge unavailable) ABI_RUNTIME=${GGML_GEMMINI_FPGA_ABI_RUNTIME}")
endif()
message(STATUS "GEMMINI FPGA provisioning: none; physical external executor")
