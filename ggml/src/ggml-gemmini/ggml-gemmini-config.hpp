#pragma once

#include <cstdint>

namespace ggml::gemmini::config
{

/* ------------------------------------------------------------------
 * Adding a new option (e.g. a new compute type or quant algorithm)
 * ------------------------------------------------------------------
 * 1.  Assign the next free integer value in the enum below.
 * 2.  Add a corresponding #elif branch in the CURRENT_* mapping so
 *     the macro → enum conversion is complete.
 * 3.  Update the static_assert upper bound to match the new max.
 * 4.  In CMakeLists.txt: extend the STRINGS list and the if/else
 *     chain that produces the numeric compile definition.
 * 5.  In the dispatch site (e.g. quants/act/quantize.cpp) add a new
 *     case / if constexpr branch for the new enum value.
 * ------------------------------------------------------------------ */

#ifndef GGML_GEMMINI_COMPUTE_TYPE
#define GGML_GEMMINI_COMPUTE_TYPE 0
#endif

#ifndef GGML_GEMMINI_ACTIVATION_QUANT
#define GGML_GEMMINI_ACTIVATION_QUANT 0
#endif

#ifndef GGML_GEMMINI_WEIGHT_QUANT
#define GGML_GEMMINI_WEIGHT_QUANT 0
#endif

#ifndef GGML_GEMMINI_BLOCK_SIZE
#define GGML_GEMMINI_BLOCK_SIZE 32
#endif

// ComputeType ----------------------------------------------------------------
// 0 = INT              : activation quant + weight unpacking + int matmul
// 1 = FLOAT            : bypass quant, call matmul_cpu_fp directly
// 2 = DEQUANT_FP_TEST  : activation quant + activation dequant + fp matmul test path
// To add: append enum entry with next integer, update CURRENT_COMPUTE_TYPE.
enum class ComputeType : uint8_t {
    INT = 0,
    FLOAT = 1,
    DEQUANT_FP_TEST = 2,
};

// ActivationQuantAlgo --------------------------------------------------------
// To add: append enum entry with next integer, update CURRENT_ACTIVATION_QUANT.
enum class ActivationQuantAlgo : uint8_t {
    EXSIA = 0,
    TENSOR = 1,
    TOKEN = 2,
};

enum class WeightQuantAlgo : uint8_t {
    TENSOR = 0,
};

// Macro → enum mapping (compile-time) ---------------------------------------
// When adding a new ComputeType value:
//   - add #elif GGML_GEMMINI_COMPUTE_TYPE == N below
//   - add the matching static_assert bound update
inline constexpr ComputeType CURRENT_COMPUTE_TYPE =
#if GGML_GEMMINI_COMPUTE_TYPE == 0
    ComputeType::INT;
#elif GGML_GEMMINI_COMPUTE_TYPE == 1
    ComputeType::FLOAT;
#elif GGML_GEMMINI_COMPUTE_TYPE == 2
    ComputeType::DEQUANT_FP_TEST;
#else
    #error "Invalid GGML_GEMMINI_COMPUTE_TYPE value"
#endif

inline constexpr ActivationQuantAlgo CURRENT_ACTIVATION_QUANT =
#if GGML_GEMMINI_ACTIVATION_QUANT == 0
    ActivationQuantAlgo::EXSIA;
#elif GGML_GEMMINI_ACTIVATION_QUANT == 1
    ActivationQuantAlgo::TENSOR;
#elif GGML_GEMMINI_ACTIVATION_QUANT == 2
    ActivationQuantAlgo::TOKEN;
#else
    #error "Invalid GGML_GEMMINI_ACTIVATION_QUANT value"
#endif

inline constexpr WeightQuantAlgo CURRENT_WEIGHT_QUANT =
#if GGML_GEMMINI_WEIGHT_QUANT == 0
    WeightQuantAlgo::TENSOR;
#else
    #error "Invalid GGML_GEMMINI_WEIGHT_QUANT value"
#endif

#define GGML_GEMMINI_ACTIVATION_QUANT_NAMEEXSIA "exsia"
#define GGML_GEMMINI_ACTIVATION_QUANT_NAMETENSOR "tensor"
#define GGML_GEMMINI_ACTIVATION_QUANT_NAMETOKEN "token"

#if GGML_GEMMINI_ACTIVATION_QUANT == 0
    #define GGML_GEMMINI_ACTIVATION_QUANT_NAME GGML_GEMMINI_ACTIVATION_QUANT_NAMEEXSIA
#elif GGML_GEMMINI_ACTIVATION_QUANT == 1
    #define GGML_GEMMINI_ACTIVATION_QUANT_NAME GGML_GEMMINI_ACTIVATION_QUANT_NAMETENSOR
#elif GGML_GEMMINI_ACTIVATION_QUANT == 2
    #define GGML_GEMMINI_ACTIVATION_QUANT_NAME GGML_GEMMINI_ACTIVATION_QUANT_NAMETOKEN
#endif

static_assert(static_cast<uint8_t>(CURRENT_COMPUTE_TYPE) <= 2, "CURRENT_COMPUTE_TYPE must be INT, FLOAT, or DEQUANT_FP_TEST");
static_assert(static_cast<uint8_t>(CURRENT_ACTIVATION_QUANT) <= 2, "CURRENT_ACTIVATION_QUANT must be EXSIA, TENSOR, or TOKEN");
static_assert(static_cast<uint8_t>(CURRENT_WEIGHT_QUANT) <= 0, "CURRENT_WEIGHT_QUANT must be TENSOR");

} // namespace
