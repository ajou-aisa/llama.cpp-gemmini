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

#ifndef GGML_GEMMINI_DEQUANT_FP_TEST
#define GGML_GEMMINI_DEQUANT_FP_TEST 0
#endif

#ifndef GGML_GEMMINI_ACTIVATION_QUANT
#define GGML_GEMMINI_ACTIVATION_QUANT 0
#endif

#ifndef GGML_GEMMINI_BLOCK_SIZE
#define GGML_GEMMINI_BLOCK_SIZE 32
#endif

// Balanced Radix-256 Residual Matrix Decomposition. OFF is a residual-compensation
// ablation only: there is no other compensation path.
#ifndef GGML_GEMMINI_ENABLE_RMD
#define GGML_GEMMINI_ENABLE_RMD 1
#endif

#ifndef GGML_GEMMINI_ACTIVATION_BITS
#define GGML_GEMMINI_ACTIVATION_BITS 8
#endif

// ComputeType ----------------------------------------------------------------
// 0 = INT              : activation quant + weight unpacking + int matmul
// 1 = FLOAT            : bypass quant, call matmul_cpu_fp directly
enum class ComputeType : uint8_t {
    INT = 0,
    FLOAT = 1,
};

// ActivationQuantAlgo --------------------------------------------------------
// To add: append enum entry with next integer, update CURRENT_ACTIVATION_QUANT.
enum class ActivationQuantAlgo : uint8_t {
  EXSIA = 0,
  TENSOR = 1,
  TOKEN = 2,
  BLOCK = 3,
  STRIPE = 4,
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
#elif GGML_GEMMINI_ACTIVATION_QUANT == 3
        ActivationQuantAlgo::BLOCK;
#elif GGML_GEMMINI_ACTIVATION_QUANT == 4
    ActivationQuantAlgo::STRIPE;
#else
    #error "Invalid GGML_GEMMINI_ACTIVATION_QUANT value"
#endif

#define GGML_GEMMINI_ACTIVATION_QUANT_NAMEEXSIA "exsia"
#define GGML_GEMMINI_ACTIVATION_QUANT_NAMETENSOR "tensor"
#define GGML_GEMMINI_ACTIVATION_QUANT_NAMETOKEN "token"
#define GGML_GEMMINI_ACTIVATION_QUANT_NAMEBLOCK "block"
#define GGML_GEMMINI_ACTIVATION_QUANT_NAMESTRIPE "stripe"

#if GGML_GEMMINI_ACTIVATION_QUANT == 0
    #define GGML_GEMMINI_ACTIVATION_QUANT_NAME GGML_GEMMINI_ACTIVATION_QUANT_NAMEEXSIA
#elif GGML_GEMMINI_ACTIVATION_QUANT == 1
    #define GGML_GEMMINI_ACTIVATION_QUANT_NAME GGML_GEMMINI_ACTIVATION_QUANT_NAMETENSOR
#elif GGML_GEMMINI_ACTIVATION_QUANT == 2
    #define GGML_GEMMINI_ACTIVATION_QUANT_NAME GGML_GEMMINI_ACTIVATION_QUANT_NAMETOKEN
#elif GGML_GEMMINI_ACTIVATION_QUANT == 3
#define GGML_GEMMINI_ACTIVATION_QUANT_NAME                                     \
  GGML_GEMMINI_ACTIVATION_QUANT_NAMEBLOCK
#elif GGML_GEMMINI_ACTIVATION_QUANT == 4
    #define GGML_GEMMINI_ACTIVATION_QUANT_NAME GGML_GEMMINI_ACTIVATION_QUANT_NAMESTRIPE
#endif

inline constexpr bool DEQUANT_FP_TEST = GGML_GEMMINI_DEQUANT_FP_TEST != 0;

static_assert(static_cast<uint8_t>(CURRENT_COMPUTE_TYPE) <= 1, "CURRENT_COMPUTE_TYPE must be INT or FLOAT");
static_assert(
    static_cast<uint8_t>(CURRENT_ACTIVATION_QUANT) <= 4,
    "CURRENT_ACTIVATION_QUANT must be EXSIA, TENSOR, TOKEN, BLOCK, or STRIPE");
static_assert(GGML_GEMMINI_ACTIVATION_BITS == 4 || GGML_GEMMINI_ACTIVATION_BITS == 8 || GGML_GEMMINI_ACTIVATION_BITS == 16,
              "GGML_GEMMINI_ACTIVATION_BITS must be 4, 8, or 16");

inline constexpr int32_t GGML_GEMMINI_ACTIVATION_QMIN = -(int32_t{1} << (GGML_GEMMINI_ACTIVATION_BITS - 1));
inline constexpr int32_t GGML_GEMMINI_ACTIVATION_QMAX =  (int32_t{1} << (GGML_GEMMINI_ACTIVATION_BITS - 1)) - 1;
inline constexpr int16_t GGML_GEMMINI_ACTIVATION_RHO   = static_cast<int16_t>(GGML_GEMMINI_ACTIVATION_BITS - 2);

} // namespace
