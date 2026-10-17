#pragma once

// Fixed execution policy of the PoTal MUL_MAT path (W4-5). These are not
// run-time options: each value names a contract the code implements and the
// logs report, so a run can be identified from its output.

#include "potal.h"

#include <cstdint>

namespace ggml::potal::options {

// Activation quantization: the existing build options GGML_GEMMINI_ACTIVATION_QUANT
// (ActivationQuantAlgo: 0 EXSIA, 1 TENSOR, ...) and GGML_GEMMINI_ACTIVATION_BITS,
// passed by ggml-potal/CMakeLists.txt. TENSOR: one scale per logical src1
// tensor, computed over the whole tensor before tiling.
#if !defined(GGML_GEMMINI_ACTIVATION_QUANT) || !defined(GGML_GEMMINI_ACTIVATION_BITS)
#    error "ggml-potal needs GGML_GEMMINI_ACTIVATION_QUANT and GGML_GEMMINI_ACTIVATION_BITS from CMake"
#endif
static_assert(GGML_GEMMINI_ACTIVATION_QUANT == 1, "ggml-potal implements GGML_GEMMINI_ACTIVATION_QUANT=TENSOR only");
static_assert(GGML_GEMMINI_ACTIVATION_BITS == 8, "PoTal hardware is A8W8");
constexpr const char * activation_quant = "TENSOR";
constexpr int32_t      activation_qmin  = -(int32_t{ 1 } << (GGML_GEMMINI_ACTIVATION_BITS - 1));
constexpr int32_t      activation_qmax  = (int32_t{ 1 } << (GGML_GEMMINI_ACTIVATION_BITS - 1)) - 1;

// Weight format consumed natively: GGML_TYPE_Q8_HP1 (qs, m -> SCU carrier, channel_scale).
constexpr const char * weight_type = "Q8_HP1";

// Error compensation is not part of ggml-potal: no outlier selection, residual
// extraction, RMD/direct correction, compensation GEMM or recomposition exists
// in this backend. The value is reported, never consulted.
constexpr bool         compensation      = false;
constexpr const char * compensation_name = "OFF";

// Scale slots the tiled path may use; every invocation releases its slots
// before the next one writes, so the whole ScaleMemory is one window.
constexpr uint32_t scale_window_base  = 0;
constexpr uint32_t scale_window_slots = POTAL_SCALE_ENTRIES;

} // namespace ggml::potal::options
