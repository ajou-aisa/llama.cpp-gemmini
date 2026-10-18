#pragma once

// Configuration of the PoTal MUL_MAT path. Every value comes from the existing
// Gemmini build options (cmake/ggml-gemmini-options.cmake -> compile
// definitions -> ggml-gemmini-config.hpp); nothing is fixed here, and the
// selected values are logged with every node so a run identifies itself.
//
//   GGML_GEMMINI_ACTIVATION_QUANT  EXSIA | TENSOR | TOKEN | BLOCK | STRIPE
//                                  (quants/act/dispatch.cpp of ggml-gemmini)
//   GGML_GEMMINI_ENABLE_RMD        error compensation ON/OFF: outlier residual
//                                  capture in the quantizer + correction/merge
//   GEMMINI_RMD_BACKEND / GGML_GEMMINI_DEFAULT_RMD_BACKEND
//                                  residual route (matmul/options.cpp); PoTal
//                                  executes cpu_direct only

#include "ggml-gemmini-config.hpp"
#include "potal.h"

#include <cstdint>

namespace ggml::potal::options {

static_assert(GGML_GEMMINI_ACTIVATION_BITS == 8 && GGML_GEMMINI_WEIGHT_BITS == 8, "PoTal hardware is A8W8");

using ggml::gemmini::config::ActivationQuantAlgo;
constexpr ActivationQuantAlgo activation_quant = ggml::gemmini::config::CURRENT_ACTIVATION_QUANT;

// Upper-case name of the configured algorithm (the CMake option value).
constexpr const char * activation_quant_name() {
    switch (activation_quant) {
        case ActivationQuantAlgo::EXSIA:  return "EXSIA";
        case ActivationQuantAlgo::TENSOR: return "TENSOR";
        case ActivationQuantAlgo::TOKEN:  return "TOKEN";
        case ActivationQuantAlgo::BLOCK:  return "BLOCK";
        case ActivationQuantAlgo::STRIPE: return "STRIPE";
    }
    return "UNKNOWN";
}

constexpr bool         compensation      = GGML_GEMMINI_ENABLE_RMD != 0;
constexpr const char * compensation_name = compensation ? "ON" : "OFF";

// Weight format consumed natively: GGML_TYPE_Q8_HP1 (qs, m -> SCU carrier, channel_scale).
constexpr const char * weight_type        = "Q8_HP1";
constexpr const char * weight_granularity = "block"; // one carrier per (32-wide K block, output column)

// Scale slots the tiled path may use; every invocation releases its slots
// before the next one writes, so the whole ScaleMemory is one window.
constexpr uint32_t scale_window_base  = 0;
constexpr uint32_t scale_window_slots = POTAL_SCALE_ENTRIES;

} // namespace ggml::potal::options
