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
//                                  residual route (matmul/options.cpp)
//   GEMMINI_MATMUL_MODE / GGML_GEMMINI_DEFAULT_MATMUL_MODE
//                                  FULL | STRIPE_PIPELINE (matmul/options.cpp)
//   GEMMINI_STRIPE_JOB_CAPACITY / GGML_GEMMINI_DEFAULT_STRIPE_JOB_CAPACITY
//                                  stripes in flight in STRIPE_PIPELINE
//
// What PoTal executes (supported_combination below); everything else is
// refused, never replaced by another path:
//   FULL             every activation algorithm; compensation ON runs the
//                    cpu_direct residual route (rmd_backend CPU) on the CPU
//   STRIPE_PIPELINE  EXSIA only (matmul/stripe.hpp); compensation ON runs the
//                    ws_packet route (rmd_backend WS, enum gemmini_ws_compact)
//                    on the PoTal WS residual executor "potal_ws": the Residual
//                    GEMM on PoTal, radix reconstruction and merge on the CPU.
//                    Compensation OFF runs the Main GEMMs alone. 1 or 2
//                    semantic stripes in flight.

#include "ggml-gemmini-config.hpp"
#include "ggml-gemmini/matmul/options.hpp"
#include "potal.h"
#include "tiling.hpp"

#include <cstddef>
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

// Scale slots the tiled path may use (tiling.hpp).
constexpr uint32_t scale_window_base  = ::ggml::potal::scale_window_base;
constexpr uint32_t scale_window_slots = ::ggml::potal::scale_window_slots;

using ggml::gemmini::MatmulInvocationMode;
using ggml::gemmini::RmdBackend;

constexpr const char * matmul_mode_name(MatmulInvocationMode mode) {
    return mode == MatmulInvocationMode::full ? "FULL" : "STRIPE_PIPELINE";
}

// The configured residual backend (the shared enum names).
constexpr const char * rmd_backend_name(RmdBackend backend) {
    return backend == RmdBackend::cpu_direct ? "cpu_direct" : "gemmini_ws_compact";
}

// The residual executor that actually runs for a mode: "none" (compensation
// OFF), "cpu_direct" (FULL) or "potal_ws" (STRIPE_PIPELINE, on PoTal).
constexpr const char * residual_executor_name(MatmulInvocationMode mode) {
    return !compensation ? "none" : mode == MatmulInvocationMode::full ? "cpu_direct" : "potal_ws";
}

// GGML_POTAL_PAIRED_AUTO: build default of the paired route (CMake option);
// the environment variable of the same name (ON / OFF / 1 / 0) overrides it.
#ifndef GGML_POTAL_PAIRED_AUTO
#    error "GGML_POTAL_PAIRED_AUTO must come from ggml-potal/CMakeLists.txt"
#endif
constexpr bool paired_auto_default = GGML_POTAL_PAIRED_AUTO != 0;

// The requested paired route: the build default, or the environment override.
// false with *invalid set when the environment value is not ON / OFF / 1 / 0.
bool paired_auto_requested(bool * invalid);

// GGML_POTAL_PAIRED_FENCES=RTL (default) | CONSERVATIVE: CONSERVATIVE adds a POTAL_FENCE
// before every Indexed LOOP_WS, ACC readback and scale release of a paired stripe (the
// H2-validated fence placement) instead of relying on the RTL's ordering of the stream.
// false with *invalid set when the value is neither.
bool paired_fences_conservative(bool * invalid);

// Explicit, default-OFF DIM32 optimization; never enables the resident extension.
bool paired_ktile_requested(bool * invalid);
bool paired_prefetch_requested(bool * invalid);
bool paired_early_prefetch_requested(bool * invalid);

// nullptr when PoTal executes the resolved mode, residual backend and stripe
// job capacity with the configured activation algorithm, else why not.
constexpr const char * unsupported_combination(MatmulInvocationMode mode, RmdBackend backend,
                                               size_t job_capacity) {
    if (mode == MatmulInvocationMode::full) {
        return compensation && backend != RmdBackend::cpu_direct
                   ? "FULL with compensation=ON executes rmd_backend=CPU (cpu_direct) only; the PoTal WS residual "
                     "executor runs in STRIPE_PIPELINE"
                   : nullptr;
    }
    if (activation_quant != ActivationQuantAlgo::EXSIA) {
        return "STRIPE_PIPELINE needs GGML_GEMMINI_ACTIVATION_QUANT=EXSIA (ExSIA stripe-ready events)";
    }
    if (compensation && backend != RmdBackend::gemmini_ws_compact) {
        return "STRIPE_PIPELINE with compensation=ON executes rmd_backend=WS (PoTal WS residual GEMM) only; "
               "the cpu_direct residual is not part of the pipeline";
    }
    if (job_capacity != 1 && job_capacity != 2) {
        return "STRIPE_PIPELINE runs 1 or 2 semantic stripes in flight (GEMMINI_STRIPE_JOB_CAPACITY)";
    }
    return nullptr;
}

} // namespace ggml::potal::options
