#include "ggml-gemmini-quantize.h"

// Activation quantization implementation has been moved to orca library.
// This file now acts as a simple wrapper that delegates to orca::ggml adapter.
//
// Rationale:
// - Orca library provides ggml-independent quantization algorithms
// - orca::ggml adapter handles ggml-specific conversions
// - This keeps ggml-gemmini focused on kernel operations
// - Percentile-based scaling has been removed (max-abs only)

#ifdef ORCA_ENABLE_GGML
#include <orca/ggml/ggml_orca.hpp>
#endif

void ggml_gemmini_quantize_activation(const ggml_tensor *src,
                                      ggml_gemmini_args_t &args,
                                      int8_t *dst)
{
#ifdef ORCA_ENABLE_GGML
    // Delegate to orca adapter
    orca::ggml::quants::ggml_gemmini_quantize_activation(src, args, dst);
#else
    // Fallback when orca is not available (should not happen in normal builds)
    args.activation_block_scaled = false;
    args.activation_outliers.clear();
    args.scale_A = 1.0f;

    #if ACTIVATION_BLOCK_SCALE
    args.A_blocks = nullptr;
    args.A_scales = nullptr;
    args.A_scale_rows = 0;
    args.A_scale_cols = 0;
    #endif

    // Simple fallback: set all to zero (should not be used in practice)
    if (src && dst && args.I > 0 && args.K > 0)
    {
        const size_t total = static_cast<size_t>(args.I) * static_cast<size_t>(args.K);
        std::memset(dst, 0, total);
    }
#endif
}
