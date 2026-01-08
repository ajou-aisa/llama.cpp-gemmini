#include "ggml-gemmini-quantize.h"

// Activation quantization for Gemmini split by concern:
//  - Build a flat view over the source tensor (respects ggml views and strides).
//  - Gather per-element stats (min/max/mean/std/near-zero) plus optional percentile samples for scale selection.
//  - Compute the scale (percentile or max-abs based) with numeric guards.
//  - Try block Q8_0 quantization first: pack blocks, emit per-block scales into dst/args, emit detailed error/saturation stats.
//  - Fall back to per-tensor quantization when block path is not aligned/usable.
// Logging mirrors the previous monolith so downstream diagnostics (DBG_SIMPLE) stay identical.
//
// New readers: the function layout mirrors the legacy inline helper, just factored into
// bite-sized pieces. Each helper has a single job (view prep, stats, scale, block path,
// tensor path, logging) so that future changes do not have to dig through one huge body.

namespace {

// Lightweight view of the activation tensor with computed strides.
struct ActivationView {
    const char *data_ptr = nullptr; // base pointer to first element (view-adjusted)
    size_t stride_i_bytes = 0;      // row stride in bytes
    size_t stride_k_bytes = 0;      // inner (K) stride in bytes
    size_t I = 0;                   // number of rows
    size_t K = 0;                   // number of columns
    size_t total = 0;               // I * K
    bool valid = false;             // false when src/type/total are unusable
};

// Accumulated statistics used for both scaling and debug logs.
struct ActivationStats {
    float max_abs = 0.0f;  // running max |x|
    float min_val = 0.0f;  // running min x
    float max_val = 0.0f;  // running max x
    double sum = 0.0;      // sum x
    double sum_sq = 0.0;   // sum x^2
    size_t near_zero = 0;  // count of |x| < eps
};

// Precomputed values for log formatting to avoid re-deriving later.
struct QuantLogContext {
    const char *layer_name = "";
    size_t total = 0;
    float min_val = 0.0f;
    float max_val = 0.0f;
    double mean = 0.0;
    double stddev = 0.0;
    double zero_ratio = 0.0;
};

// Per-tensor quantization error and saturation counters.
struct TensorQuantStats {
    size_t sat_pos = 0;
    size_t sat_neg = 0;
    long double err_abs_sum = 0.0L;
    long double err_sq_sum = 0.0L;
    long double x_sq_sum = 0.0L;
    float max_abs_err = 0.0f;
};

#if ACTIVATION_BLOCK_SCALE
// Per-block quantization error and saturation counters.
struct BlockQuantStats {
    size_t sat_pos = 0;
    size_t sat_neg = 0;
    long double err_abs_sum = 0.0L;
    long double err_sq_sum = 0.0L;
    long double x_sq_sum = 0.0L;
    float max_abs_err = 0.0f;
    double avg_scale = 0.0;
};

// Thread-local scratch buffers reused across calls to reduce allocations.
struct BlockBuffers {
    std::vector<float> input_linear;   // flattened activations (float)
    std::vector<block_q8_0> blocks;    // quantized Q8_0 blocks
    std::vector<float> dequantized;    // dequantized back to float for error metrics
    std::vector<float> block_scales;   // fp32 scale per block (from fp16 d)
};

static BlockBuffers &block_buffers()
{
    static thread_local BlockBuffers buffers;
    return buffers;
}
#endif // ACTIVATION_BLOCK_SCALE

static ActivationView make_activation_view(const ggml_tensor *src, const ggml_gemmini_args_t &args)
{
    ActivationView view{};

    // Only float activations are supported; early-return keeps logic simple.
    if (src == nullptr || src->type != GGML_TYPE_F32) {
        return view;
    }

    GGML_ASSERT(src->data != nullptr);

    const char *base_ptr = static_cast<const char *>(src->view_src ? src->view_src->data : src->data);
    const size_t view_offs = src->view_src ? src->view_offs : 0; // apply ggml view offset

    // Derive effective data pointer/strides in bytes. Fallback stride for contiguous tensors uses args.K.
    view.data_ptr = base_ptr + view_offs;                    // start of first element
    view.stride_k_bytes = src->nb[0] ? src->nb[0] : sizeof(float); // stride along K (default contiguous float)
    view.stride_i_bytes = src->nb[1]
        ? src->nb[1]
        : static_cast<size_t>(args.K) * view.stride_k_bytes; // stride to next row; default K * stride_k
    view.I = args.I;                                         // rows from args
    view.K = args.K;                                         // cols from args
    view.total = view.I * view.K;                            // element count
    view.valid = view.total != 0;                            // avoid division by zero later
    return view;
}

// Walk the activation view once to:
//  - compute scalar stats (min/max/mean/std proxies)
//  - copy into a flat buffer for Q8_0 block path (when enabled)
//  - optionally subsample magnitudes for percentile-based cap
static void collect_activation_stats(const ActivationView &view,
        ActivationStats &stats,
        std::vector<float> *q80_input_linear,
        std::vector<float> *qact_samples)
{
    // Collect statistics and (optionally) linearize activations for block path.
    // When block path is enabled, activations are flattened into input_linear so they can be fed to quantize_row_q8_0_ref.
    // When percentile sampling is enabled, subsample magnitudes into qact_samples to avoid O(N log N) sort costs.
    if (!view.valid) {
        return;
    }

    if (q80_input_linear) {
        q80_input_linear->resize(view.total); // prepare contiguous buffer for block quantization
    }

    constexpr float zero_eps = 1e-6f; // threshold to consider near-zero
    const float first_val =
        *reinterpret_cast<const float *>(view.data_ptr + 0 * view.stride_i_bytes + 0 * view.stride_k_bytes);
    stats.min_val = first_val; // seed min/max
    stats.max_val = first_val;

    size_t sample_cap = 0;    // how many samples to collect
    size_t sample_stride = 1; // sampling stride across flattened elements
    // Precompute sampling stride to evenly spread QACT_SAMPLE_MAX samples across the tensor.
    if (qact_samples) {
        qact_samples->clear();
        qact_samples->reserve(static_cast<size_t>(QACT_SAMPLE_MAX));
        sample_cap = static_cast<size_t>(QACT_SAMPLE_MAX);
        sample_stride = sample_cap ? std::max<size_t>(size_t{1}, view.total / sample_cap) : size_t{1};
    }

    for (size_t i = 0; i < view.I; ++i) {
        const char *row_ptr = view.data_ptr + i * view.stride_i_bytes;
        for (size_t k = 0; k < view.K; ++k) {
            const float v = *reinterpret_cast<const float *>(row_ptr + k * view.stride_k_bytes);
            if (q80_input_linear) {
                (*q80_input_linear)[i * view.K + k] = v; // flatten into row-major buffer
            }
            stats.max_abs = std::max(stats.max_abs, std::fabs(v));              // update max |x|
            stats.min_val = std::min(stats.min_val, v);                         // update min x
            stats.max_val = std::max(stats.max_val, v);                         // update max x
            stats.sum += static_cast<double>(v);                                // accumulate sum
            stats.sum_sq += static_cast<double>(v) * static_cast<double>(v);    // accumulate sum of squares
            if (std::fabs(v) < zero_eps) {
                ++stats.near_zero; // track sparsity of near-zero values
            }
#if QACT_USE_PERCENTILE
            if (qact_samples) {
                const size_t idx = i * view.K + k;
                if ((idx % sample_stride) == 0 && qact_samples->size() < sample_cap) {
                    qact_samples->push_back(std::fabs(v)); // store magnitude sample
                }
            }
#endif
        }
    }
}

// Decide the quantization cap and derived scale.
// - If percentile sampling is enabled and has data, use that percentile (slightly inflated by 5%) as cap.
// - Otherwise use max-abs.
// - Guard against tiny/NaN values by clamping and falling back to scale 1.0f.
static float compute_activation_scale(float max_abs, const std::vector<float> *qact_samples)
{
    // Choose scale based on max-abs or percentile sample, guarding tiny/NaN cases.
    constexpr float eps = 1e-8f;
    float cap = std::max(max_abs, eps);
#if QACT_USE_PERCENTILE
    if (qact_samples && !qact_samples->empty()) {
        const float pct_raw = QACT_PCTL; // user-configured percentile in [0,1]
        const float pct = pct_raw <= 0.0f ? 0.0f : (pct_raw >= 1.0f ? 1.0f : pct_raw); // clamp percentile
        const size_t last_idx = qact_samples->size() - 1; // largest index in sample array
        const size_t pidx = static_cast<size_t>(std::floor(static_cast<double>(last_idx) * static_cast<double>(pct))); // target percentile index
        std::nth_element(qact_samples->begin(), qact_samples->begin() + pidx, qact_samples->end()); // partial sort to find percentile
        cap = std::max((*qact_samples)[pidx] * 1.05f, eps); // inflate a bit to reduce saturation
    }
#else
    (void) qact_samples;
#endif
    float scale = cap / 127.0f; // symmetric int8 range [-127,127]
    if (!std::isfinite(scale) || scale < eps) {
        scale = 1.0f; // fallback when degenerate
    }
    return scale;
}

// Package stats + args into a struct that feeds logging helpers.
static QuantLogContext make_log_context(const ActivationView &view, const ActivationStats &stats, const ggml_gemmini_args_t &args)
{
    // Precompute values used by both block/tensor log messages.
    QuantLogContext ctx{};
    ctx.layer_name = args.layer_name ? args.layer_name : "";
    ctx.total = view.total;
    ctx.min_val = stats.min_val;
    ctx.max_val = stats.max_val;
    const double mean = stats.sum / static_cast<double>(view.total); // E[x]
    const double variance = std::max(0.0, (stats.sum_sq / static_cast<double>(view.total)) - mean * mean); // Var[x]
    ctx.mean = mean; // store mean
    ctx.stddev = std::sqrt(variance); // store stddev
    ctx.zero_ratio = static_cast<double>(stats.near_zero) / static_cast<double>(view.total); // fraction of near-zero
    return ctx;
}

#if ACTIVATION_BLOCK_SCALE
static void log_block_quant(const QuantLogContext &ctx, const BlockQuantStats &bqs, double sat_ratio)
{
    // Mirrors legacy logging format for block-scaled activations.
    DBG_SIMPLE("[layer=%s][q80.qact] N=%zu scale_A=%.6g sat=%.3f%% min=%g max=%g mean=%.6g std=%.6g mae=%.6g rmse=%.6g max|err|=%.6g snr=%.2f dB near0=%.2f%%",
        ctx.layer_name, ctx.total, bqs.avg_scale, sat_ratio, ctx.min_val, ctx.max_val, ctx.mean, ctx.stddev,
        static_cast<double>(bqs.err_abs_sum) / static_cast<double>(ctx.total),
        std::sqrt(static_cast<double>(bqs.err_sq_sum) / static_cast<double>(ctx.total)),
        bqs.max_abs_err,
        (bqs.err_sq_sum > 0.0L && bqs.x_sq_sum > 0.0L)
            ? 10.0 * std::log10(static_cast<double>(bqs.x_sq_sum / bqs.err_sq_sum))
            : INFINITY,
        ctx.zero_ratio * 100.0);

    if (bqs.sat_pos || bqs.sat_neg) {
        DBG_SIMPLE("[layer=%s][q80.qact.warn] saturation pos=%zu neg=%zu (%.4f%%)",
            ctx.layer_name, bqs.sat_pos, bqs.sat_neg, sat_ratio);
    }
}
#endif // ACTIVATION_BLOCK_SCALE

#if ACTIVATION_TENSOR_SCALE
static void log_tensor_quant(const QuantLogContext &ctx, const TensorQuantStats &tqs, double scale)
{
    // Mirrors legacy logging format for per-tensor activation quantization.
    const double mae = static_cast<double>(tqs.err_abs_sum) / static_cast<double>(ctx.total);
    const double rmse = std::sqrt(static_cast<double>(tqs.err_sq_sum) / static_cast<double>(ctx.total));
    const double snr_db = (tqs.err_sq_sum > 0.0L && tqs.x_sq_sum > 0.0L)
        ? 10.0 * std::log10(static_cast<double>(tqs.x_sq_sum / tqs.err_sq_sum))
        : INFINITY;
    const double sat_ratio = (static_cast<double>(tqs.sat_pos + tqs.sat_neg) * 100.0) / static_cast<double>(ctx.total);

    DBG_SIMPLE("[layer=%s][qact] N=%zu scale_A=%.6g sat=%.3f%% min=%g max=%g mean=%.6g std=%.6g mae=%.6g rmse=%.6g max|err|=%.6g snr=%.2f dB near0=%.2f%%",
        ctx.layer_name, ctx.total, scale, sat_ratio, ctx.min_val, ctx.max_val, ctx.mean, ctx.stddev, mae, rmse, tqs.max_abs_err, snr_db, ctx.zero_ratio * 100.0);

    if (tqs.sat_pos || tqs.sat_neg) {
        DBG_SIMPLE("[layer=%s][qact.warn] saturation pos=%zu neg=%zu (%.4f%%)",
            ctx.layer_name, tqs.sat_pos, tqs.sat_neg, sat_ratio);
    }
}
#endif // ACTIVATION_TENSOR_SCALE

#if ACTIVATION_BLOCK_SCALE
static bool quantize_per_block(const ActivationView &view,
        const QuantLogContext &ctx,
        ggml_gemmini_args_t &args,
        int8_t *dst,
        BlockQuantStats &bqs)
{
    // Try block Q8_0 path; returns true if dst/args were populated with block data.
    if ((view.total % QK8_0) != 0) {
        DBG_SIMPLE("[layer=%s][q80.qact] skipped (N=%zu not aligned to %d)", ctx.layer_name, view.total, QK8_0);
        return false;
    }

    BlockBuffers &buffers = block_buffers();
    const size_t block_cnt = view.total / QK8_0;
    // Quantize contiguous activations to Q8_0 blocks.
    buffers.blocks.resize(block_cnt);
    quantize_row_q8_0_ref(buffers.input_linear.data(), buffers.blocks.data(), static_cast<int64_t>(view.total));

    // Dequantize back to float to measure reconstruction error and saturation versus original input.
    buffers.dequantized.resize(view.total);
    dequantize_row_q8_0(buffers.blocks.data(), buffers.dequantized.data(), static_cast<int64_t>(view.total));

    // Extract per-block scales (fp16 d -> fp32).
    buffers.block_scales.resize(block_cnt);
    for (size_t blk = 0; blk < block_cnt; ++blk) {
        buffers.block_scales[blk] = ggml_fp16_to_fp32(buffers.blocks[blk].d); // d holds fp16 scale
    }

    // Track block-wise min/max from dequantized values to detect saturation when comparing to the original.
    std::vector<float> block_min(block_cnt, std::numeric_limits<float>::infinity());
    std::vector<float> block_max(block_cnt, -std::numeric_limits<float>::infinity());
    for (size_t idx = 0; idx < view.total; ++idx) {
        const size_t blk = idx / QK8_0;
        const float dq = buffers.dequantized[idx];
        block_min[blk] = std::min(block_min[blk], dq);
        block_max[blk] = std::max(block_max[blk], dq);
    }

    // Compute saturation counters and error metrics against the original float input.
    bqs = {};
    for (size_t idx = 0; idx < view.total; ++idx) {
        const size_t blk = idx / QK8_0;
        const float orig = buffers.input_linear[idx];
        const float dq = buffers.dequantized[idx];
        if (orig > block_max[blk]) {
            ++bqs.sat_pos; // original exceeds max representable in this block
        } else if (orig < block_min[blk]) {
            ++bqs.sat_neg; // original below min representable in this block
        }

        const float err = dq - orig; // reconstruction error
        const float aerr = std::fabs(err); // |error|
        bqs.err_abs_sum += aerr; // accumulate |error|
        bqs.err_sq_sum += static_cast<long double>(err) * static_cast<long double>(err); // accumulate error^2
        bqs.x_sq_sum += static_cast<long double>(orig) * static_cast<long double>(orig); // accumulate x^2 (for SNR)
        bqs.max_abs_err = std::max(bqs.max_abs_err, aerr); // track max |error|
    }

    double q80_scale_sum = 0.0; // sum of per-block scales to compute average
    for (const float s : buffers.block_scales) {
        q80_scale_sum += static_cast<double>(s);
    }
    bqs.avg_scale = block_cnt ? (q80_scale_sum / static_cast<double>(block_cnt)) : 0.0; // average scale for logging

    // Wire per-block buffers into args so downstream Gemmini kernel and DEC can reuse them.
    args.A_blocks = buffers.blocks.data();
    args.A_scales = buffers.block_scales.data();
    args.A_scale_rows = 0;
    args.A_scale_cols = 0;

    bool used_block_scale = false;
    // Only mark block-scale usable when rows/cols align to QK8_0 tiling; otherwise fall back to per-tensor.
    if (view.I > 0 && (view.K % QK8_0) == 0) {
        const size_t blocks_per_row = static_cast<size_t>(view.K) / QK8_0;
        if (blocks_per_row != 0 && block_cnt == view.I * blocks_per_row) {
            args.A_scale_rows = view.I;
            args.A_scale_cols = blocks_per_row;
            if (dst != nullptr) {
                for (size_t blk = 0; blk < block_cnt; ++blk) {
                    const block_q8_0 &b = buffers.blocks[blk];
                    std::memcpy(dst + blk * QK8_0, b.qs, QK8_0 * sizeof(int8_t)); // copy quantized bytes into dst
                }
                args.activation_block_scaled = true;
                used_block_scale = true;
            }
        }
    }

    const double sat_ratio = (static_cast<double>(bqs.sat_pos + bqs.sat_neg) * 100.0) / static_cast<double>(ctx.total);
    log_block_quant(ctx, bqs, sat_ratio);
    return used_block_scale;
}
#endif // ACTIVATION_BLOCK_SCALE

#if ACTIVATION_TENSOR_SCALE
static void quantize_per_tensor(const ActivationView &view,
        float scale,
        const QuantLogContext &ctx,
        int8_t *dst)
{
    // Straight per-tensor symmetric quantization using the chosen scale.
    const float inv_scale = 1.0f / scale;
    TensorQuantStats tqs{};

    for (size_t i = 0; i < view.I; ++i) {
        const char *row_ptr = view.data_ptr + i * view.stride_i_bytes;
        int8_t *row_dst = dst + i * view.K;
        for (size_t k = 0; k < view.K; ++k) {
            const float x = *reinterpret_cast<const float *>(row_ptr + k * view.stride_k_bytes);
            float scaled = x * inv_scale; // scale into int8 domain
            int qx = static_cast<int>(std::lrintf(scaled)); // round to nearest
            if (qx > 127) {
                qx = 127;      // clamp positive saturation
                ++tqs.sat_pos; // count pos saturations
            } else if (qx < -127) {
                qx = -127;     // clamp negative saturation
                ++tqs.sat_neg; // count neg saturations
            }
            row_dst[k] = static_cast<int8_t>(qx); // write quantized value

            const float deq = static_cast<float>(qx) * scale; // dequantize for error stats
            const float err = deq - x;                        // reconstruction error
            const float aerr = std::fabs(err);                // |error|
            tqs.err_abs_sum += aerr;                          // accumulate |error|
            tqs.err_sq_sum += static_cast<long double>(err) * static_cast<long double>(err); // accumulate error^2
            tqs.x_sq_sum += static_cast<long double>(x) * static_cast<long double>(x);       // accumulate x^2
            tqs.max_abs_err = std::max(tqs.max_abs_err, aerr);                                // track max |error|
        }
    }

    log_tensor_quant(ctx, tqs, scale);
}
#endif // ACTIVATION_TENSOR_SCALE

} // namespace

void ggml_gemmini_quantize_activation(const ggml_tensor *src,
        ggml_gemmini_args_t &args,
        int8_t *dst)
{
    args.activation_block_scaled = false; // default: assume tensor scale unless block path succeeds

    if (src == nullptr || dst == nullptr) {
        return; // nothing to do without input/output buffers
    }

    const ActivationView view = make_activation_view(src, args);
    if (!view.valid) {
        return; // guard against zero-size or unsupported types
    }

#if ACTIVATION_BLOCK_SCALE
    args.A_blocks = nullptr;
    args.A_scales = nullptr;
    args.A_scale_rows = 0;
    args.A_scale_cols = 0;
#endif

#if ACTIVATION_BLOCK_SCALE
    BlockBuffers &buffers = block_buffers();
    std::vector<float> *q80_input_linear = &buffers.input_linear;
#else
    std::vector<float> *q80_input_linear = nullptr;
#endif

#if QACT_USE_PERCENTILE
    static thread_local std::vector<float> qact_samples;
    std::vector<float> *sample_ptr = &qact_samples;
#else
    std::vector<float> *sample_ptr = nullptr;
#endif

    ActivationStats stats{};
    collect_activation_stats(view, stats, q80_input_linear, sample_ptr); // one pass over activations

    const float scale = compute_activation_scale(stats.max_abs, sample_ptr);
    args.scale_A = scale; // record chosen scale for downstream Gemmini

    const QuantLogContext log_ctx = make_log_context(view, stats, args);
#if !ACTIVATION_BLOCK_SCALE && !ACTIVATION_TENSOR_SCALE
    (void) log_ctx;
#endif

#if ACTIVATION_BLOCK_SCALE
    BlockQuantStats bqs{};
    if (quantize_per_block(view, log_ctx, args, dst, bqs)) {
        return; // block path succeeded; dst/args are populated
    }
#endif

#if ACTIVATION_TENSOR_SCALE
    quantize_per_tensor(view, scale, log_ctx, dst); // fallback tensor path
#endif
}
