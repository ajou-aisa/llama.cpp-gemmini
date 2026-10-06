#include "local.hpp"
#include "exsia_shift.hpp"
#include "ggml.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <tuple>
#include <utility>
#include <vector>

#if EXSIA_STAGE_PROFILE_ENABLED
#if defined(__linux__) && defined(__aarch64__)
#define EXSIA_STAGE_CYCLE_READ() ggml::gemmini::cycle::read_sample()
#else
#define EXSIA_STAGE_CYCLE_READ() ggml::gemmini::cycle::read()
#endif
#else
#define EXSIA_STAGE_CYCLE_READ() static_cast<uint64_t>(0)
#endif

namespace ggml::gemmini::quants::act::exsia {
using namespace detail;

static_assert(GGML_GEMMINI_EXSIA_SIGMA > 0, "GGML_GEMMINI_EXSIA_SIGMA must be positive");

namespace {
static inline int32_t quantize_to_i32(float x, int16_t theta) {
    const int16_t neg_inf = std::numeric_limits<int16_t>::min();

    if (theta == neg_inf || !std::isfinite(x))
        return 0;

    const double scaled = std::ldexp(static_cast<double>(x), -static_cast<int>(theta));

    if (!std::isfinite(scaled))
        return scaled < 0.0 ? std::numeric_limits<int32_t>::min()
                            : std::numeric_limits<int32_t>::max();

    const double min_i32 = static_cast<double>(std::numeric_limits<int32_t>::min());
    const double max_i32 = static_cast<double>(std::numeric_limits<int32_t>::max());

    if (scaled <= min_i32)
        return std::numeric_limits<int32_t>::min();
    if (scaled >= max_i32)
        return std::numeric_limits<int32_t>::max();

    return static_cast<int32_t>(std::lrint(scaled));
}

static inline int64_t magnitude_i32(int32_t value) noexcept {
    const int64_t widened = static_cast<int64_t>(value);
    return widened < 0 ? -widened : widened;
}

#if EXSIA_VALIDATION
std::atomic<size_t> validation_block_top2_exp_rescan_counter{0};
#endif

} // namespace

int16_t ExpScanner::unbiased_exp(const float & x) {
    if (x == 0.f || !std::isfinite(x))
        return std::numeric_limits<int16_t>::min();

    return static_cast<int16_t>(std::ilogb(std::abs(x)));
}

void ExpScanner::scan_top2_exp(const std::vector<float> & x, BlockState & blk) {
    const size_t n = x.size();
    GGML_ASSERT(blk.e.size() >= n);
    blk.reset();
    blk.blk_size = n;
    for (size_t i = 0; i < n; ++i) {
        int16_t exp = unbiased_exp(x[i]);
        blk.e[i]    = exp;
        if (exp > blk.e1) {
            blk.e2 = blk.e1;
            blk.e1 = exp;
        } else if (exp < blk.e1 && exp > blk.e2)
            blk.e2 = exp;
    }
}

void ExpScanner::scan_top2_exp(const float * x, size_t count, BlockState & blk) {
    GGML_ASSERT(x != nullptr);
    GGML_ASSERT(blk.e.size() >= count);
    blk.reset();
    blk.blk_size = count;
    for (size_t i = 0; i < count; ++i) {
        const int16_t exp = unbiased_exp(x[i]);
        blk.e[i]          = exp;
        if (exp > blk.e1) {
            blk.e2 = blk.e1;
            blk.e1 = exp;
        } else if (exp < blk.e1 && exp > blk.e2)
            blk.e2 = exp;
    }
}

void ExpScanner::update_block_top2_exp(const BlockMask & mask, BlockState & blk) {
#if EXSIA_VALIDATION
    validation_block_top2_exp_rescan_counter.fetch_add(1, std::memory_order_relaxed);
#endif
    blk.e1 = std::numeric_limits<int16_t>::min();
    blk.e2 = std::numeric_limits<int16_t>::min();

    const size_t n = blk.blk_size;
    for (size_t i = 0; i < n; ++i) {
        if (mask.is_set(i))
            continue;

        const int16_t exp = blk.e[i];
        if (exp > blk.e1) {
            blk.e2 = blk.e1;
            blk.e1 = exp;
        } else if (exp < blk.e1 && exp > blk.e2)
            blk.e2 = exp;
    }
}

#if EXSIA_VALIDATION
void reset_validation_block_top2_exp_rescan_count() {
    validation_block_top2_exp_rescan_counter.store(0, std::memory_order_relaxed);
}

size_t validation_block_top2_exp_rescan_count() {
    return validation_block_top2_exp_rescan_counter.load(std::memory_order_relaxed);
}
#endif

void ExpScanner::update_stripe_top2_exp(StripeState & stripe, int16_t exp) {
    if (exp > stripe.e1) {
        stripe.e2 = stripe.e1;
        stripe.e1 = exp;
    } else if (exp < stripe.e1 && exp > stripe.e2)
        stripe.e2 = exp;
}

std::vector<int32_t> WideQuantizer::quantize_block(const std::vector<float> & x, int16_t theta_b) {
    std::vector<int32_t> q(x.size());
    quantize_block(x, theta_b, q);
    return q;
}

void WideQuantizer::quantize_block(const std::vector<float> & x,
                                   int16_t                    theta_b,
                                   std::vector<int32_t> &     q) const {
    GGML_ASSERT(q.size() >= x.size());
    const bool null_theta = theta_b == std::numeric_limits<int16_t>::min();
    for (size_t i = 0; i < x.size(); ++i)
        q[i] = null_theta ? 0 : quantize_to_i32(x[i], theta_b);
}

std::tuple<std::vector<int32_t>, __int128_t, __int128_t>
WideQuantizer::quantize_block(const std::vector<float> & x,
                              size_t                     row,
                              size_t                     col_offset,
                              const BitMask &            mask,
                              int16_t                    theta_b) {
    std::vector<int32_t> q(x.size());
    __int128_t           S  = 0;
    __int128_t           SS = 0;
    quantize_block(x, row, col_offset, mask, theta_b, q, S, SS);
    return {q, S, SS};
}

void WideQuantizer::quantize_block(const std::vector<float> & x,
                                   size_t                     row,
                                   size_t                     col_offset,
                                   const BitMask &            mask,
                                   int16_t                    theta_b,
                                   std::vector<int32_t> &     q,
                                   __int128_t &               S,
                                   __int128_t &               SS) const {
    const size_t n = x.size();
    GGML_ASSERT(q.size() >= n);

    S                     = 0;
    SS                    = 0;
    const bool use_mask   = mask.rows != 0 && mask.cols != 0;
    const bool null_theta = theta_b == std::numeric_limits<int16_t>::min();
    for (size_t i = 0; i < n; ++i) {
        size_t        col = col_offset + i;
        const int32_t tmp = null_theta ? 0 : quantize_to_i32(x[i], theta_b);
        q[i]              = tmp;
        if (!use_mask || !mask.is_set(row, col)) {
            const __int128_t magnitude = static_cast<__int128_t>(magnitude_i32(tmp));
            S += magnitude;
            SS += magnitude * magnitude;
        }
    }
}

void WideQuantizer::quantize_block(const std::vector<float> & x,
                                   const BlockMask &          mask,
                                   int16_t                    theta_b,
                                   std::vector<int32_t> &     q,
                                   __int128_t &               S,
                                   __int128_t &               SS) const {
    const size_t n = x.size();
    GGML_ASSERT(q.size() >= n);
    GGML_ASSERT(mask.bit_count >= n);

    S                     = 0;
    SS                    = 0;
    const bool null_theta = theta_b == std::numeric_limits<int16_t>::min();
    for (size_t i = 0; i < n; ++i) {
        const int32_t tmp = null_theta ? 0 : quantize_to_i32(x[i], theta_b);
        q[i]              = tmp;
        if (!mask.is_set(i)) {
            const __int128_t magnitude = static_cast<__int128_t>(magnitude_i32(tmp));
            S += magnitude;
            SS += magnitude * magnitude;
        }
    }
}

void WideQuantizer::quantize_block(const float *          x,
                                   size_t                 count,
                                   const BlockMask &      mask,
                                   int16_t                theta_b,
                                   std::vector<int32_t> & q,
                                   __int128_t &           S,
                                   __int128_t &           SS) const {
    GGML_ASSERT(x != nullptr);
    GGML_ASSERT(q.size() >= count);
    GGML_ASSERT(mask.bit_count >= count);

    S                     = 0;
    SS                    = 0;
    const bool null_theta = theta_b == std::numeric_limits<int16_t>::min();
    for (size_t i = 0; i < count; ++i) {
        const int32_t tmp = null_theta ? 0 : quantize_to_i32(x[i], theta_b);
        q[i]              = tmp;
        if (!mask.is_set(i)) {
            const __int128_t magnitude = static_cast<__int128_t>(magnitude_i32(tmp));
            S += magnitude;
            SS += magnitude * magnitude;
        }
    }
}

void WideQuantizer::quantize_block(const float *          x,
                                   size_t                 count,
                                   int16_t                theta_b,
                                   std::vector<int32_t> & q) const {
    GGML_ASSERT(x != nullptr);
    GGML_ASSERT(q.size() >= count);
    const bool null_theta = theta_b == std::numeric_limits<int16_t>::min();
    for (size_t i = 0; i < count; ++i)
        q[i] = null_theta ? 0 : quantize_to_i32(x[i], theta_b);
}

SigmaDetector::SigmaContext SigmaDetector::prepare(__int128_t S, __int128_t SS, size_t N) const {
    SigmaContext context;
    context.n = static_cast<__int128_t>(N);
    context.S = S;
    if (N == 0)
        return context;

    const __int128_t variance_numer = context.n * SS - S * S;
    if (variance_numer <= 0)
        return context;

    const __int128_t tau = GGML_GEMMINI_EXSIA_SIGMA;
    context.threshold    = tau * tau * variance_numer;
    context.valid        = true;
    return context;
}

bool SigmaDetector::detect(int32_t q, const SigmaContext & context) const {
    if (!context.valid)
        return false;

    const __int128_t centered = context.n * static_cast<__int128_t>(magnitude_i32(q)) - context.S;
    return centered > 0 && centered * centered > context.threshold;
}

bool SigmaDetector::detect_sigma(int32_t q, __int128_t S, __int128_t SS, size_t N) {
    return detect(q, prepare(S, SS, N));
}

bool LocalStage::run_optimized(Meta &          meta,
                               ExSIAState &    state,
                               const float *   x,
                               size_t          valid_count,
                               size_t          block_size,
                               size_t          local_row,
                               size_t          blk_idx,
                               StripeScratch & scratch,
                               BlockMask &     block_mask,
                               int32_t *       q_out,
                               int16_t &       block_exp_out
#if EXSIA_BRANCH_COUNTS_ENABLED
                               ,
                               LocalBlockCycleSample & cycle_sample)
#else
)
#endif
{
    GGML_ASSERT(x != nullptr);
    GGML_ASSERT(q_out != nullptr);
    GGML_ASSERT(block_size == state.B_size);
    GGML_ASSERT(valid_count <= block_size);
    (void)local_row;
    (void)blk_idx;
#if GGML_GEMMINI_ACT_QUANT_METRICS
    scratch.actual_requantized = false;
#endif

    if (valid_count == block_size) {
        return run_optimized_full(meta,
                                  x,
                                  block_size,
                                  scratch,
                                  block_mask,
                                  q_out,
                                  block_exp_out
#if EXSIA_BRANCH_COUNTS_ENABLED
                                  ,
                                  cycle_sample
#endif
        );
    }
    return run_optimized_partial(meta,
                                 x,
                                 valid_count,
                                 block_size,
                                 scratch,
                                 block_mask,
                                 q_out,
                                 block_exp_out
#if EXSIA_BRANCH_COUNTS_ENABLED
                                 ,
                                 cycle_sample
#endif
    );
}

bool LocalStage::run_optimized_full(Meta &          meta,
                                    const float *   x,
                                    size_t          block_size,
                                    StripeScratch & scratch,
                                    BlockMask &     block_mask,
                                    int32_t *       q_out,
                                    int16_t &       block_exp_out
#if EXSIA_BRANCH_COUNTS_ENABLED
                                    ,
                                    LocalBlockCycleSample & cycle_sample)
#else
)
#endif
{
#if EXSIA_BRANCH_COUNTS_ENABLED
    cycle_sample = LocalBlockCycleSample{};
#endif

    GGML_ASSERT(x != nullptr);
    GGML_ASSERT(q_out != nullptr);
    BlockState &  blk             = scratch.block;
    const int16_t neg_inf         = std::numeric_limits<int16_t>::min();
    __int128_t    S               = 0;
    __int128_t    SS              = 0;
    size_t        unmasked_count  = 0;
    bool          has_int_outlier = false;
    int16_t       final_exp       = neg_inf;
    int16_t       e_pre           = neg_inf;
    int16_t       theta_pre       = neg_inf;
#if EXSIA_STAGE_PROFILE_ENABLED
#if defined(__linux__) && defined(__aarch64__)
    const auto t0 = EXSIA_STAGE_CYCLE_READ();
#else
    const uint64_t t0 = EXSIA_STAGE_CYCLE_READ();
#endif
#endif

    blk.reset();
    blk.blk_size = block_size;
    block_mask.clear();
    for (size_t i = 0; i < block_size; ++i) {
        const int16_t exp = unit_exp_.unbiased_exp(x[i]);
        blk.e[i]          = exp;
        if (exp > blk.e1) {
            blk.e2 = blk.e1;
            blk.e1 = exp;
            block_mask.clear();
            block_mask.set(i);
        } else if (exp == blk.e1 && exp != neg_inf)
            block_mask.set(i);
        else if (exp < blk.e1 && exp > blk.e2)
            blk.e2 = exp;
    }
    const bool has_second_bucket = blk.e2 != neg_inf;
    if (!has_second_bucket)
        block_mask.clear();
    e_pre = has_second_bucket ? blk.e2 : blk.e1;
#if EXSIA_STAGE_PROFILE_ENABLED
#if defined(__linux__) && defined(__aarch64__)
    const auto t1 = EXSIA_STAGE_CYCLE_READ();
#else
    const uint64_t t1 = EXSIA_STAGE_CYCLE_READ();
#endif
#endif

    theta_pre             = exp_to_theta(e_pre, meta.rho);
    const bool null_theta = theta_pre == neg_inf;
    for (size_t i = 0; i < block_size; ++i) {
        const int32_t tmp = null_theta ? 0 : quantize_to_i32(x[i], theta_pre);
        q_out[i]          = tmp;
        if (block_mask.is_set(i))
            continue;

        const __int128_t magnitude = static_cast<__int128_t>(magnitude_i32(tmp));
        S += magnitude;
        SS += magnitude * magnitude;
        ++unmasked_count;
    }
#if EXSIA_VALIDATION
    scratch.reference.p0_e1    = blk.e1;
    scratch.reference.p0_e2    = blk.e2;
    scratch.reference.p0_e_pre = e_pre;
    std::copy_n(block_mask.words,
                BlockMask::word_count(block_size),
                scratch.reference.p0_top_mask_words.begin());
    scratch.reference.p1_S  = S;
    scratch.reference.p1_SS = SS;
    scratch.reference.p1_N  = unmasked_count;
#endif

#if EXSIA_STAGE_PROFILE_ENABLED
#if defined(__linux__) && defined(__aarch64__)
    const auto t2 = EXSIA_STAGE_CYCLE_READ();
#else
    const uint64_t t2 = EXSIA_STAGE_CYCLE_READ();
#endif
#endif

    const SigmaDetector::SigmaContext sigma_context = unit_sigma_.prepare(S, SS, unmasked_count);
#if EXSIA_BRANCH_COUNTS_ENABLED
    ++cycle_sample.sigma_context_prepare_count;
#endif
    for (size_t i = 0; i < block_size; ++i) {
        if (block_mask.is_set(i))
            continue;

        if (unit_sigma_.detect(q_out[i], sigma_context)) {
            block_mask.set(i);
            has_int_outlier = true;
        } else
            final_exp = std::max(final_exp, blk.e[i]);
    }
#if EXSIA_BRANCH_COUNTS_ENABLED
    cycle_sample.has_int_outlier = has_int_outlier;
#endif
#if EXSIA_VALIDATION
    cycle_sample.final_remaining_exp = final_exp;
#endif

#if EXSIA_STAGE_PROFILE_ENABLED
#if defined(__linux__) && defined(__aarch64__)
    const auto t3 = EXSIA_STAGE_CYCLE_READ();
#else
    const uint64_t t3 = EXSIA_STAGE_CYCLE_READ();
#endif
#endif

    if (!has_int_outlier) {
        blk.e_b     = e_pre;
        blk.theta_b = theta_pre;
#if EXSIA_BRANCH_COUNTS_ENABLED
        cycle_sample.p3_path = P3Path::BypassNoIntegerOutlier;
#endif
    } else {
        blk.e_b     = final_exp;
        blk.theta_b = exp_to_theta(blk.e_b, meta.rho);

        if (blk.theta_b == theta_pre) {
#if EXSIA_BRANCH_COUNTS_ENABLED
            cycle_sample.p3_path = P3Path::BypassSameScale;
#endif
        } else {
            const bool final_null_theta = blk.theta_b == neg_inf;
            for (size_t i = 0; i < block_size; ++i)
                q_out[i] = final_null_theta ? 0 : quantize_to_i32(x[i], blk.theta_b);
#if GGML_GEMMINI_ACT_QUANT_METRICS
            scratch.actual_requantized = true;
#endif
#if EXSIA_BRANCH_COUNTS_ENABLED
            ++cycle_sample.replay_overwrite_count;
            cycle_sample.p3_path = P3Path::Replay;
#endif
        }
    }

    if (force_recompute_ && (!has_int_outlier || blk.theta_b == theta_pre)) {
        for (size_t i = 0; i < block_size; ++i)
            q_out[i] = quantize_to_i32(x[i], blk.theta_b);
#if EXSIA_STAGE_PROFILE_ENABLED
        ++cycle_sample.forced_recompute_count;
#endif
    }
    block_exp_out = blk.e_b;
#if EXSIA_BRANCH_COUNTS_ENABLED
    ++cycle_sample.block_exp_commit_count;
#endif

#if EXSIA_STAGE_PROFILE_ENABLED
#if defined(__linux__) && defined(__aarch64__)
    const auto t4 = EXSIA_STAGE_CYCLE_READ();
#else
    const uint64_t t4 = EXSIA_STAGE_CYCLE_READ();
#endif
#if defined(__linux__) && defined(__aarch64__)
    record_stage_cycles(cycle_sample, {t0, t1, t2, t3, t4});
#else
    cycle_sample.p0 = t1 >= t0 ? t1 - t0 : 0;
    cycle_sample.p1 = t2 >= t1 ? t2 - t1 : 0;
    cycle_sample.p2 = t3 >= t2 ? t3 - t2 : 0;
    cycle_sample.p3 = t4 >= t3 ? t4 - t3 : 0;
#endif
#endif

    return true;
}

bool LocalStage::run_optimized_partial(Meta &          meta,
                                       const float *   x,
                                       size_t          valid_count,
                                       size_t          block_size,
                                       StripeScratch & scratch,
                                       BlockMask &     block_mask,
                                       int32_t *       q_out,
                                       int16_t &       block_exp_out
#if EXSIA_BRANCH_COUNTS_ENABLED
                                       ,
                                       LocalBlockCycleSample & cycle_sample)
#else
)
#endif
{
#if EXSIA_BRANCH_COUNTS_ENABLED
    cycle_sample = LocalBlockCycleSample{};
#endif
    BlockState &  blk     = scratch.block;
    const int16_t neg_inf = std::numeric_limits<int16_t>::min();
    std::copy_n(x, valid_count, blk.x.begin());
    std::fill(blk.x.begin() + valid_count, blk.x.begin() + block_size, 0.0f);
    __int128_t S               = 0;
    __int128_t SS              = 0;
    size_t     unmasked_count  = 0;
    bool       has_int_outlier = false;
    int16_t    final_exp       = neg_inf;
    int16_t    e_pre           = neg_inf;
    int16_t    theta_pre       = neg_inf;
#if EXSIA_STAGE_PROFILE_ENABLED
#if defined(__linux__) && defined(__aarch64__)
    const auto t0 = EXSIA_STAGE_CYCLE_READ();
#else
    const uint64_t t0 = EXSIA_STAGE_CYCLE_READ();
#endif
#endif

    blk.reset();
    blk.blk_size = block_size;
    block_mask.clear();
    for (size_t i = 0; i < valid_count; ++i) {
        const int16_t exp = unit_exp_.unbiased_exp(blk.x[i]);
        blk.e[i]          = exp;
        if (exp > blk.e1) {
            blk.e2 = blk.e1;
            blk.e1 = exp;
            block_mask.clear();
            block_mask.set(i);
        } else if (exp == blk.e1 && exp != neg_inf)
            block_mask.set(i);
        else if (exp < blk.e1 && exp > blk.e2)
            blk.e2 = exp;
    }
    std::fill(blk.e.begin() + valid_count, blk.e.begin() + block_size, neg_inf);
    const bool has_second_bucket = blk.e2 != neg_inf;
    if (!has_second_bucket)
        block_mask.clear();
    e_pre = has_second_bucket ? blk.e2 : blk.e1;
#if EXSIA_STAGE_PROFILE_ENABLED
#if defined(__linux__) && defined(__aarch64__)
    const auto t1 = EXSIA_STAGE_CYCLE_READ();
#else
    const uint64_t t1 = EXSIA_STAGE_CYCLE_READ();
#endif
#endif

    theta_pre             = exp_to_theta(e_pre, meta.rho);
    const bool null_theta = theta_pre == neg_inf;
    for (size_t i = 0; i < block_size; ++i) {
        const int32_t tmp = null_theta ? 0 : quantize_to_i32(blk.x[i], theta_pre);
        q_out[i]          = tmp;
        if (i >= valid_count || block_mask.is_set(i))
            continue;

        const __int128_t magnitude = static_cast<__int128_t>(magnitude_i32(tmp));
        S += magnitude;
        SS += magnitude * magnitude;
        ++unmasked_count;
    }
#if EXSIA_VALIDATION
    scratch.reference.p0_e1    = blk.e1;
    scratch.reference.p0_e2    = blk.e2;
    scratch.reference.p0_e_pre = e_pre;
    std::copy_n(block_mask.words,
                BlockMask::word_count(block_size),
                scratch.reference.p0_top_mask_words.begin());
    scratch.reference.p1_S  = S;
    scratch.reference.p1_SS = SS;
    scratch.reference.p1_N  = unmasked_count;
#endif
#if EXSIA_STAGE_PROFILE_ENABLED
#if defined(__linux__) && defined(__aarch64__)
    const auto t2 = EXSIA_STAGE_CYCLE_READ();
#else
    const uint64_t t2 = EXSIA_STAGE_CYCLE_READ();
#endif
#endif

    const SigmaDetector::SigmaContext sigma_context = unit_sigma_.prepare(S, SS, unmasked_count);
#if EXSIA_BRANCH_COUNTS_ENABLED
    ++cycle_sample.sigma_context_prepare_count;
#endif
    for (size_t i = 0; i < valid_count; ++i) {
        if (block_mask.is_set(i))
            continue;
        if (unit_sigma_.detect(q_out[i], sigma_context)) {
            block_mask.set(i);
            has_int_outlier = true;
        } else
            final_exp = std::max(final_exp, blk.e[i]);
    }
#if EXSIA_BRANCH_COUNTS_ENABLED
    cycle_sample.has_int_outlier = has_int_outlier;
#endif
#if EXSIA_VALIDATION
    cycle_sample.final_remaining_exp = final_exp;
#endif
#if EXSIA_STAGE_PROFILE_ENABLED
#if defined(__linux__) && defined(__aarch64__)
    const auto t3 = EXSIA_STAGE_CYCLE_READ();
#else
    const uint64_t t3 = EXSIA_STAGE_CYCLE_READ();
#endif
#endif

    if (!has_int_outlier) {
        blk.e_b     = e_pre;
        blk.theta_b = theta_pre;
#if EXSIA_BRANCH_COUNTS_ENABLED
        cycle_sample.p3_path = P3Path::BypassNoIntegerOutlier;
#endif
    } else {
        blk.e_b     = final_exp;
        blk.theta_b = exp_to_theta(blk.e_b, meta.rho);
        if (blk.theta_b == theta_pre) {
#if EXSIA_BRANCH_COUNTS_ENABLED
            cycle_sample.p3_path = P3Path::BypassSameScale;
#endif
        } else {
            const bool final_null_theta = blk.theta_b == neg_inf;
            for (size_t i = 0; i < block_size; ++i)
                q_out[i] = final_null_theta ? 0 : quantize_to_i32(blk.x[i], blk.theta_b);
#if GGML_GEMMINI_ACT_QUANT_METRICS
            scratch.actual_requantized = true;
#endif
#if EXSIA_BRANCH_COUNTS_ENABLED
            ++cycle_sample.replay_overwrite_count;
            cycle_sample.p3_path = P3Path::Replay;
#endif
        }
    }

    if (force_recompute_ && (!has_int_outlier || blk.theta_b == theta_pre)) {
        for (size_t i = 0; i < block_size; ++i)
            q_out[i] = quantize_to_i32(blk.x[i], blk.theta_b);
#if EXSIA_STAGE_PROFILE_ENABLED
        ++cycle_sample.forced_recompute_count;
#endif
    }
    block_exp_out = blk.e_b;
#if EXSIA_BRANCH_COUNTS_ENABLED
    ++cycle_sample.block_exp_commit_count;
#endif
#if EXSIA_STAGE_PROFILE_ENABLED
#if defined(__linux__) && defined(__aarch64__)
    const auto t4 = EXSIA_STAGE_CYCLE_READ();
#else
    const uint64_t t4 = EXSIA_STAGE_CYCLE_READ();
#endif
#if defined(__linux__) && defined(__aarch64__)
    record_stage_cycles(cycle_sample, {t0, t1, t2, t3, t4});
#else
    cycle_sample.p0 = t1 >= t0 ? t1 - t0 : 0;
    cycle_sample.p1 = t2 >= t1 ? t2 - t1 : 0;
    cycle_sample.p2 = t3 >= t2 ? t3 - t2 : 0;
    cycle_sample.p3 = t4 >= t3 ? t4 - t3 : 0;
#endif
#endif
    return true;
}

#if EXSIA_VALIDATION
bool LocalStage::run_reference(Meta &                     meta,
                               ExSIAState &               state,
                               const std::vector<float> & x,
                               size_t                     local_row,
                               size_t                     blk_idx,
                               StripeScratch &            scratch,
                               BlockMask &                block_mask,
                               std::vector<int32_t> &     stripe_q_wide,
                               std::vector<int16_t> &     stripe_block_exp,
                               LocalBlockCycleSample &    cycle_sample) {
    const size_t blk_size = state.B_size;
    const size_t base     = local_row * state.K_padded + blk_idx * blk_size;
    cycle_sample          = LocalBlockCycleSample{};

    GGML_ASSERT(x.size() == blk_size);
    BlockState &           blk               = scratch.block;
    const int16_t          neg_inf           = std::numeric_limits<int16_t>::min();
    bool                   has_second_bucket = false;
    std::vector<int32_t> & q_tmp             = scratch.reference.q_tmp;
    std::vector<int32_t> & q_final           = scratch.reference.q_final;
    __int128_t             S                 = 0;
    __int128_t             SS                = 0;
    size_t                 unmasked_count    = 0;
    bool                   has_int_outlier   = false;
    int16_t                e_pre             = neg_inf;
    int16_t                theta_pre         = neg_inf;
#if EXSIA_STAGE_PROFILE_ENABLED
#if defined(__linux__) && defined(__aarch64__)
    const auto t0 = EXSIA_STAGE_CYCLE_READ();
#else
    const uint64_t t0 = EXSIA_STAGE_CYCLE_READ();
#endif
#endif

    unit_exp_.scan_top2_exp(x, blk);
    has_second_bucket = (blk.e2 != neg_inf);
    block_mask.clear();

    if (has_second_bucket) {
        for (size_t i = 0; i < blk_size; ++i) {
            const size_t col = blk_idx * blk_size + i;
            if (col < state.K_logical && blk.e[i] != neg_inf && blk.e[i] == blk.e1)
                block_mask.set(i);
        }
    }

    e_pre = has_second_bucket ? blk.e2 : blk.e1;
#if EXSIA_STAGE_PROFILE_ENABLED
#if defined(__linux__) && defined(__aarch64__)
    const auto t1 = EXSIA_STAGE_CYCLE_READ();
#else
    const uint64_t t1 = EXSIA_STAGE_CYCLE_READ();
#endif
#endif

    theta_pre = exp_to_theta(e_pre, meta.rho);
    unit_quant_.quantize_block(blk.x, block_mask, theta_pre, q_tmp, S, SS);

    for (size_t i = 0; i < blk_size; ++i) {
        const size_t col = blk_idx * blk_size + i;
        if (col < state.K_logical && !block_mask.is_set(i))
            ++unmasked_count;
    }

#if EXSIA_STAGE_PROFILE_ENABLED
#if defined(__linux__) && defined(__aarch64__)
    const auto t2 = EXSIA_STAGE_CYCLE_READ();
#else
    const uint64_t t2 = EXSIA_STAGE_CYCLE_READ();
#endif
#endif

    for (size_t i = 0; i < blk_size; ++i) {
        const size_t col = blk_idx * blk_size + i;
        if (col >= state.K_logical || block_mask.is_set(i))
            continue;

        if (unit_sigma_.detect_sigma(q_tmp[i], S, SS, unmasked_count)) {
            block_mask.set(i);
            has_int_outlier = true;
        }
    }

#if EXSIA_STAGE_PROFILE_ENABLED
#if defined(__linux__) && defined(__aarch64__)
    const auto t3 = EXSIA_STAGE_CYCLE_READ();
#else
    const uint64_t t3 = EXSIA_STAGE_CYCLE_READ();
#endif
#endif

    if (!has_int_outlier) {
        blk.e_b     = e_pre;
        blk.theta_b = theta_pre;
        std::copy_n(q_tmp.begin(), blk_size, q_final.begin());
        cycle_sample.p3_path = P3Path::BypassNoIntegerOutlier;
    } else {
        unit_exp_.update_block_top2_exp(block_mask, blk);
        blk.e_b     = blk.e1;
        blk.theta_b = exp_to_theta(blk.e_b, meta.rho);

        if (blk.theta_b == theta_pre) {
            std::copy_n(q_tmp.begin(), blk_size, q_final.begin());
            cycle_sample.p3_path = P3Path::BypassSameScale;
        } else {
            unit_quant_.quantize_block(blk.x, blk.theta_b, q_final);
            cycle_sample.p3_path = P3Path::Replay;
        }
    }

    GGML_ASSERT(stripe_q_wide.size() >= base + blk_size);
    for (size_t i = 0; i < blk_size; ++i)
        stripe_q_wide[base + i] = q_final[i];

    const size_t block_exp_idx = local_row * state.blocks_per_row + blk_idx;
    GGML_ASSERT(stripe_block_exp.size() > block_exp_idx);
    stripe_block_exp[block_exp_idx] = blk.e_b;

#if EXSIA_STAGE_PROFILE_ENABLED
#if defined(__linux__) && defined(__aarch64__)
    const auto t4 = EXSIA_STAGE_CYCLE_READ();
#else
    const uint64_t t4 = EXSIA_STAGE_CYCLE_READ();
#endif
#if defined(__linux__) && defined(__aarch64__)
    record_stage_cycles(cycle_sample, {t0, t1, t2, t3, t4});
#else
    cycle_sample.p0 = t1 >= t0 ? t1 - t0 : 0;
    cycle_sample.p1 = t2 >= t1 ? t2 - t1 : 0;
    cycle_sample.p2 = t3 >= t2 ? t3 - t2 : 0;
    cycle_sample.p3 = t4 >= t3 ? t4 - t3 : 0;
#endif
#endif

    return true;
}
#endif

} // namespace ggml::gemmini::quants::act::exsia
