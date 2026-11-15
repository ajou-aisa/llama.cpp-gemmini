#include "../ggml/src/ggml-gemmini/quants/act/exsia/local.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <random>
#include <type_traits>
#include <vector>

namespace ex = ggml::gemmini::quants::act::exsia;

static_assert(std::is_same_v<decltype(ex::SigmaDetector::SigmaContext::n), int64_t>);
static_assert(std::is_same_v<decltype(ex::SigmaDetector::SigmaContext::S), int64_t>);
static_assert(std::is_same_v<decltype(ex::SigmaDetector::SigmaContext::threshold), int64_t>);

static bool check_sigma(const std::vector<int32_t> & values) {
    int64_t sum = 0, squares = 0;
    for (int32_t q : values) {
        const int64_t magnitude = q < 0 ? -int64_t(q) : q;
        sum += magnitude;
        squares += magnitude * magnitude;
    }
    ex::SigmaDetector detector;
    const auto context = detector.prepare(sum, squares, values.size());
    const __int128_t n = values.size(), s = sum, ss = squares;
    const __int128_t variance = n * ss - s * s;
    const __int128_t sigma = GGML_GEMMINI_EXSIA_SIGMA;
    for (int32_t q : values) {
        const __int128_t magnitude = q < 0 ? -__int128_t(q) : q;
        const __int128_t centered = n * magnitude - s;
        const bool expected = variance > 0 && centered > 0 &&
                              centered * centered > sigma * sigma * variance;
        if (detector.detect(q, context) != expected) {
            std::fprintf(stderr, "sigma mismatch: N=%zu q=%d sigma=%d\n",
                         values.size(), q, GGML_GEMMINI_EXSIA_SIGMA);
            return false;
        }
    }
    return !values.empty() || !detector.detect(0, context);
}

static bool check_local(const std::vector<float> & values, size_t block_size,
                        int16_t rho = ggml::gemmini::config::GGML_GEMMINI_ACTIVATION_RHO) {
    ex::Meta meta;
    meta.rho = rho;
    ex::ExSIAState state;
    state.B_size = state.K_padded = block_size;
    state.K_logical = values.size();
    state.blocks_per_row = 1;
    ex::StripeScratch optimized_scratch, reference_scratch;
    if (!optimized_scratch.prepare(block_size) || !reference_scratch.prepare(block_size))
        return false;
    std::vector<float> padded(block_size, 0);
    std::copy(values.begin(), values.end(), padded.begin());
    reference_scratch.block.x = padded;
    std::vector<uint64_t> optimized_words(ex::BlockMask::word_count(block_size), 0);
    std::vector<uint64_t> reference_words(optimized_words.size(), 0);
    ex::BlockMask optimized_mask(optimized_words.data(), block_size);
    ex::BlockMask reference_mask(reference_words.data(), block_size);
    std::vector<int32_t> optimized(block_size), reference(block_size);
    int16_t optimized_exp = 0;
    std::vector<int16_t> reference_exp(1);
    ex::LocalBlockCycleSample optimized_sample, reference_sample;
    ex::LocalStage stage;
    if (!stage.run_optimized(meta, state, values.data(), values.size(), block_size, 0, 0,
                             optimized_scratch, optimized_mask, optimized.data(), optimized_exp,
                             optimized_sample) ||
        !stage.run_reference(meta, state, padded, 0, 0, reference_scratch, reference_mask,
                             reference, reference_exp, reference_sample))
        return false;
    int16_t remaining_exp = std::numeric_limits<int16_t>::min();
    for (size_t i = 0; i < values.size(); ++i)
        if (!reference_mask.is_set(i))
            remaining_exp = std::max(remaining_exp, reference_scratch.block.e[i]);
    if (optimized_words != reference_words || optimized != reference ||
        optimized_exp != reference_exp[0] ||
        optimized_scratch.block.theta_b != reference_scratch.block.theta_b ||
        optimized_sample.p3_path != reference_sample.p3_path ||
        optimized_sample.final_remaining_exp != remaining_exp) {
        std::fprintf(stderr, "Local/reference mismatch: B=%zu valid=%zu A%d rho=%d sigma=%d\n",
                     block_size, values.size(), GGML_GEMMINI_ACTIVATION_BITS,
                     rho, GGML_GEMMINI_EXSIA_SIGMA);
        return false;
    }
    return true;
}

int main() {
    constexpr int32_t bound = int32_t{1} << (GGML_GEMMINI_ACTIVATION_BITS - 1);
    std::mt19937 random(0x51a64);
    for (size_t n : {0, 1, 2, 4, 5, 6, 31, 32, 63, 64, 127, 128}) {
        if (!check_sigma(std::vector<int32_t>(n, 0)) ||
            !check_sigma(std::vector<int32_t>(n, -bound)))
            return 1;
        std::vector<int32_t> sample(n, 0);
        if (n) {
            // N=5 is exactly the strict 2-sigma boundary; N=6 selects the peak.
            sample.back() = bound;
            if (!check_sigma(sample))
                return 1;
            std::fill(sample.begin(), sample.begin() + n / 2, -bound);
            if (!check_sigma(sample))
                return 1;
        }
        for (size_t trial = 0; trial < 256; ++trial) {
            for (auto & q : sample) {
                q = static_cast<int32_t>(random() % (bound + 1));
                if (random() & 1u)
                    q = -q;
            }
            if (!check_sigma(sample))
                return 1;
        }
    }
    for (size_t block_size : {32, 64, 128}) {
        const float tiny = std::numeric_limits<float>::denorm_min();
        const float inf = std::numeric_limits<float>::infinity();
        const float nan = std::numeric_limits<float>::quiet_NaN();
        const std::vector<std::vector<float>> cases = {
            {0.0f, -0.0f, inf, -inf, nan},
            {256, -2, 2, -3, 3},                 // Nonzero variance without a 2-sigma outlier.
            {256, -8, tiny, tiny, tiny, tiny},   // N=5: exact strict 2-sigma boundary.
            {256, -8, tiny, tiny, tiny, tiny, tiny}, // N=6: rounded zeros still count.
            {256, -8, 1.375f, 1.375f, 1.375f, 1.375f, 1.375f}, // Scale replay.
            {256, 2, -2, 2, -2, 2, -3.75f},    // Selection can retain the same exponent.
            {1, 2, 4, 8, -8, 4, 0, 8},        // A later maximum replaces earlier buckets.
            {8, -8, 4, 8, 2, 4, 1},           // Repeated maxima keep every top-bucket bit.
            {std::nextafter(2.0f, 0.0f), -std::nextafter(2.0f, 0.0f)},
        };
        for (int16_t rho : {ex::Meta{}.rho, int16_t{14}}) {
            for (const auto & input : cases) {
                for (size_t valid : {input.size(), block_size - 1, block_size}) {
                    std::vector<float> values = input;
                    values.resize(valid, 0.0f);
                    if (!check_local(values, block_size, rho))
                        return 1;
                }
            }
        }
        std::vector<float> cross_word(block_size, 1.0f);
        cross_word.front() = 4.0f;
        cross_word.back() = -8.0f;
        if (!check_local(cross_word, block_size))
            return 1;
        for (size_t valid : {size_t{1}, block_size - 1, block_size}) {
            std::vector<float> values(valid, 0);
            if (!check_local(values, block_size))
                return 1;
            // Rounded inliers can reach the width's magnitude bound.
            std::fill(values.begin(), values.end(), std::nextafter(2.0f, 0.0f));
            if (!check_local(values, block_size))
                return 1;
            values.front() = std::numeric_limits<float>::max();
            if (!check_local(values, block_size))
                return 1;
            values.front() = -std::numeric_limits<float>::max();
            if (!check_local(values, block_size))
                return 1;
            std::fill(values.begin(), values.end(), std::numeric_limits<float>::denorm_min());
            if (!check_local(values, block_size))
                return 1;
            for (size_t trial = 0; trial < 128; ++trial) {
                for (float & value : values) {
                    const int exponent = static_cast<int>(random() % 270) - 149;
                    const float fraction = 1.0f + static_cast<float>(random() % 65536) / 65536.0f;
                    value = (random() % 8 == 0) ? 0.0f : std::ldexp(fraction, exponent);
                    if (random() & 1u)
                        value = -value;
                }
                if (!check_local(values, block_size))
                    return 1;
            }
        }
    }
    std::printf("PASS: int64 sigma and Local match int128 reference; A%d sigma=%d B=32/64/128\n",
                GGML_GEMMINI_ACTIVATION_BITS, GGML_GEMMINI_EXSIA_SIGMA);
    return 0;
}
