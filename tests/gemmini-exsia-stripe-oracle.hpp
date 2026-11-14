// Included by the ExSIA behavioral test owner inside its anonymous namespace.
// Independent original-source oracle. No production quantizer, shifts, clipping,
// or reference Local implementation participates in the expected values.
int32_t stripe_rn_oracle(double value, int theta) {
    if (!std::isfinite(value) || theta == std::numeric_limits<int16_t>::min())
        return 0;
    const double scaled = std::ldexp(double(value), -theta);
    if (scaled >= 2147483647.0) return INT32_MAX;
    if (scaled <= -2147483648.0) return INT32_MIN;
    const double lower = std::floor(scaled);
    const double fraction = scaled - lower;
    const int64_t integer = int64_t(lower);
    return int32_t(integer + (fraction > 0.5 || (fraction == 0.5 && integer % 2 != 0)));
}

struct StripeOracle {
    std::vector<int16_t> theta, block_exp;
    std::vector<bool> selected, local_selected;
    std::vector<int32_t> dense, residual, preliminary;
};

StripeOracle original_stripe_oracle(const std::vector<float> & source, size_t rows, size_t cols,
                                   int rank, bool direct) {
    constexpr int absent = std::numeric_limits<int16_t>::min();
    constexpr int rho = GGML_GEMMINI_ACTIVATION_BITS - 2;
    constexpr int64_t radix = int64_t{1} << GGML_GEMMINI_ACTIVATION_BITS;
    constexpr int64_t half = radix / 2;
    const size_t blocks = (cols + 31) / 32;
    StripeOracle out;
    out.selected.resize(rows * cols);
    out.block_exp.assign(rows * blocks, absent);
    out.dense.resize(rows * cols);
    out.residual.resize(rows * cols);
    out.preliminary.resize(rows * cols);
    std::vector<int> exponents(rows * cols, absent);
    for (size_t p = 0; p < source.size(); ++p)
        if (source[p] != 0 && std::isfinite(source[p]))
            exponents[p] = std::ilogb(source[p]);
    for (size_t row = 0; row < rows; ++row)
        for (size_t block = 0; block < blocks; ++block) {
            const size_t begin = row * cols + block * 32;
            const size_t end = row * cols + std::min(cols, block * 32 + 32);
            std::vector<int> buckets(exponents.begin() + begin, exponents.begin() + end);
            std::sort(buckets.begin(), buckets.end(), std::greater<int>());
            buckets.erase(std::unique(buckets.begin(), buckets.end()), buckets.end());
            buckets.erase(std::remove(buckets.begin(), buckets.end(), absent), buckets.end());
            const int pre = buckets.empty() ? absent : buckets[buckets.size() > 1 ? 1 : 0];
            __int128 S = 0, SS = 0, N = 0;
            std::vector<int64_t> magnitudes(end - begin);
            for (size_t p = begin; p < end; ++p) {
                out.selected[p] = buckets.size() > 1 && exponents[p] == buckets[0];
                const int64_t q = stripe_rn_oracle(source[p], pre == absent ? absent : pre - rho);
                out.preliminary[p] = out.selected[p] ? 0 : int32_t(q);
                const int64_t a = q < 0 ? -q : q;
                magnitudes[p - begin] = a;
                if (!out.selected[p] && exponents[p] != absent) {
                    ++N; S += a; SS += __int128(a) * a;
                }
            }
            const __int128 V = N * SS - S * S;
            bool added = false;
            int remaining = absent;
            for (size_t p = begin; p < end; ++p) {
                if (out.selected[p] || exponents[p] == absent) continue;
                const __int128 C = N * magnitudes[p - begin] - S;
                if (V > 0 && C > 0 && C * C >
                    __int128(GGML_GEMMINI_EXSIA_SIGMA) * GGML_GEMMINI_EXSIA_SIGMA * V) {
                    out.selected[p] = true;
                    added = true;
                } else remaining = std::max(remaining, exponents[p]);
            }
            out.block_exp[row * blocks + block] = int16_t(added ? remaining : pre);
        }
    out.local_selected = out.selected;
    for (size_t r0 = 0; r0 < rows; r0 += DIM) {
        const size_t r1 = std::min(rows, r0 + DIM);
        std::vector<int16_t> buckets(out.block_exp.begin() + r0 * blocks,
                                     out.block_exp.begin() + r1 * blocks);
        std::sort(buckets.begin(), buckets.end(), std::greater<int16_t>());
        buckets.erase(std::unique(buckets.begin(), buckets.end()), buckets.end());
        buckets.erase(std::remove(buckets.begin(), buckets.end(), absent), buckets.end());
        const int anchor = buckets.empty() ? 0 : buckets[std::min<size_t>(rank - 1, buckets.size() - 1)];
        out.theta.push_back(int16_t(anchor - rho));
        for (size_t row = r0; row < r1; ++row)
            for (size_t col = 0; col < cols; ++col) {
                const size_t p = row * cols + col;
                out.selected[p] = out.selected[p] ||
                    (buckets.size() > 1 && out.block_exp[row * blocks + col / 32] > anchor);
                const int block_exp = out.block_exp[row * blocks + col / 32];
                const int32_t local = stripe_rn_oracle(
                    source[p], block_exp == absent ? absent : block_exp - rho);
                const int64_t u = direct ? stripe_rn_oracle(source[p], anchor - rho)
                    : stripe_rn_oracle(double(local), block_exp == absent ? 0 : anchor - block_exp);
                const int64_t saturated = std::max(-half, std::min(half - 1, u));
                int64_t low = saturated, residual = 0;
                if (out.selected[p] && GGML_GEMMINI_ENABLE_RMD) {
                    low = ((u + half) % radix + radix) % radix - half;
                    residual = u - low;
                    if (residual > INT32_MAX) { low = saturated; residual = u - low; }
                }
                out.dense[p] = int32_t(low);
                out.residual[p] = int32_t(residual);
            }
    }
    return out;
}
