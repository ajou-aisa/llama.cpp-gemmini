#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

struct replay_work {
    const int8_t * a;
    const int8_t * b;
    const int64_t * rows;
    const int64_t * shifts;
    const int64_t * regions;
    int m, k, n, region_count;
};

template<class RawDot>
void execute_replay_batch(const replay_work * work, int count, int rows, int columns,
                          int direct, int64_t * out, uint64_t * stats, RawDot dot) {
    auto require = [](bool ok, const char * message) {
        if (!ok) throw std::runtime_error(message);
    };
    using clock = std::chrono::steady_clock;
    const auto begin = clock::now();
    require(count >= 0 && rows > 0 && columns > 0 && out && stats && (work || !count), "Invalid batch");
    require(direct == 0 || direct == 1, "Invalid batch mode");
    std::fill(stats, stats+8, 0);
    size_t scratch_size = 0;
    for (int t = 0; t < count; ++t) {
        const auto & w = work[t];
        require(w.m > 0 && w.k > 0 && w.k <= 32 && w.n == columns && w.region_count >= 0 &&
                w.a && w.b && w.rows && w.shifts && (w.regions || !w.region_count), "Invalid work item");
        for (size_t i = 0; i < size_t(w.m)*w.k; ++i)
            require(w.a[i] >= -8 && w.a[i] <= 7, "Activation is not a signed nibble");
        for (size_t i = 0; i < size_t(w.k)*w.n; ++i)
            require(w.b[i] >= -8 && w.b[i] <= 7, "Weight is not a signed nibble");
        for (int i = 0; i < w.m; ++i) require(w.rows[i] >= 0 && w.rows[i] < rows, "Invalid original row");
        for (int j = 0; j < w.n; ++j) require(w.shifts[j] >= 0, "Negative relative shift");
        for (int r = 0; r < w.region_count; ++r) {
            const auto * region = w.regions + size_t(r)*4;
            require(region[0] >= 0 && region[0] < region[1] && region[1] <= w.m &&
                    region[2] >= 0 && region[2] < region[3] && region[3] <= w.n, "Invalid retained region");
            scratch_size = std::max(scratch_size, size_t(region[1]-region[0])*size_t(region[3]-region[2]));
        }
        if (!direct) scratch_size = std::max(scratch_size, size_t(w.m)*w.n);
    }
    std::vector<int64_t> acc(size_t(rows)*columns, 0);
    std::vector<int32_t> scratch(scratch_size);
    stats[0] = std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now()-begin).count();
    stats[3] = acc.capacity()*sizeof(int64_t) + scratch.capacity()*sizeof(int32_t);
    stats[7] = count;
    for (int t = 0; t < count; ++t) {
        const auto & w = work[t];
        auto merge = [&](const int32_t * part, int first_row, int last_row, int first_col, int last_col, int stride) {
            const auto started = clock::now();
            for (int i = first_row; i < last_row; ++i) {
                auto * target = acc.data() + size_t(w.rows[i])*columns;
                const auto * source = part + size_t(i-first_row)*stride;
                for (int j = first_col; j < last_col; ++j) {
                    const int32_t raw = source[j-first_col];
                    if (!raw) continue;
                    require(w.shifts[j] <= 63, "Batch scaled partial exceeds int64");
                    const __int128 value = __int128(raw)*(__int128(1) << w.shifts[j]) + target[j];
                    require(value >= std::numeric_limits<int64_t>::min() &&
                            value <= std::numeric_limits<int64_t>::max(), "Batch accumulator exceeds int64");
                    target[j] = int64_t(value);
                }
            }
            stats[2] += std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now()-started).count();
        };
        if (!direct) {
            std::fill(scratch.begin(), scratch.begin()+size_t(w.m)*w.n, 0);
            stats[5] += size_t(w.m)*w.n*sizeof(int32_t);
        }
        for (int r = 0; r < w.region_count; ++r) {
            const auto * region = w.regions + size_t(r)*4;
            const int r0 = int(region[0]), r1 = int(region[1]), c0 = int(region[2]), c1 = int(region[3]);
            auto * part = scratch.data() + (direct ? 0 : size_t(r0)*w.n+c0);
            const int stride = direct ? c1-c0 : w.n;
            const auto started = clock::now();
            dot(w.a+size_t(r0)*w.k, w.b+c0, part, r1-r0, w.k, c1-c0, w.k, w.n, stride);
            stats[1] += std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now()-started).count();
            ++stats[4];
            stats[6] += size_t(r1-r0)*(c1-c0)*sizeof(int32_t);
            if (direct) merge(part, r0, r1, c0, c1, stride);
        }
        if (!direct) merge(scratch.data(), 0, w.m, 0, w.n, w.n);
    }
    // A rejected provider or overflowing later partial must not publish a prefix.
    std::copy(acc.begin(), acc.end(), out);
}
