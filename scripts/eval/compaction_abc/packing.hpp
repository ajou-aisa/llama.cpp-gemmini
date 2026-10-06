#pragma once
#include "types.hpp"

namespace abc {
inline Packed pack(const Input &input, const Digits &digits, Variant variant) {
    Packed result;
    result.lanes = digits.lanes;
    result.bits = input.bits;
    result.variant = variant;
    const bool prune_rows = variant != Variant::C;
    const bool compact_k = variant == Variant::A;
    std::vector<uint8_t> row_active(prune_rows ? input.m * digits.lanes : 0);
    std::vector<uint8_t> column_active(compact_k ? input.k : 0);
    for (const Digit &digit : digits.entries) {
        if (prune_rows) row_active[digit.lane * input.m + digit.row] = 1;
        if (compact_k) column_active[digit.k] = 1;
    }
    std::vector<size_t> row_map(row_active.size()), column_map(input.k);
    for (size_t row = 0; row < row_active.size(); ++row) if (row_active[row]) {
        row_map[row] = result.rows.size();
        result.rows.push_back(row);
    }
    for (size_t block = 0; block < input.k / 32; ++block) {
        Run run{static_cast<uint32_t>(block),0,static_cast<uint32_t>(compact_k ? result.columns.size() : block * 32),0};
        for (size_t local = 0; local < 32; ++local) if (!compact_k || column_active[block * 32 + local]) {
            if (compact_k) {
                column_map[block * 32 + local] = result.columns.size();
                result.columns.push_back(block * 32 + local);
            }
            run.mask |= uint32_t{1} << local;
            ++run.count;
        }
        if (run.count) result.runs.push_back(run);
    }
    result.m = prune_rows ? result.rows.size() : input.m * digits.lanes;
    result.k = compact_k ? result.columns.size() : input.k;
    result.a.assign(result.m * result.k, 0);
    for (const Digit &digit : digits.entries)
        result.a[(prune_rows ? row_map[digit.lane * input.m + digit.row] : digit.lane * input.m + digit.row) * result.k +
                 (compact_k ? column_map[digit.k] : digit.k)] = digit.value;
    return result;
}
inline void gather_weights(Packed &packed, const Weight &weight, size_t dim) {
    packed.w.assign(packed.k * weight.n, 0);
    for (const Run &run : packed.runs)
        for (size_t kb = 0; kb < run.count; kb += dim)
            for (size_t cb = 0; cb < weight.n; cb += dim)
                for (size_t col = cb; col < std::min(cb + dim, size_t{weight.n}); ++col)
                    for (size_t k = kb; k < std::min(kb + dim, size_t{run.count}); ++k)
                        packed.w[(run.begin + k) * weight.n + col] = weight.code(packed.column_identity(run.begin + k), col);
}
inline std::vector<int64_t> restore(const Input &input, const Packed &packed,
                                   const std::vector<int32_t> &output) {
    std::vector<int64_t> result(input.m * input.n, 0);
    for (size_t row = 0; row < packed.m; ++row) {
        const size_t identity = packed.row_identity(row);
        const size_t lane = identity / input.m;
        const size_t source = identity % input.m;
        const int64_t radix = int64_t{1} << (input.bits * lane);
        for (size_t col = 0; col < input.n; ++col)
            result[source * input.n + col] += int64_t{output[row * input.n + col]} * radix;
    }
    return result;
}
}
