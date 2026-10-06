#pragma once
#include "packing.hpp"
#include "quants/common/hp1_scu.hpp"

namespace abc {
struct Check { uint64_t mismatches = 0, saturations = 0; int64_t max_error = 0; };
inline uint64_t all_column_bound(const Input &input, const Digits &digits, const Weight &weight) {
    std::vector<uint64_t> bounds(input.m * digits.lanes,0);
    for (const Digit &digit : digits.entries)
        bounds[digit.lane * input.m + digit.row] += static_cast<uint64_t>(std::abs(int{digit.value})) * weight.block_bounds[digit.k/32];
    return *std::max_element(bounds.begin(),bounds.end());
}
inline Check numerical_check(const Input &input, const Packed &packed, const Weight &weight) {
    namespace hp1 = ggml::gemmini::quants::hp1;
    const size_t columns = std::min(uint64_t{32}, input.n);
    std::vector<int64_t> expected(input.m * columns, 0), actual(input.m * columns, 0);
    for (const Event &event : input.events)
        for (size_t c = 0; c < columns; ++c) {
            const size_t col = c * input.n / columns;
            const int16_t exponent = weight.exponent(event.k / 32, col);
            if (exponent == INT16_MIN) continue;
            if (exponent < 0 || exponent > 31) throw std::runtime_error("reference exponent out of int64 range");
            expected[event.row * columns + c] += int64_t{event.residual} * weight.code(event.k, col) * (int64_t{1} << exponent);
        }
    Check check;
    for (size_t row = 0; row < packed.m; ++row) {
        std::vector<int32_t> accumulator(columns, 0);
        for (const Run &run : packed.runs)
            for (size_t base = 0; base < run.count; base += packed.array_dim) {
                std::vector<int32_t> partial(columns, 0);
                for (size_t k = base; k < std::min(base + packed.array_dim, size_t{run.count}); ++k) {
                    const int digit = packed.a[row * packed.k + run.begin + k];
                    if (!digit) continue;
                    for (size_t c = 0; c < columns; ++c)
                        partial[c] += digit * packed.w[(run.begin + k) * input.n + c * input.n / columns];
                }
                for (size_t c = 0; c < columns; ++c) {
                    const int16_t exponent = weight.exponent(run.block, c * input.n / columns);
                    uint32_t carrier = 0;
                    if (!hp1::encode_carrier(exponent, carrier)) throw std::runtime_error("invalid HP1 exponent");
                    const int32_t scaled = hp1::apply_validated(partial[c], carrier);
                    if (carrier != hp1::zero_carrier && partial[c] &&
                        (carrier >= 32 || int64_t{partial[c]} * (int64_t{1} << carrier) != scaled)) ++check.saturations;
                    const int32_t sum = hp1::accumulate(accumulator[c],scaled);
                    if (int64_t{accumulator[c]} + scaled != sum) ++check.saturations;
                    accumulator[c] = sum;
                }
            }
        const size_t identity = packed.row_identity(row);
        const size_t lane = identity / input.m, source_row = identity % input.m;
        const int64_t place = int64_t{1} << (input.bits * lane);
        for (size_t c = 0; c < columns; ++c) actual[source_row * columns + c] += int64_t{accumulator[c]} * place;
    }
    for (size_t i = 0; i < expected.size(); ++i) {
        if (expected[i] != actual[i]) ++check.mismatches;
        check.max_error = std::max(check.max_error, std::abs(expected[i] - actual[i]));
    }
    return check;
}
inline void self_test() {
    Input input{4,1,1,32,0,0,1,"fixture",{{0,0,1},{0,1,1},{0,16,-1},{0,17,-1}}};
    Weight weight{4,1,32,std::vector<uint8_t>(24,0),{}};
    std::fill(weight.bytes.begin(),weight.bytes.begin()+16,0xff);
    const int16_t exponent = 28;
    std::memcpy(weight.bytes.data()+16,&exponent,2);
    const auto digits = decompose(input);
    for (Variant variant : {Variant::A,Variant::B,Variant::C}) {
        auto p = pack(input,digits,variant);
        gather_weights(p,weight,16);
        const Check check = numerical_check(input,p,weight);
        const bool expected_difference = variant != Variant::A;
        if ((check.mismatches != 0) != expected_difference || (expected_difference && check.max_error != 1))
            throw std::runtime_error("saturation boundary was not detected");
    }
    input.events = {{0,0,INT32_MIN},{0,1,INT32_MAX}};
    const auto extreme = decompose(input);
    for (const Event &event : input.events) {
        int64_t reconstructed = 0;
        for (const Digit &digit : extreme.entries) if (digit.k == event.k)
            reconstructed += int64_t{digit.value} * (int64_t{1} << (input.bits * digit.lane));
        if (reconstructed != event.residual) throw std::runtime_error("balanced carry reconstruction failed");
    }
}
}
