#include "../ggml/src/ggml-cuda/cpu-exact/ggml-cuda-hp1-scu.h"
#include "../ggml/src/ggml-gemmini/ggml-gemmini-args.h"
#include "../ggml/src/ggml-gemmini/quants/common/hp1_scu.hpp"

#include <algorithm>
#include <array>
#include <climits>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace scu = ggml::gemmini::quants::hp1;
using status = ggml_cuda_residual_status;
using event = ggml_cuda_residual_event;

static void require(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "HP1_SCU FAIL: %s (CUDA: %s)\n", message, ggml_cuda_hp1_scu_error());
        throw std::runtime_error(message);
    }
}

struct weights {
    unsigned bits;
    size_t columns, k;
    std::vector<block_q4_hp1> q4;
    std::vector<block_q8_hp1> q8;

    weights(unsigned width, size_t cols, size_t length)
        : bits(width), columns(cols), k(length), q4(cols*length/32), q8(cols*length/32) {
        for (size_t b = 0; b < q4.size(); ++b) {
            std::memset(q4[b].qs, 0x88, sizeof(q4[b].qs));
            q4[b].channel_scale = q8[b].channel_scale = 0.125f;
        }
    }
    const void * data() const { return bits == 4 ? static_cast<const void *>(q4.data()) : q8.data(); }
    size_t bytes() const { return q4.size()*(bits == 4 ? sizeof(q4[0]) : sizeof(q8[0])); }
    void exponent(size_t col, size_t block, int16_t value) {
        q4[col*(k/32)+block].m = q8[col*(k/32)+block].m = value;
    }
    uint32_t carrier(size_t col, size_t block) const {
        uint32_t result = 0;
        require(scu::encode_carrier(q8[col*(k/32)+block].m, result), "invalid oracle carrier");
        return result;
    }
    void code(size_t col, size_t pos, int value) {
        const size_t b = col*(k/32)+pos/32, t = pos%32;
        q8[b].qs[t] = int8_t(value);
        if (bits == 4) {
            require(value >= -8 && value <= 7, "test weight exceeds q4");
            const unsigned shift = t < 16 ? 0 : 4;
            auto & packed = q4[b].qs[t%16];
            packed = uint8_t((packed & ~(15u << shift)) | (unsigned(value+8) << shift));
        }
    }
    int code(size_t col, size_t pos) const {
        const size_t b = col*(k/32)+pos/32, t = pos%32;
        return bits == 4 ? int((q4[b].qs[t%16] >> (t < 16 ? 0 : 4)) & 15)-8 : q8[b].qs[t];
    }
};

struct counts { uint64_t fragments = 0, shifts = 0, adds = 0; };

static int32_t align_and_add(int32_t acc, int32_t raw, uint32_t carrier, counts & observed) {
    ++observed.fragments;
    if (raw && carrier != scu::zero_carrier) {
        if (carrier >= 32) {
            ++observed.shifts;
        } else {
            const int64_t exact = int64_t(raw)*(int64_t{1} << carrier);
            observed.shifts += exact < INT32_MIN || exact > INT32_MAX;
        }
    }
    const int32_t aligned = scu::apply_validated(raw, carrier);
    const int64_t exact = int64_t(acc)+aligned;
    observed.adds += exact < INT32_MIN || exact > INT32_MAX;
    return scu::accumulate(acc, aligned);
}

static std::vector<int32_t> main_oracle(const weights & w, const std::vector<int32_t> & a, size_t rows,
                                      counts & observed) {
    std::vector<int32_t> result(rows*w.columns);
    for (size_t r = 0; r < rows; ++r) for (size_t c = 0; c < w.columns; ++c) {
        int32_t acc = 0;
        for (size_t b = 0; b < w.k/32; ++b) {
            int64_t raw = 0;
            for (size_t t = b*32; t < (b+1)*32; ++t) raw += int64_t(a[r*w.k+t])*w.code(c,t);
            require(raw >= INT32_MIN && raw <= INT32_MAX, "test dot exceeds INT32");
            acc = align_and_add(acc, int32_t(raw), w.carrier(c,b), observed);
        }
        result[r*w.columns+c] = acc;
    }
    return result;
}

// Independent signed-remainder decomposition, including the positive carry limb
// needed by INT32_MAX. Each digit is in [-2^(n-1), 2^(n-1)-1].
static std::array<int32_t,9> digits(int32_t value, unsigned bits) {
    std::array<int32_t,9> result{};
    const int64_t radix = int64_t{1} << bits, half = radix/2;
    int64_t remaining = value;
    __int128 reconstructed = 0;
    size_t lane = 0;
    while (remaining) {
        require(lane < result.size(), "balanced decomposition exceeds lane count");
        int64_t digit = remaining%radix;
        if (digit >= half) digit -= radix;
        else if (digit < -half) digit += radix;
        require(digit >= -half && digit < half, "balanced digit exceeds operand width");
        result[lane] = int32_t(digit);
        reconstructed += __int128(digit)*(__int128{1} << (bits*lane));
        remaining = (remaining-digit)/radix;
        ++lane;
    }
    require(reconstructed == value, "balanced decomposition changed input");
    return result;
}

static bool residual_oracle(const weights & w, const std::vector<uint32_t> & offsets,
                            const std::vector<event> & events, std::vector<int64_t> & result, counts & observed) {
    const size_t rows = offsets.size()-1;
    result.assign(rows*w.columns,0);
    bool fits = true;
    for (size_t r = 0; r < rows; ++r) for (size_t c = 0; c < w.columns; ++c) {
        // Hardware accumulates every K block into the same lane accumulator.
        std::array<int32_t,9> acc{};
        for (size_t b = 0; b < w.k/32; ++b) {
            std::array<int32_t,9> raw{};
            std::array<bool,9> used{};
            for (size_t e = offsets[r]; e < offsets[r+1]; ++e) if (events[e].k/32 == b) {
                const auto split = digits(events[e].value, w.bits);
                for (size_t lane = 0; lane < split.size(); ++lane) {
                    raw[lane] += split[lane]*w.code(c,events[e].k);
                    used[lane] = used[lane] || split[lane] != 0;
                }
            }
            for (size_t lane = 0; lane < acc.size(); ++lane) if (used[lane])
                acc[lane] = align_and_add(acc[lane],raw[lane],w.carrier(c,b),observed);
        }
        __int128 reconstructed = 0;
        for (size_t lane = 0; lane <= 32/w.bits; ++lane)
            reconstructed += __int128(acc[lane])*(__int128{1} << (lane*w.bits));
        if (reconstructed < INT64_MIN || reconstructed > INT64_MAX) fits = false;
        else result[r*w.columns+c] = int64_t(reconstructed);
    }
    return fits;
}

static void check_main(const weights & w, const std::vector<int32_t> & a, size_t rows) {
    counts observed;
    const auto expected = main_oracle(w,a,rows,observed);
    std::vector<int32_t> actual(expected.size(),-777);
    const auto before = ggml_cuda_hp1_scu_stats();
    require(ggml_cuda_hp1_scu_main(a.data(),w.data(),w.bits,rows,w.columns,w.k,actual.data()), "main launch");
    const auto after = ggml_cuda_hp1_scu_stats();
    require(actual == expected, "main integer mismatch");
    require(after.main_launches-before.main_launches == 1 &&
            after.main_fragments-before.main_fragments == observed.fragments &&
            after.main_shift_saturations-before.main_shift_saturations == observed.shifts &&
            after.main_acc_saturations-before.main_acc_saturations == observed.adds, "main saturation counters");
}

static void check_residual(const weights & w, const std::vector<uint32_t> & offsets,
                           const std::vector<event> & events) {
    counts observed;
    std::vector<int64_t> expected;
    const bool fits = residual_oracle(w,offsets,events,expected,observed);
    std::vector<int64_t> actual(expected.size(),-777);
    const auto before = ggml_cuda_hp1_scu_stats();
    const auto outcome = ggml_cuda_hp1_scu_residual(w.data(),w.bytes(),w.bits,offsets.size()-1,w.columns,w.k,
            offsets.data(),events.data(),events.size(),actual.data());
    const auto after = ggml_cuda_hp1_scu_stats();
    require(outcome == (fits ? status::success : status::overflow), "residual status");
    require(actual == (fits ? expected : std::vector<int64_t>(expected.size(),-777)), "residual integer/publish mismatch");
    if (fits) {
        require(after.residual_launches-before.residual_launches == 1 &&
                after.residual_fragments-before.residual_fragments == observed.fragments &&
                after.residual_shift_saturations-before.residual_shift_saturations == observed.shifts &&
                after.residual_acc_saturations-before.residual_acc_saturations == observed.adds, "residual saturation counters");
    } else {
        require(after.residual_launches == before.residual_launches, "failed residual published success counter");
    }
}

static void main_fixtures(unsigned bits) {
    const int limit = 1 << (bits-1);
    constexpr int16_t exponents[] = {0,31,32,32767,INT16_MIN};
    weights boundary(bits,5,32);
    std::vector<int32_t> signs(64);
    signs[0] = 1; signs[32] = -1;
    for (size_t c = 0; c < 5; ++c) {
        boundary.exponent(c,0,exponents[c]);
        boundary.code(c,0,1);
    }
    check_main(boundary,signs,2);

    weights mixed(bits,19,160);
    std::vector<int32_t> a(7*mixed.k);
    for (size_t i = 0; i < a.size(); ++i) a[i] = int((i*53)%(2*limit))-limit;
    for (size_t c = 0; c < mixed.columns; ++c) {
        for (size_t b = 0; b < mixed.k/32; ++b) mixed.exponent(c,b,exponents[(c+b)%5]);
        for (size_t t = 0; t < mixed.k; ++t) mixed.code(c,t,int((c*37+t*19)%(2*limit))-limit);
    }
    check_main(mixed,a,7);

    weights cancellation(bits,1,96);
    std::vector<int32_t> ones(96,1);
    for (size_t b = 0; b < 3; ++b) {
        cancellation.exponent(0,b,32);
        cancellation.code(0,b*32,b == 2 ? -1 : 1);
    }
    counts observed;
    require(main_oracle(cancellation,ones,1,observed)[0] == -1 && observed.adds == 1,
            "cross-block main cancellation fixture");
    check_main(cancellation,ones,1);

    const auto reject = [&](const weights & w, const std::vector<int32_t> & input, unsigned width, size_t k) {
        int32_t output = -777;
        require(!ggml_cuda_hp1_scu_main(input.data(),w.data(),width,1,1,k,&output) && output == -777,
                "invalid main input changed output");
    };
    weights invalid(bits,1,32);
    std::vector<int32_t> input(32,0);
    input[0] = limit; reject(invalid,input,bits,32);
    input[0] = -limit-1; reject(invalid,input,bits,32);
    input[0] = 0;
    invalid.exponent(0,0,-1); reject(invalid,input,bits,32);
    invalid.exponent(0,0,0);
    invalid.q4[0].padding[0] = invalid.q8[0].padding[0] = 1;
    reject(invalid,input,bits,32);
    invalid.q4[0].padding[0] = invalid.q8[0].padding[0] = 0;
    invalid.q4[0].channel_scale = invalid.q8[0].channel_scale = std::numeric_limits<float>::infinity();
    reject(invalid,input,bits,32);
    invalid.q4[0].channel_scale = invalid.q8[0].channel_scale = 0.125f;
    reject(invalid,input,3,32);
    reject(invalid,input,bits,31);
    std::printf("HP1_SCU main bits=%u bounds carriers tails saturation cross_block_cancellation PASS\n",bits);
}

static void residual_fixtures(unsigned bits) {
    const int limit = 1 << (bits-1), radix = 1 << bits;
    constexpr int16_t exponents[] = {0,31,32,32767,INT16_MIN};
    weights mixed(bits,19,160);
    for (size_t c = 0; c < mixed.columns; ++c) {
        for (size_t b = 0; b < mixed.k/32; ++b) mixed.exponent(c,b,exponents[(c+b)%5]);
        for (size_t t = 0; t < mixed.k; ++t) mixed.code(c,t,int((c*37+t*19)%(2*limit))-limit);
    }
    check_residual(mixed,{0,6,6,9},{{0,17},{1,-39},{31,10000},{32,-100000},{95,0},{159,71},
            {32,radix},{64,-radix},{128,limit}});
    check_residual(mixed,{0,0},{});

    weights extremes(bits,3,96);
    for (size_t c = 0; c < extremes.columns; ++c) for (size_t t = 0; t < extremes.k; ++t)
        extremes.code(c,t,t%2 ? -limit : limit-1);
    check_residual(extremes,{0,4},{{0,INT32_MIN},{31,INT32_MAX},{32,-limit},{95,limit}});

    weights cancellation(bits,1,96);
    for (size_t b = 0; b < 3; ++b) {
        cancellation.exponent(0,b,32);
        cancellation.code(0,b*32,b == 2 ? -1 : 1);
    }
    const std::vector<event> same_lane = {{0,radix+1},{32,radix+1},{64,radix+1}};
    counts observed;
    std::vector<int64_t> expected;
    require(residual_oracle(cancellation,{0,3},same_lane,expected,observed) && expected[0] == -(radix+1) &&
            observed.adds == 2, "cross-block residual lane cancellation fixture");
    check_residual(cancellation,{0,3},same_lane);

    // These four events leave only a +1 digit at radix place 2^32.
    // Its negative aligned value reconstructs to INT64_MIN exactly.
    weights boundary(bits,1,64);
    for (size_t b = 0; b < 2; ++b) boundary.exponent(0,b,32);
    for (size_t t : {size_t(0),size_t(1),size_t(2),size_t(3),size_t(32)}) boundary.code(0,t,-1);
    std::vector<event> highest = {{0,INT32_MAX},{1,1<<30},{2,1<<30},{3,1}};
    counts edge_counts;
    require(residual_oracle(boundary,{0,4},highest,expected,edge_counts) && expected[0] == INT64_MIN,
            "INT64_MIN reconstruction fixture");
    check_residual(boundary,{0,4},highest);
    for (size_t t : {size_t(0),size_t(1),size_t(2),size_t(3),size_t(32)}) boundary.code(0,t,1);
    check_residual(boundary,{0,4},highest);
    highest.push_back({32,radix});
    counts overflow_counts;
    require(!residual_oracle(boundary,{0,5},highest,expected,overflow_counts), "INT64 overflow fixture");
    check_residual(boundary,{0,5},highest);

    const auto reject = [&](const weights & w, const std::vector<uint32_t> & offsets,
                            const std::vector<event> & events, size_t bytes) {
        int64_t output = -777;
        require(ggml_cuda_hp1_scu_residual(w.data(),bytes,bits,1,1,w.k,offsets.data(),events.data(),events.size(),
                &output) == status::invalid_input && output == -777, "invalid residual input changed output");
    };
    weights invalid(bits,1,64);
    reject(invalid,{1,1},{{0,1}},invalid.bytes());
    reject(invalid,{0,2},{{0,1}},invalid.bytes());
    reject(invalid,{0,2},{{32,1},{0,1}},invalid.bytes());
    reject(invalid,{0,2},{{0,1},{0,1}},invalid.bytes());
    reject(invalid,{0,1},{{64,1}},invalid.bytes());
    reject(invalid,{0,1},{{0,1}},invalid.bytes()-1);
    invalid.exponent(0,0,-1);
    reject(invalid,{0,1},{{0,1}},invalid.bytes());
    std::printf("HP1_SCU residual bits=%u balanced_limbs cross_block_lane_saturation INT64 no_partial_publish PASS\n",bits);
}

int main() {
    try {
        for (unsigned bits : {4u,8u}) {
            main_fixtures(bits);
            residual_fixtures(bits);
        }
    } catch (const std::exception &) {
        return 1;
    }
    return 0;
}
