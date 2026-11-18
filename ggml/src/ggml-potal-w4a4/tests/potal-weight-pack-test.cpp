// Differential checked W4 packing against the scalar algorithm from 43bc313.
// Compile weight_reader.cpp and this file with GGML_GEMMINI_TESTING for actual
// address-resolution counts. No hardware, model files, or timing measurements.
#include "quants/weight/hp1/hp1.hpp"
#include "ggml-gemmini-args.h"
#include "quants/common/hp1_scu.hpp"
#include "ggml.h"

#include <gemmini_params.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

namespace hp = ggml::potal::quants::weight::hp1;
namespace wr = ggml::gemmini::quants::wreader;
namespace route = ggml::gemmini::quants::wroute;
using RS = wr::WeightReaderStatus;

namespace {

uint32_t bits(float v) {
    uint32_t u;
    std::memcpy(&u, &v, sizeof(u));
    return u;
}

// Retain the old checked metadata and K-then-N scalar read/write order.
// Deliberately use neither the tile reader nor the packed-pair helper.
hp::status scalar(const ggml_gemmini_args_t & a, int8_t * B, size_t ldb,
                  uint32_t * carriers, size_t stride, float * scales,
                  hp::stats * out, RS * reader) {
    if (!B || !carriers || !scales || !out || ldb < a.J || stride < a.J || (ldb & 1) ||
        a.weight_format != ggml_gemmini_args_t::im2p_weight_format_t::q4_hp1) {
        return hp::status::invalid;
    }
    const auto p = route::resolve_weight_route_plan(a, route::WeightScaleInfoMode::ResidualHp1Scu);
    if (!p.valid) {
        if (p.hp1_carriers && p.status == route::WeightRouteStatus::InvalidMetadata) {
            const RS rs = wr::validate(a, p);
            if (rs != RS::Success) {
                if (reader) *reader = rs;
                return hp::status::metadata;
            }
        }
        return hp::status::route;
    }
    if (!p.hp1_carriers || p.scales.block_size != 32 || a.K % hp::block_k != 0) return hp::status::route;
    if (reader) *reader = RS::Success;
    hp::stats s;
    bool have_m = false, have_scale = false;
    for (size_t n = 0; n < a.J; ++n) {
        const auto cs = wr::read_column_scale_validated(a, p, n, 0);
        if (!cs.ok()) {
            if (reader) *reader = cs.status;
            return hp::status::metadata;
        }
        const float scale = cs.column_scale;
        if (bits(scale) & 0x80000000u) return hp::status::channel_scale;
        const bool zero = bits(scale) == 0;
        for (size_t b = 0; b < a.native_blocks_per_row; b += hp::block_k / 32) {
            const auto c = wr::read_hp1_carrier_validated(a, p, n, b);
            if (!c.ok()) {
                if (reader) *reader = c.status;
                return hp::status::metadata;
            }
            if (bits(c.column_scale) != bits(scale)) return hp::status::scale_mismatch;
            if constexpr (hp::block_k == 64) {
                const auto other = wr::read_hp1_carrier_validated(a, p, n, b + 1);
                if (!other.ok()) { if (reader) *reader = other.status; return hp::status::metadata; }
                if (other.carrier != c.carrier || bits(other.column_scale) != bits(scale)) return hp::status::pair_mismatch;
            }
            if (c.carrier == ggml::gemmini::quants::hp1::zero_carrier) {
                ++s.zero_blocks;
            } else {
                if (zero) return hp::status::channel_scale;
                const int32_t m = static_cast<int32_t>(c.carrier);
                s.min_m = have_m ? std::min(s.min_m, m) : m;
                s.max_m = have_m ? std::max(s.max_m, m) : m;
                s.sum_m += m;
                s.shift_blocks += m > 0 ? 1 : 0;
                have_m = true;
            }
            carriers[(b / (hp::block_k / 32)) * stride + n] = c.carrier;
        }
        s.blocks += a.native_blocks_per_row / (hp::block_k / 32);
        scales[n] = scale;
        if (zero) {
            ++s.zero_rows;
        } else {
            s.min_channel_scale = have_scale ? std::min(s.min_channel_scale, scale) : scale;
            s.max_channel_scale = have_scale ? std::max(s.max_channel_scale, scale) : scale;
            have_scale = true;
        }
    }
    for (size_t k = 0; k < a.K; ++k) {
        auto * dst = reinterpret_cast<uint8_t *>(B) + k * (ldb / 2);
        std::memset(dst, 0, ldb / 2);
        for (size_t n = 0; n < a.J; ++n) {
            const auto code = wr::read_code_validated(a, p, n, k);
            if (!code.ok()) {
                if (reader) *reader = code.status;
                return hp::status::metadata;
            }
            const unsigned shift = 4 * (n % 2);
            dst[n / 2] = static_cast<uint8_t>((dst[n / 2] & ~(15u << shift)) |
                                             ((static_cast<uint32_t>(code.value) & 15u) << shift));
        }
    }
    *out = s;
    return hp::status::ok;
}

struct output {
    // Prefix/suffix guards and all row gaps start poisoned, including stats.
    std::vector<int8_t> B;
    std::vector<uint32_t> carriers;
    std::vector<float> scales;
    hp::stats stats;
    RS reader = RS::UnsupportedFormat;

    output(size_t K, size_t N, size_t ldb, size_t stride) :
        B(16 + K * (ldb / 2), 0x5a),
        carriers(4 + ((K + 31) / 32) * stride, 0xa5a5a5a5u),
        scales(4 + N, -123.0f) {
        stats.blocks = stats.zero_blocks = stats.shift_blocks = stats.zero_rows = 123;
        stats.min_m = stats.max_m = 123;
        stats.sum_m = stats.min_channel_scale = stats.max_channel_scale = 123;
    }
};

bool same_stats(const hp::stats & a, const hp::stats & b) {
    return a.blocks == b.blocks && a.zero_blocks == b.zero_blocks &&
           a.shift_blocks == b.shift_blocks && a.zero_rows == b.zero_rows &&
           a.min_m == b.min_m && a.max_m == b.max_m && a.sum_m == b.sum_m &&
           bits(a.min_channel_scale) == bits(b.min_channel_scale) &&
           bits(a.max_channel_scale) == bits(b.max_channel_scale);
}

ggml_gemmini_args_t args_for(const std::vector<block_q4_hp1> & blocks, size_t N, size_t K) {
    ggml_gemmini_args_t a;
    a.weight_format = ggml_gemmini_args_t::im2p_weight_format_t::q4_hp1;
    a.block_size_k = 32;
    a.J = N;
    a.K = K;
    a.q4_hp1_blocks = blocks.data();
    a.native_blocks_per_row = K / 32;
    a.native_block_count = blocks.size();
    a.native_weight_bytes = blocks.size() * sizeof(block_q4_hp1);
    return a;
}

std::vector<block_q4_hp1> fixture(size_t N, size_t K) {
    std::vector<block_q4_hp1> blocks(N * (K / 32));
    for (size_t n = 0; n < N; ++n) {
        for (size_t b = 0; b < K / 32; ++b) {
            auto & q = blocks[n * (K / 32) + b];
            q.m = (n % 7 == 3 || (n + b / (hp::block_k / 32)) % 5 == 2) ? INT16_MIN :
                  static_cast<int16_t>((n + b / (hp::block_k / 32)) % 3 == 0 ? 32767 : (n + b / (hp::block_k / 32)) % 33);
            q.channel_scale = n % 7 == 3 ? 0.0f : 0.125f * (1 + n % 9);
            // Every nibble, including -8, both half-blocks, and distinct columns.
            // Even zero-carrier blocks retain nonzero payload: packing is not math.
            for (size_t i = 0; i < 16; ++i) {
                q.qs[i] = static_cast<uint8_t>(((i + n + 3 * b) % 16) |
                                             (((15 - i + 5 * n + b) % 16) << 4));
            }
        }
    }
    return blocks;
}

int cases = 0, failures = 0;

// null_output: 1 B, 2 carriers, 3 scales, 4 stats, 5 reader (optional).
void compare(const char * name, const ggml_gemmini_args_t & a, size_t ldb, size_t stride,
             hp::status want, int null_output = 0) {
    output ref(a.K, a.J, ldb, stride), got = ref;
    auto invoke = [&](auto fn, output & o) {
        return fn(a, null_output == 1 ? nullptr : o.B.data() + 8, ldb,
                  null_output == 2 ? nullptr : o.carriers.data() + 2, stride,
                  null_output == 3 ? nullptr : o.scales.data() + 2,
                  null_output == 4 ? nullptr : &o.stats,
                  null_output == 5 ? nullptr : &o.reader);
    };
#if defined(GGML_GEMMINI_TESTING)
    wr::test_reset_weight_reader_counters();
#endif
    const auto rs = invoke(scalar, ref);
#if defined(GGML_GEMMINI_TESTING)
    const size_t scalar_addresses = wr::test_weight_reader_code_address_resolutions();
    const size_t scalar_validations = wr::test_weight_reader_storage_validations();
    wr::test_reset_weight_reader_counters();
#endif
    const auto gs = invoke(hp::unpack, got);
    bool ok = gs == rs && gs == want && got.reader == ref.reader &&
              got.B == ref.B && got.carriers == ref.carriers &&
              std::memcmp(got.scales.data(), ref.scales.data(), got.scales.size() * sizeof(float)) == 0 &&
              same_stats(got.stats, ref.stats);
    if (gs == hp::status::ok) {
        // Independent source-byte decode and destination padding/guard oracle.
        for (size_t k = 0; k < a.K; ++k) {
            const auto * row = reinterpret_cast<const uint8_t *>(got.B.data() + 8) + k * (ldb / 2);
            for (size_t n = 0; n < ldb; ++n) {
                int expected = 0;
                if (n < a.J) {
                    const auto & block = a.q4_hp1_blocks[n * a.native_blocks_per_row + k / 32];
                    const unsigned u = (block.qs[k % 16] >> (k % 32 < 16 ? 0 : 4)) & 15;
                    expected = static_cast<int>(u) - 8;
                }
                const unsigned u = (row[n / 2] >> (4 * (n % 2))) & 15;
                const int actual = u < 8 ? static_cast<int>(u) : static_cast<int>(u) - 16;
                ok = ok && actual == expected;
            }
        }
        for (size_t i = 0; i < 8; ++i) {
            ok = ok && got.B[i] == 0x5a && got.B[got.B.size() - 1 - i] == 0x5a;
        }
        for (size_t b = 0; b < a.native_blocks_per_row; ++b) {
            for (size_t n = a.J; n < stride; ++n) {
                ok = ok && got.carriers[2 + b * stride + n] == 0xa5a5a5a5u;
            }
        }
    } else {
        ok = ok && std::all_of(got.B.begin(), got.B.end(), [](int8_t v) { return v == 0x5a; });
    }
#if defined(GGML_GEMMINI_TESTING)
    const size_t addresses = wr::test_weight_reader_code_address_resolutions();
    const size_t validations = wr::test_weight_reader_storage_validations();
    const size_t tile_k = std::min(static_cast<size_t>(DIM), size_t(32));
    ok = ok && validations == scalar_validations;
    if (gs == hp::status::ok) {
        ok = ok && validations == 1 && scalar_addresses == a.J * a.K &&
             addresses == a.J * (a.K / 32) * ((32 + tile_k - 1) / tile_k);
    } else {
        ok = ok && addresses == 0 && scalar_addresses == 0;
    }
    std::printf("W4_PACK_WORK %s scalar_addresses=%zu tile_addresses=%zu validations=%zu\n",
                name, scalar_addresses, addresses, validations);
#endif
    ++cases;
    failures += !ok;
    std::printf("W4_PACK_CASE %s %s status=%s reference=%s reader=%s\n", name, ok ? "PASS" : "FAIL",
                hp::status_name(gs), hp::status_name(rs), wr::weight_reader_status_name(got.reader));
}

} // namespace

int main() {
    for (size_t N : {size_t(1), size_t(2), size_t(15), size_t(16), size_t(17),
                     size_t(31), size_t(32), size_t(33), size_t(65)}) {
        for (size_t K : {size_t(hp::block_k), size_t(2 * hp::block_k), size_t(3 * hp::block_k)}) {
            const auto blocks = fixture(N, K);
            for (size_t gap : {size_t(0), size_t(10)}) {
                char name[80];
                std::snprintf(name, sizeof(name), "N%zu_K%zu_gap%zu", N, K, gap);
                compare(name, args_for(blocks, N, K), (N + 1) / 2 * 2 + gap, N + gap,
                        hp::status::ok);
            }
        }
    }
    const size_t N = 33, K = 3 * hp::block_k, ldb = 48, stride = 40;
    const auto original = fixture(N, K);
    auto check = [&](const char * name, hp::status want, auto mutate) {
        auto blocks = original;
        auto a = args_for(blocks, N, K);
        mutate(blocks, a);
        compare(name, a, ldb, stride, want);
    };
    check("all-zero", hp::status::ok, [](auto & b, auto &) {
        for (auto & q : b) { q.m = INT16_MIN; q.channel_scale = 0.0f; }
    });
    check("all-unit-carriers", hp::status::ok, [](auto & b, auto &) {
        for (auto & q : b) { q.m = 0; q.channel_scale = 1.0f; }
    });
    check("padding-late", hp::status::metadata, [](auto & b, auto &) { b.back().padding[1] = 1; });
    check("negative-carrier", hp::status::metadata, [](auto & b, auto &) { b.back().m = -1; });
    check("min-plus-one", hp::status::metadata, [](auto & b, auto &) { b.back().m = INT16_MIN + 1; });
    check("nan", hp::status::metadata, [](auto & b, auto &) {
        b.back().channel_scale = std::numeric_limits<float>::quiet_NaN();
    });
    check("infinity", hp::status::metadata, [](auto & b, auto &) {
        b.back().channel_scale = std::numeric_limits<float>::infinity();
    });
    check("metadata-before-row-error", hp::status::metadata, [](auto & b, auto &) {
        b[0].channel_scale = -1.0f;
        b.back().padding[0] = 1;
    });
    check("late-scale-mismatch", hp::block_k == 64 ? hp::status::pair_mismatch : hp::status::scale_mismatch, [](auto & b, auto &) {
        b[4 * (3 * hp::block_k / 32) + 1].channel_scale *= 2.0f;
    });
    check("negative-scale", hp::status::channel_scale, [](auto & b, auto &) {
        b[4 * (3 * hp::block_k / 32)].channel_scale = -1.0f;
    });
    check("negative-zero", hp::status::channel_scale, [](auto & b, auto &) {
        b[3 * (3 * hp::block_k / 32)].channel_scale = -0.0f;
    });
    check("zero-scale-live-carrier", hp::block_k == 64 ? hp::status::pair_mismatch : hp::status::channel_scale, [](auto & b, auto &) { b[3 * (3 * hp::block_k / 32) + 1].m = 0; });
    if constexpr (hp::block_k == 64) {
        check("late-group-scale-mismatch", hp::status::scale_mismatch, [](auto & b, auto &) {
            b[4 * 6 + 2].channel_scale *= 2.0f;
            b[4 * 6 + 3].channel_scale *= 2.0f;
        });
        check("zero-scale-live-group", hp::status::channel_scale, [](auto & b, auto &) {
            b[3 * 6 + 2].m = b[3 * 6 + 3].m = 0;
        });
        // Real quantizers: ordinary MXINT4 chooses each native32 exponent alone and is
        // refused; the Bk64 quantizer shares one exponent per 64-K group and is accepted.
        std::vector<float> wf(N * K);
        for (size_t i = 0; i < wf.size(); ++i)
            wf[i] = float(int(i % 7) - 3) * ((i / 32) % 2 ? 8.0f : 0.25f);
        check("bk32-quantized-model", hp::status::pair_mismatch, [&](auto & b, auto &) {
            ggml_quantize_chunk(GGML_TYPE_MXINT4, wf.data(), b.data(), 0, N, K, nullptr);
        });
        check("bk64-quantized-model", hp::status::ok, [&](auto & b, auto &) {
            ggml_quantize_chunk_mxint4_b64(wf.data(), b.data(), 0, N, K);
        });
        check("native-pair-m-mismatch", hp::status::pair_mismatch, [](auto & b, auto &) {
            b[0].m = 0; b[1].m = 1;
        });
        check("zero-half-in-live-group", hp::status::ok, [](auto & b, auto &) {
            b[0].m = b[1].m = 2;
            std::fill_n(b[0].qs, 16, uint8_t(0x88));
        });
    }
    check("truncated-bytes", hp::status::route, [](auto &, auto & a) { --a.native_weight_bytes; });
    check("truncated-count", hp::status::route, [](auto &, auto & a) { --a.native_block_count; });
    check("wrong-row-blocks", hp::status::route, [](auto &, auto & a) { ++a.native_blocks_per_row; });
    check("K-tail-rejected", hp::status::route, [](auto &, auto & a) { --a.K; });
    check("zero-K", hp::status::route, [](auto &, auto & a) { a.K = 0; });
    check("zero-N", hp::status::route, [](auto &, auto & a) { a.J = 0; });
    check("null-source", hp::status::route, [](auto &, auto & a) { a.q4_hp1_blocks = nullptr; });
    check("unaligned-source", hp::status::route, [](auto & b, auto & a) {
        a.q4_hp1_blocks = reinterpret_cast<const block_q4_hp1 *>(
            reinterpret_cast<const uint8_t *>(b.data()) + 1);
    });
    check("wrong-format", hp::status::invalid, [](auto &, auto & a) {
        a.weight_format = ggml_gemmini_args_t::im2p_weight_format_t::q4_h0;
    });
    const auto a = args_for(original, N, K);
    compare("short-ldb", a, 32, stride, hp::status::invalid);
    compare("odd-ldb", a, 49, stride, hp::status::invalid);
    compare("short-carrier-stride", a, ldb, 32, hp::status::invalid);
    for (int i = 1; i <= 5; ++i) {
        char name[32];
        std::snprintf(name, sizeof(name), "null-output-%d", i);
        compare(name, a, ldb, stride, i == 5 ? hp::status::ok : hp::status::invalid, i);
    }
#if !defined(GGML_GEMMINI_TESTING)
    std::printf("W4_PACK_WORK SKIP: compile test and weight_reader.cpp with GGML_GEMMINI_TESTING\n");
#endif
    std::printf("W4_PACK %s cases=%d failures=%d\n", failures ? "FAIL" : "PASS", cases, failures);
    return failures ? 1 : 0;
}
