#include <ggml.h>
#include "../ggml/src/ggml-metal/ggml-metal-quantized-producer.h"

#define GGML_COMMON_DECL_CPP
#include "../ggml/src/ggml-common.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

namespace {

using Payload = std::unique_ptr<ggml_metal_quantized_payload, decltype(&ggml_metal_quantized_free)>;

bool check(bool condition, const char * message) {
    if (!condition) std::fprintf(stderr, "FAIL: %s\n", message);
    return condition;
}

struct Fixture {
    static constexpr size_t k = 64, n = 3;
    ggml_metal_quantized_profile profile = ggml_metal_quantized_get_profile();
    bool hp1 = profile.mode == GGML_METAL_QUANTIZED_HP1_EXSIA;
    ggml_type type = hp1 ? (profile.bits == 4 ? GGML_TYPE_Q4_HP1 : GGML_TYPE_Q8_HP1)
                        : (profile.bits == 4 ? GGML_TYPE_Q4_0 : GGML_TYPE_Q8_0);
    ggml_context * context = ggml_init({4 * ggml_tensor_overhead(), nullptr, true});
    ggml_tensor * weight = ggml_new_tensor_2d(context, type, k, n);
    ggml_tensor * activation = ggml_new_tensor_2d(context, GGML_TYPE_F32, k, 1);
    size_t row_bytes = ggml_row_size(type, k);
    std::vector<uint8_t> storage;
    std::vector<float> source = std::vector<float>(k, 1.0f);

    explicit Fixture(bool padded = false) {
        const size_t stride = row_bytes + (padded ? ggml_type_size(type) : 0);
        storage.resize(ggml_type_size(type) + n * stride, 0xa5);
        weight->data = storage.data() + ggml_type_size(type);
        weight->nb[1] = stride;
        weight->nb[2] = weight->nb[3] = n * stride;
        activation->data = source.data();
        for (size_t row = 0; row < n; ++row) {
            for (size_t block = 0; block < k / 32; ++block) {
                auto * bytes = static_cast<uint8_t *>(weight->data) + row * stride + block * ggml_type_size(type);
                std::memset(bytes, 0, ggml_type_size(type));
                std::memset(bytes + (hp1 ? 0 : sizeof(ggml_half)), profile.bits == 4 ? 0x97 : 1,
                            profile.bits == 4 ? 16 : 32);
                if (hp1) {
                    const float scale = 0.25f;
                    std::memcpy(bytes + scale_offset(), &scale, sizeof(scale));
                } else {
                    const ggml_half scale = ggml_fp32_to_fp16(0.25f);
                    std::memcpy(bytes, &scale, sizeof(scale));
                }
            }
        }
    }

    ~Fixture() { ggml_free(context); }

    size_t scale_offset() const {
        return profile.bits == 4 ? offsetof(block_q4_hp1, channel_scale) : offsetof(block_q8_hp1, channel_scale);
    }

    Payload prepare() const {
        ggml_metal_quantized_payload * result = nullptr;
        char error[256]{};
        if (!ggml_metal_quantized_prepare(weight, activation, &result, error, sizeof(error))) {
            std::fprintf(stderr, "FAIL: prepare: %s\n", error);
        }
        return Payload(result, ggml_metal_quantized_free);
    }

    void mutate_code() {
        auto * codes = static_cast<uint8_t *>(weight->data) + (hp1 ? 0 : sizeof(ggml_half));
        codes[0] = profile.bits == 4 ? 0x96 : 2;
    }
};

bool mutation_and_lifetime(bool padded) {
    ggml_metal_quantized_clear_weight_cache();
    Fixture fixture(padded);
    auto original = fixture.prepare();
    if (!original) return false;
    const auto & first = *ggml_metal_quantized_get_view(original.get());
    auto repeated = fixture.prepare();
    if (!repeated) return false;
    const auto & again = *ggml_metal_quantized_get_view(repeated.get());
    bool ok = check(first.weights == again.weights, "unchanged weights share immutable decoded data");
    ok &= check(first.weight_scales == again.weight_scales && first.carriers == again.carriers &&
                first.column_scales == again.column_scales, "unchanged weight metadata shares ownership");
    std::fill(fixture.source.begin(), fixture.source.end(), -2.0f);
    auto activation_changed = fixture.prepare();
    if (!activation_changed) return false;
    const auto & changed_a = *ggml_metal_quantized_get_view(activation_changed.get());
    ok &= check(first.weights == changed_a.weights && first.activations != changed_a.activations &&
                first.activations[0] != changed_a.activations[0], "cache hit reruns activation producer");
    if (padded) {
        static_cast<uint8_t *>(fixture.weight->data)[fixture.row_bytes] ^= 0xff;
        auto padding_changed = fixture.prepare();
        if (!padding_changed) return false;
        ok &= check(first.weights == ggml_metal_quantized_get_view(padding_changed.get())->weights,
                    "unused row padding does not invalidate exact logical bytes");
    }
    fixture.mutate_code();
    auto mutated = fixture.prepare();
    if (!mutated) return false;
    const auto & changed_w = *ggml_metal_quantized_get_view(mutated.get());
    ok &= check(changed_w.weights != first.weights && changed_w.weights[0] == (fixture.profile.bits == 4 ? -2 : 2),
                "same-address packed code mutation is decoded");
    ok &= check(first.weights[0] == (fixture.profile.bits == 4 ? -1 : 1), "retained old decoded data stays valid");
    auto * bytes = static_cast<uint8_t *>(fixture.weight->data);
    if (fixture.hp1) {
        const float scale = 0.5f;
        for (size_t block = 0; block < Fixture::k / 32; ++block) {
            std::memcpy(bytes + block * ggml_type_size(fixture.type) + fixture.scale_offset(), &scale, sizeof(scale));
        }
        const int16_t carrier = 1;
        const size_t offset = fixture.profile.bits == 4 ? offsetof(block_q4_hp1, m) : offsetof(block_q8_hp1, m);
        std::memcpy(bytes + offset, &carrier, sizeof(carrier));
    } else {
        const ggml_half scale = ggml_fp32_to_fp16(0.5f);
        std::memcpy(bytes, &scale, sizeof(scale));
    }
    auto metadata_changed = fixture.prepare();
    if (!metadata_changed) return false;
    const auto & changed_metadata = *ggml_metal_quantized_get_view(metadata_changed.get());
    ok &= check(fixture.hp1 ? changed_metadata.column_scales[0] == 0.5f && changed_metadata.carriers[0] == 1 &&
                             first.column_scales[0] == 0.25f && first.carriers[0] == 0
                           : changed_metadata.weight_scales[0] == 0.5f && first.weight_scales[0] == 0.25f,
                "scale and carrier mutations refresh metadata without changing retained data");
    std::vector<uint8_t> valid(bytes, bytes + ggml_nbytes(fixture.weight));
    if (fixture.hp1) {
        const float invalid = std::numeric_limits<float>::infinity();
        std::memcpy(bytes + fixture.scale_offset(), &invalid, sizeof(invalid));
    } else {
        const ggml_half invalid = ggml_fp32_to_fp16(std::numeric_limits<float>::infinity());
        std::memcpy(bytes, &invalid, sizeof(invalid));
    }
    auto * unpublished = original.get();
    char error[256]{};
    ok &= check(!ggml_metal_quantized_prepare(fixture.weight, fixture.activation, &unpublished, error, sizeof(error)) &&
                unpublished == original.get(), "invalid scale mutation rejects without publishing");
    std::memcpy(bytes, valid.data(), valid.size());
    auto restored = fixture.prepare();
    if (!restored) return false;
    ok &= check(ggml_metal_quantized_get_view(restored.get())->weights[0] == changed_w.weights[0],
                "valid storage can be reused after a rejected mutation");
    ggml_metal_quantized_clear_weight_cache();
    auto cleared = fixture.prepare();
    if (!cleared) return false;
    ok &= check(ggml_metal_quantized_get_view(cleared.get())->weights != ggml_metal_quantized_get_view(restored.get())->weights &&
                first.weights[0] == (fixture.profile.bits == 4 ? -1 : 1), "clear releases cache ownership and preserves live payloads");
    return ok;
}

bool pointer_reuse_and_layout() {
    ggml_metal_quantized_clear_weight_cache();
    Fixture fixture(true);
    auto original = fixture.prepare();
    if (!original) return false;
    const auto & first = *ggml_metal_quantized_get_view(original.get());
    ggml_tensor replacement = *fixture.weight;
    fixture.weight = &replacement;
    fixture.mutate_code();
    auto reused = fixture.prepare();
    if (!reused) return false;
    bool ok = check(ggml_metal_quantized_get_view(reused.get())->weights[0] == (fixture.profile.bits == 4 ? -2 : 2) &&
                    first.weights[0] == (fixture.profile.bits == 4 ? -1 : 1),
                    "a replacement tensor at reused storage cannot retrieve stale codes");
    auto * bytes = static_cast<uint8_t *>(replacement.data);
    const size_t padded_stride = replacement.nb[1];
    for (size_t row = 0; row < Fixture::n; ++row) {
        std::memcpy(bytes + row * padded_stride + fixture.row_bytes, bytes, ggml_type_size(fixture.type));
    }
    replacement.nb[1] = fixture.row_bytes;
    replacement.nb[2] = replacement.nb[3] = Fixture::n * fixture.row_bytes;
    auto restrided = fixture.prepare();
    if (!restrided) return false;
    const auto & changed = *ggml_metal_quantized_get_view(restrided.get());
    ok &= check(changed.weights[Fixture::k] == (fixture.profile.bits == 4 ? -2 : 2) &&
                first.weights[Fixture::k] == (fixture.profile.bits == 4 ? -1 : 1),
                "same base with changed row stride decodes the new logical rows");
    replacement.nb[1] = padded_stride;
    replacement.nb[2] = replacement.nb[3] = Fixture::n * padded_stride;
    auto original_stride = fixture.prepare();
    if (!original_stride) return false;
    ok &= check(ggml_metal_quantized_get_view(original_stride.get())->weights == ggml_metal_quantized_get_view(reused.get())->weights,
                "distinct stride keys retain independent cache entries");
    replacement.ne[0] = 32;
    replacement.ne[1] = Fixture::n * 2;
    replacement.nb[1] = fixture.row_bytes / 2;
    replacement.nb[2] = replacement.nb[3] = Fixture::n * fixture.row_bytes;
    fixture.activation->ne[0] = 32;
    fixture.activation->nb[1] = fixture.activation->nb[2] = fixture.activation->nb[3] = 32 * sizeof(float);
    auto reshaped = fixture.prepare();
    if (!reshaped) return false;
    const auto & reshaped_view = *ggml_metal_quantized_get_view(reshaped.get());
    ok &= check(reshaped_view.k == 32 && reshaped_view.n == Fixture::n * 2 && reshaped_view.weights != changed.weights,
                "identical packed bytes with different dimensions use a distinct cache entry");
    return ok;
}

bool bounded_cache() {
    ggml_metal_quantized_clear_weight_cache();
    ggml_metal_quantized_set_weight_cache_limit(1536);
    Fixture a, b, c;
    b.mutate_code();
    c.mutate_code();
    auto first = a.prepare();
    auto hit = a.prepare();
    if (!first || !hit) return false;
    const auto * first_codes = ggml_metal_quantized_get_view(first.get())->weights;
    bool ok = check(first_codes == ggml_metal_quantized_get_view(hit.get())->weights, "one small entry fits cache limit");
    auto second = b.prepare();
    auto third = c.prepare();
    if (!second || !third) return false;
    const auto * second_codes = ggml_metal_quantized_get_view(second.get())->weights;
    const auto * third_codes = ggml_metal_quantized_get_view(third.get())->weights;
    for (size_t pass = 0; pass < 3; ++pass) {
        auto repeated_a = a.prepare();
        auto repeated_b = b.prepare();
        auto repeated_c = c.prepare();
        if (!repeated_a || !repeated_b || !repeated_c) return false;
        const auto & av = *ggml_metal_quantized_get_view(repeated_a.get());
        const auto & bv = *ggml_metal_quantized_get_view(repeated_b.get());
        const auto & cv = *ggml_metal_quantized_get_view(repeated_c.get());
        ok &= check(av.weights == first_codes && bv.weights == second_codes && cv.weights != third_codes,
                    "over-budget weight stream retains early entries and bypasses later entries");
        ok &= check(av.weights[0] == (a.profile.bits == 4 ? -1 : 1) &&
                    bv.weights[0] == (a.profile.bits == 4 ? -2 : 2) && cv.weights[0] == bv.weights[0],
                    "resident and uncached stream weights retain exact independent codes");
        ok &= check(std::memcmp(bv.weights, cv.weights, Fixture::n * Fixture::k) == 0 &&
                    std::memcmp(bv.activations, cv.activations, Fixture::k) == 0 &&
                    std::memcmp(bv.activation_scales, cv.activation_scales,
                                (b.hp1 ? 1 : Fixture::k / 32) * sizeof(float)) == 0,
                    "cache admission leaves numerical activation and weight payloads bitwise identical");
        ok &= check(b.hp1 ? std::memcmp(bv.carriers, cv.carriers, Fixture::n * Fixture::k / 32 * sizeof(uint32_t)) == 0 &&
                           std::memcmp(bv.column_scales, cv.column_scales, Fixture::n * sizeof(float)) == 0
                         : std::memcmp(bv.weight_scales, cv.weight_scales, Fixture::n * Fixture::k / 32 * sizeof(float)) == 0,
                    "resident and uncached stream scales and carriers are bitwise identical");
    }
    auto newest_a = a.prepare();
    if (!newest_a) return false;
    ggml_metal_quantized_set_weight_cache_limit(1024);
    auto evicted_b = b.prepare();
    auto resident_a = a.prepare();
    if (!evicted_b || !resident_a) return false;
    ok &= check(ggml_metal_quantized_get_view(resident_a.get())->weights == first_codes &&
                ggml_metal_quantized_get_view(evicted_b.get())->weights != second_codes &&
                second_codes[0] == (a.profile.bits == 4 ? -2 : 2),
                "explicit limit shrink evicts the older entry and preserves live payloads");
    ggml_metal_quantized_set_weight_cache_limit(1);
    auto oversized = a.prepare();
    auto uncached = a.prepare();
    if (!oversized || !uncached) return false;
    ok &= check(ggml_metal_quantized_get_view(oversized.get())->weights != ggml_metal_quantized_get_view(uncached.get())->weights,
                "oversized entries bypass cache");
    ggml_metal_quantized_set_weight_cache_limit(0);
    auto disabled = a.prepare();
    auto still_disabled = a.prepare();
    if (!disabled || !still_disabled) return false;
    ok &= check(ggml_metal_quantized_get_view(disabled.get())->weights != ggml_metal_quantized_get_view(still_disabled.get())->weights,
                "zero limit disables reuse");
    ggml_metal_quantized_set_weight_cache_limit(size_t{2} * 1024 * 1024 * 1024);
    return ok;
}

} // namespace

int main() {
    bool ok = mutation_and_lifetime(false);
    ok &= mutation_and_lifetime(true);
    ok &= pointer_reuse_and_layout();
    ok &= bounded_cache();
    ggml_metal_quantized_clear_weight_cache();
    std::printf("METAL_WEIGHT_CACHE %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
