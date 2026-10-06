#include "ggml-metal-quantized-producer.h"

#include "../ggml-gemmini/ggml-gemmini-args.h"
#include "../ggml-gemmini/quants/act/dispatch.hpp"
#include "../ggml-gemmini/quants/common/weight_reader.hpp"
#include "../ggml-gemmini/residual/direct/direct-builder.hpp"
#include "../ggml-gemmini/residual/rmd/rmd-run-aware.hpp"

#include <gemmini.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <list>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace act = ggml::gemmini::quants::act;
namespace reader = ggml::gemmini::quants::wreader;
namespace route = ggml::gemmini::quants::wroute;
namespace rmd = ggml::gemmini::rmd;

struct MetalResidualRequest {
    rmd::RunAwareRequest owned;
    std::vector<ggml_metal_quantized_run> runs;
    std::vector<ggml_metal_quantized_row> rows;
    ggml_metal_quantized_request view{};

    void bind() {
        runs.reserve(owned.runs.size());
        for (const auto & run : owned.runs) {
            runs.push_back({run.original_block_id, run.original_global_k_begin,
                run.union_k_mask, run.compact_k_begin, run.compact_k_count,
                run.original_local_k.data()});
        }
        rows.reserve(owned.rows.size());
        for (const auto & row : owned.rows) {
            rows.push_back({row.original_lane_id, row.source_row});
        }
        view = {owned.m, owned.n, owned.k, owned.source_row_begin,
            owned.source_row_count, owned.tile_i, owned.tile_j, owned.tile_k,
            runs.size(), runs.data(), rows.data(), owned.activations.data(),
            owned.weights.data(), owned.carriers.data()};
    }
};

struct MetalPreparedWeights {
    std::vector<uint8_t> packed_rows;
    std::vector<int8_t> codes;
    std::vector<float> scales, column_scales;
    std::vector<uint32_t> carriers;
};

struct ggml_metal_quantized_payload {
    ggml_metal_quantized_view view{};
    ggml_gemmini_args_t args;
    std::shared_ptr<const MetalPreparedWeights> weights;
    std::vector<float> activation_scales;
    std::vector<int32_t> block_residual;
    std::vector<int16_t> theta;
    std::vector<MetalResidualRequest> requests;
};

namespace {

bool fail(char * error, size_t capacity, const char * message) {
    if (error != nullptr && capacity != 0) {
        std::snprintf(error, capacity, "%s", message);
    }
    return false;
}

bool size_product(size_t a, size_t b, size_t & result) {
    return !__builtin_mul_overflow(a, b, &result);
}

constexpr size_t kWeightCacheLimit = size_t{2} * 1024 * 1024 * 1024;

class MetalWeightCache {
    struct Key {
        const void * data;
        ggml_type type;
        int64_t ne[GGML_MAX_DIMS];
        size_t nb[GGML_MAX_DIMS];

        explicit Key(const ggml_tensor & tensor) : data(tensor.data), type(tensor.type) {
            std::copy_n(tensor.ne, GGML_MAX_DIMS, ne);
            std::copy_n(tensor.nb, GGML_MAX_DIMS, nb);
        }

        bool matches(const ggml_tensor & tensor) const {
            return data == tensor.data && type == tensor.type &&
                std::equal(ne, ne + GGML_MAX_DIMS, tensor.ne) &&
                std::equal(nb, nb + GGML_MAX_DIMS, tensor.nb);
        }
    };

    struct Entry {
        Key key;
        std::shared_ptr<const MetalPreparedWeights> weights;
        size_t bytes;
    };

    std::mutex mutex;
    std::list<Entry> entries;
    size_t bytes = 0;
    size_t limit = kWeightCacheLimit;

    static bool same_bytes(const ggml_tensor & weight, const MetalPreparedWeights & prepared) {
        const size_t row_bytes = ggml_row_size(weight.type, weight.ne[0]);
        if (weight.nb[1] == row_bytes) {
            return std::memcmp(weight.data, prepared.packed_rows.data(), prepared.packed_rows.size()) == 0;
        }
        for (size_t row = 0; row < static_cast<size_t>(weight.ne[1]); ++row) {
            if (std::memcmp(static_cast<const uint8_t *>(weight.data) + row * weight.nb[1],
                    prepared.packed_rows.data() + row * row_bytes, row_bytes) != 0) {
                return false;
            }
        }
        return true;
    }

    void evict_to_limit() {
        while (!entries.empty() && bytes > limit) {
            bytes -= entries.back().bytes;
            entries.pop_back();
        }
    }

public:
    std::shared_ptr<const MetalPreparedWeights> find(const ggml_tensor & weight) {
        std::lock_guard<std::mutex> lock(mutex);
        for (auto entry = entries.begin(); entry != entries.end(); ++entry) {
            if (!entry->key.matches(weight)) continue;
            if (!same_bytes(weight, *entry->weights)) {
                bytes -= entry->bytes;
                entries.erase(entry);
                return {};
            }
            entries.splice(entries.begin(), entries, entry);
            return entries.front().weights;
        }
        return {};
    }

    std::shared_ptr<const MetalPreparedWeights> insert(const ggml_tensor & weight,
            std::shared_ptr<const MetalPreparedWeights> prepared) {
        std::lock_guard<std::mutex> lock(mutex);
        size_t size = sizeof(Entry) + sizeof(MetalPreparedWeights) + 64;
        const auto add = [&](size_t count, size_t element) {
            size_t allocation = 0;
            if (!size_product(count, element, allocation) || allocation > kWeightCacheLimit ||
                size > kWeightCacheLimit - allocation) return false;
            size += allocation;
            return true;
        };
        if (!add(prepared->packed_rows.capacity(), sizeof(uint8_t)) ||
            !add(prepared->codes.capacity(), sizeof(int8_t)) ||
            !add(prepared->scales.capacity(), sizeof(float)) ||
            !add(prepared->column_scales.capacity(), sizeof(float)) ||
            !add(prepared->carriers.capacity(), sizeof(uint32_t)) || size > limit) {
            return prepared;
        }
        for (auto entry = entries.begin(); entry != entries.end(); ++entry) {
            if (!entry->key.matches(weight)) continue;
            if (entry->weights->packed_rows == prepared->packed_rows) {
                entries.splice(entries.begin(), entries, entry);
                return entries.front().weights;
            }
            bytes -= entry->bytes;
            entries.erase(entry);
            break;
        }
        // Preserve useful resident weights when a model exceeds the budget;
        // streaming LRU eviction would miss every layer on the next pass.
        if (size > limit - bytes || entries.size() >= 256) return prepared;
        entries.push_front({Key(weight), prepared, size});
        bytes += size;
        return prepared;
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mutex);
        entries.clear();
        bytes = 0;
    }

    void set_limit(size_t value) {
        std::lock_guard<std::mutex> lock(mutex);
        limit = std::min(value, kWeightCacheLimit);
        evict_to_limit();
    }
};

MetalWeightCache & weight_cache() {
    static MetalWeightCache cache;
    return cache;
}

bool shape_supported(const ggml_tensor * weight, const ggml_tensor * activation) {
    const auto profile = ggml_metal_quantized_get_profile();
    if (weight == nullptr || activation == nullptr || activation->type != GGML_TYPE_F32 ||
        weight->ne[0] <= 0 || weight->ne[1] <= 0 || activation->ne[1] <= 0 ||
        weight->ne[0] != activation->ne[0] || weight->ne[0] % 32 != 0 ||
        weight->ne[2] != 1 || weight->ne[3] != 1 ||
        activation->ne[2] != 1 || activation->ne[3] != 1) {
        return false;
    }
    const auto expected = profile.mode == GGML_METAL_QUANTIZED_BLOCK
        ? (profile.bits == 4 ? GGML_TYPE_Q4_0 : GGML_TYPE_Q8_0)
        : (profile.bits == 4 ? GGML_TYPE_Q4_HP1 : GGML_TYPE_Q8_HP1);
    if (weight->type != expected) {
        return false;
    }
    size_t weight_row_bytes = 0;
    size_t activation_row_bytes = 0;
    if (!size_product(static_cast<size_t>(weight->ne[0] / 32), ggml_type_size(weight->type), weight_row_bytes) ||
        !size_product(static_cast<size_t>(activation->ne[0]), sizeof(float), activation_row_bytes) ||
        weight->nb[0] != ggml_type_size(weight->type) || activation->nb[0] != sizeof(float) ||
        weight->nb[1] < weight_row_bytes || activation->nb[1] < activation_row_bytes) {
        return false;
    }
    for (const auto * tensor : {weight, activation}) {
        size_t last_row = 0;
        const size_t row_bytes = tensor == weight ? weight_row_bytes : activation_row_bytes;
        if (!size_product(static_cast<size_t>(tensor->ne[1] - 1), tensor->nb[1], last_row) ||
            row_bytes > std::numeric_limits<size_t>::max() - last_row) {
            return false;
        }
    }
    size_t activation_count = 0, weight_count = 0, output_count = 0;
    return size_product(static_cast<size_t>(activation->ne[1]),
                        static_cast<size_t>(weight->ne[0]), activation_count) &&
        size_product(static_cast<size_t>(weight->ne[1]),
                     static_cast<size_t>(weight->ne[0]), weight_count) &&
        size_product(static_cast<size_t>(activation->ne[1]),
                     static_cast<size_t>(weight->ne[1]), output_count);
}

bool bind_weights(const ggml_tensor * weight, ggml_metal_quantized_payload & payload,
                     char * error, size_t capacity) {
    auto & args = payload.args;
    using Format = ggml_gemmini_args_t::im2p_weight_format_t;
    args.transpose_B = true;
    args.blocks_K = args.K / 32;
    args.blocks_J = args.J;
    args.blocks_I = args.J;
    args.block_size_k = 32;
    args.sB = args.K;
    args.native_blocks_per_row = args.blocks_K;
    args.native_block_count = args.blocks_K * args.J;
    args.native_weight_bytes = ggml_nbytes(weight);
    const size_t alignment = payload.view.profile.mode == GGML_METAL_QUANTIZED_BLOCK
        ? alignof(block_q8_0) : alignof(block_q8_hp1);
    if (reinterpret_cast<uintptr_t>(weight->data) % alignment != 0) {
        return fail(error, capacity, "unaligned original weight storage");
    }
    switch (weight->type) {
        case GGML_TYPE_Q4_0:
            args.weight_format = Format::q4_h0;
            args.q4_h0_blocks = static_cast<const block_q4_h0 *>(weight->data);
            break;
        case GGML_TYPE_Q8_0:
            args.weight_format = Format::q8_h0;
            args.B_blocks = static_cast<const block_q8_0 *>(weight->data);
            break;
        case GGML_TYPE_Q4_HP1:
            args.weight_format = Format::q4_hp1;
            args.q4_hp1_blocks = static_cast<const block_q4_hp1 *>(weight->data);
            break;
        case GGML_TYPE_Q8_HP1:
            args.weight_format = Format::q8_hp1;
            args.q8_hp1_blocks = static_cast<const block_q8_hp1 *>(weight->data);
            args.q8_hp1_blocks_per_row = args.blocks_K;
            args.q8_hp1_block_count = args.native_block_count;
            break;
        default:
            return fail(error, capacity, "unsupported original weight type");
    }
    return true;
}

bool decode_weights(ggml_metal_quantized_payload & payload, MetalPreparedWeights & weights,
                    char * error, size_t capacity) {
    auto & args = payload.args;
    const bool hp1 = payload.view.profile.mode == GGML_METAL_QUANTIZED_HP1_EXSIA;
    const auto plan = route::resolve_weight_route_plan(args,
        hp1 ? route::WeightScaleInfoMode::ResidualHp1Scu : route::WeightScaleInfoMode::Residual);
    if (!plan.valid) {
        return fail(error, capacity, plan.reject_reason);
    }
    weights.codes.resize(args.J * args.K);
    if (hp1) {
        weights.carriers.resize(args.blocks_K * args.J);
        weights.column_scales.resize(args.J);
    } else {
        weights.scales.resize(args.J * args.blocks_K);
    }
    for (size_t column = 0; column < args.J; ++column) {
        for (size_t block = 0; block < args.blocks_K; ++block) {
            if (hp1) {
                const auto value = reader::read_hp1_carrier_validated(args, plan, column, block);
                if (!value.ok() || (block > 0 && value.column_scale != weights.column_scales[column])) {
                    return fail(error, capacity, "invalid HP1 carrier or inconsistent column scale");
                }
                weights.carriers[block * args.J + column] = value.carrier;
                weights.column_scales[column] = value.column_scale;
            } else {
                const auto value = reader::read_scale_validated(args, plan, column, block);
                if (!value.ok() || !std::isfinite(value.floating_block_scale)) {
                    return fail(error, capacity, "invalid original block scale");
                }
                weights.scales[column * args.blocks_K + block] = value.floating_block_scale;
            }
            for (size_t local = 0; local < 32; ++local) {
                const size_t k = block * 32 + local;
                const auto code = reader::read_code_validated(args, plan, column, k);
                const int32_t bound = int32_t{1} << (payload.view.profile.bits - 1);
                if (!code.ok() || code.value < -bound || code.value >= bound) {
                    return fail(error, capacity, "invalid original signed weight code");
                }
                weights.codes[column * args.K + k] = static_cast<int8_t>(code.value);
            }
        }
    }
    return true;
}

bool prepare_weights(const ggml_tensor * weight, ggml_metal_quantized_payload & payload,
                     char * error, size_t capacity) {
    const bool contiguous = ggml_is_contiguous(weight);
    if (contiguous && !bind_weights(weight, payload, error, capacity)) return false;
    const size_t row_bytes = ggml_row_size(weight->type, weight->ne[0]);
    size_t packed_bytes = 0;
    if (!size_product(payload.args.J, row_bytes, packed_bytes)) {
        return fail(error, capacity, "packed weight row extent overflow");
    }
    ggml_tensor packed = *weight;
    packed.view_src = nullptr;
    packed.view_offs = 0;
    packed.nb[1] = row_bytes;
    packed.nb[2] = packed.nb[3] = packed_bytes;
    payload.weights = weight_cache().find(*weight);
    if (!payload.weights) {
        auto prepared = std::make_shared<MetalPreparedWeights>();
        prepared->packed_rows.resize(packed_bytes);
        for (size_t row = 0; row < payload.args.J; ++row) {
            std::memcpy(prepared->packed_rows.data() + row * row_bytes,
                static_cast<const uint8_t *>(weight->data) + row * weight->nb[1], row_bytes);
        }
        packed.data = prepared->packed_rows.data();
        if (!bind_weights(&packed, payload, error, capacity) ||
            !decode_weights(payload, *prepared, error, capacity)) return false;
        payload.weights = weight_cache().insert(*weight, std::move(prepared));
    }
    packed.data = const_cast<uint8_t *>(payload.weights->packed_rows.data());
    return bind_weights(contiguous ? weight : &packed, payload, error, capacity);
}

bool prepare_activation(const ggml_tensor * activation, ggml_metal_quantized_payload & payload,
                        char * error, size_t capacity) {
    auto & args = payload.args;
    const bool hp1 = payload.view.profile.mode == GGML_METAL_QUANTIZED_HP1_EXSIA;
    args.sA = args.K;
    args.sC = args.J;
    args.residual_route = hp1 ? act::ResidualRoute::ws_packet : act::ResidualRoute::cpu_direct;
    ggml::gemmini::gemmini_set_tile_ws(&args);
    const auto geometry = args.activation_geometry();
    if (!geometry.ok()) {
        return fail(error, capacity, "invalid production activation geometry");
    }
    args.activation_rows_per_stripe = geometry.geometry.stripe_rows;
    if (!args.A.allocate(args.I, args.K, payload.view.profile.bits) || !act::quantize(activation, args)) {
        return fail(error, capacity, "CPU activation producer failed");
    }
    act::ActivationMetadataView metadata(args, 0, args.I);
    if (!metadata.valid()) {
        return fail(error, capacity, "invalid producer activation metadata");
    }
    payload.activation_scales.resize(args.I * (hp1 ? 1 : args.blocks_K));
    for (size_t row = 0; row < args.I; ++row) {
        for (size_t block = 0; block < (hp1 ? 1 : args.blocks_K); ++block) {
            if (!metadata.scale(row, block * 32,
                    payload.activation_scales[row * (hp1 ? 1 : args.blocks_K) + block])) {
                return fail(error, capacity, "invalid producer activation scale");
            }
        }
    }
    if (hp1) {
        payload.theta.resize(metadata.stripe_count());
        for (size_t stripe = 0; stripe < payload.theta.size(); ++stripe) {
            if (!metadata.theta(stripe, payload.theta[stripe])) {
                return fail(error, capacity, "invalid producer stripe theta");
            }
        }
        const auto & packets = act::rmd_packets(args);
        payload.requests.resize(packets.size());
        for (size_t index = 0; index < packets.size(); ++index) {
            auto & request = payload.requests[index];
            if (rmd::build_run_aware_request(args, packets[index], request.owned) != rmd::RmdStatus::success ||
                request.owned.empty()) {
                return fail(error, capacity, "invalid production residual run request");
            }
            request.bind();
        }
    } else {
        const auto & direct = act::direct_residuals(args);
        if (!direct.empty()) {
            payload.block_residual.assign(args.I * args.K, 0);
            if (ggml::gemmini::residual::expand_direct_payloads_to_plane(
                    direct, 0, args.I, args.K, args.J, args.K, payload.block_residual) != rmd::RmdStatus::success) {
                return fail(error, capacity, "invalid block-local residual payload");
            }
        }
    }
    return true;
}

} // namespace

extern "C" ggml_metal_quantized_profile ggml_metal_quantized_get_profile(void) {
    static_assert(GGML_GEMMINI_ACTIVATION_BITS == 4 || GGML_GEMMINI_ACTIVATION_BITS == 8,
                  "Metal quantized producer supports A4/A8 only");
    static_assert(GGML_GEMMINI_ACTIVATION_QUANT == 0 || GGML_GEMMINI_ACTIVATION_QUANT == 3,
                  "Metal quantized producer supports BLOCK or EXSIA only");
    return {GGML_GEMMINI_ACTIVATION_QUANT == 3 ? GGML_METAL_QUANTIZED_BLOCK : GGML_METAL_QUANTIZED_HP1_EXSIA,
        GGML_GEMMINI_ACTIVATION_BITS, DIM, GGML_GEMMINI_ENABLE_RMD != 0};
}

extern "C" bool ggml_metal_quantized_can_prepare(const ggml_tensor * weight, const ggml_tensor * activation) {
    return shape_supported(weight, activation);
}

extern "C" bool ggml_metal_quantized_prepare(const ggml_tensor * weight, const ggml_tensor * activation,
        ggml_metal_quantized_payload ** out, char * error, size_t error_capacity) {
    if (out == nullptr || !shape_supported(weight, activation) ||
        weight->data == nullptr || activation->data == nullptr) {
        return fail(error, error_capacity, "unsupported Metal quantized tensor type, width, shape or layout");
    }
    try {
        auto payload = std::make_unique<ggml_metal_quantized_payload>();
        payload->view.profile = ggml_metal_quantized_get_profile();
        auto & args = payload->args;
        args.I = static_cast<size_t>(activation->ne[1]);
        args.J = static_cast<size_t>(weight->ne[1]);
        args.K = static_cast<size_t>(weight->ne[0]);
        args.matmul_layer = weight->name;
        ggml_tensor packed_activation = *activation;
        packed_activation.view_src = nullptr;
        packed_activation.view_offs = 0;
        std::vector<float> activation_rows;
        if (!ggml_is_contiguous(activation)) {
            activation_rows.resize(args.I * args.K);
            for (size_t row = 0; row < args.I; ++row) {
                std::memcpy(activation_rows.data() + row * args.K,
                    static_cast<const uint8_t *>(activation->data) + row * activation->nb[1], args.K * sizeof(float));
            }
            packed_activation.data = activation_rows.data();
            packed_activation.nb[1] = args.K * sizeof(float);
            packed_activation.nb[2] = packed_activation.nb[3] = args.I * packed_activation.nb[1];
        }
        if (!prepare_weights(weight, *payload, error, error_capacity) ||
            !prepare_activation(&packed_activation, *payload, error, error_capacity)) {
            return false;
        }
        auto & view = payload->view;
        view.m = args.I;
        view.n = args.J;
        view.k = args.K;
        view.activation_rows_per_stripe = args.activation_rows_per_stripe;
        view.activations = reinterpret_cast<const int8_t *>(args.A.raw_data());
        const auto & weights = *payload->weights;
        view.weights = weights.codes.data();
        view.activation_scales = payload->activation_scales.data();
        view.weight_scales = weights.scales.empty() ? nullptr : weights.scales.data();
        view.block_residual = payload->block_residual.empty() ? nullptr : payload->block_residual.data();
        view.carriers = weights.carriers.empty() ? nullptr : weights.carriers.data();
        view.column_scales = weights.column_scales.empty() ? nullptr : weights.column_scales.data();
        view.stripe_theta = payload->theta.empty() ? nullptr : payload->theta.data();
        view.stripe_count = payload->theta.size();
        view.request_count = payload->requests.size();
        view.original_weights = weight->data;
        view.original_weight_bytes = ggml_nbytes(weight);
        view.original_weight_row_stride = weight->nb[1];
        view.original_weight_type = weight->type;
        *out = payload.release();
        if (error != nullptr && error_capacity > 0) {
            error[0] = '\0';
        }
        return true;
    } catch (const std::bad_alloc &) {
        return fail(error, error_capacity, "Metal CPU producer allocation failed");
    } catch (const std::length_error &) {
        return fail(error, error_capacity, "Metal CPU producer payload extent overflow");
    } catch (const std::exception & exception) {
        return fail(error, error_capacity, exception.what());
    }
}

extern "C" const ggml_metal_quantized_view * ggml_metal_quantized_get_view(
        const ggml_metal_quantized_payload * payload) {
    return payload == nullptr ? nullptr : &payload->view;
}

extern "C" const ggml_metal_quantized_request * ggml_metal_quantized_get_request(
        const ggml_metal_quantized_payload * payload, size_t index) {
    return payload == nullptr || index >= payload->requests.size() ? nullptr : &payload->requests[index].view;
}

extern "C" void ggml_metal_quantized_free(ggml_metal_quantized_payload * payload) {
    delete payload;
}

extern "C" void ggml_metal_quantized_clear_weight_cache(void) {
    weight_cache().clear();
}

extern "C" void ggml_metal_quantized_set_weight_cache_limit(size_t bytes) {
    weight_cache().set_limit(bytes);
}
