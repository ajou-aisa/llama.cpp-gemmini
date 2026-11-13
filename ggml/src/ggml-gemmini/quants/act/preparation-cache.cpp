#include "preparation-cache.hpp"
#include "quantize.hpp"
#include "exsia/exsia.hpp"
#include "../../ggml-gemmini-args.h"
#include "../../potal-sensitivity.hpp"

#include <cfenv>
#include <cstdlib>
#include <cstring>
#include <string_view>

namespace ggml::gemmini::quants {
namespace {
std::pair<std::string, unsigned> projection_family(const ggml_tensor * node) {
    if (!node || !node->src[0] || !node->src[1]) return {};
    const std::string_view name(node->src[0]->name);
    const char * suffixes[] = {"attn_q.weight", "attn_k.weight", "attn_v.weight",
                              "ffn_gate.weight", "ffn_up.weight"};
    for (unsigned role = 0; role < 5; ++role) {
        const std::string_view suffix(suffixes[role]);
        if (name.size() >= suffix.size() && name.substr(name.size()-suffix.size()) == suffix)
            return {std::string(name.substr(0, name.size()-suffix.size())) +
                    (role < 3 ? "attention" : "ffn"), 1u << role};
    }
    return {};
}

bool rebind_columns(act::exsia::Meta & meta, size_t columns) {
    const size_t padded = rmd::align_up(columns, DIM);
    if (!padded) return false;
    for (auto & handle : meta.rmd_packets) {
        if (handle->logical_j == columns) continue;
        auto packet = std::make_shared<rmd::StripePacket>(*handle);
        packet->logical_j = columns;
        packet->j_padded = padded;
        packet->total_output_values = 0;
        for (auto & block : packet->blocks) {
            if (padded > UINT32_MAX / block.rows_padded) return false;
            const size_t stride = block.rows_padded * padded;
            if (block.active_lane_count > (UINT32_MAX-packet->total_output_values)/stride)
                return false;
            block.output_value_offset = static_cast<uint32_t>(packet->total_output_values);
            block.lane_stride_values = static_cast<uint32_t>(stride);
            packet->total_output_values += stride * block.active_lane_count;
        }
        handle = std::move(packet);
    }
    for (auto & handle : meta.direct_residuals) {
        if (handle->logical_j == columns) continue;
        auto payload = std::make_shared<residual::DirectStripePayload>(*handle);
        payload->logical_j = columns;
        handle = std::move(payload);
    }
    return true;
}

bool publish(ggml_gemmini_args_t & args, const act::exsia::Meta & meta) {
    const auto * sink = args.exsia_stripe_ready_sink;
    if (!sink || !sink->on_ready) return true;
    size_t packet = 0, direct = 0;
    for (size_t stripe = 0; stripe < meta.theta.size(); ++stripe) {
        act::exsia::StripeReadyEvent event{};
        event.run_id = *meta.run_id;
        event.stripe_id = stripe;
        event.slot = stripe % act::exsia::EXSIA_PIPELINE_SLOT_COUNT;
        event.row_begin = stripe * args.activation_rows_per_stripe;
        event.row_end = std::min(args.I, event.row_begin + args.activation_rows_per_stripe);
        event.activation_metadata = act::exsia::StripeMetadataSnapshot{
            static_cast<int16_t>(meta.theta[stripe]+meta.rho), meta.rho, meta.sigma, meta.theta[stripe]};
        if (packet < meta.rmd_packets.size() && meta.rmd_packets[packet]->stripe_id == stripe)
            event.rmd_packet = meta.rmd_packets[packet++];
        if (direct < meta.direct_residuals.size() && meta.direct_residuals[direct]->stripe_id == stripe)
            event.direct_residual = meta.direct_residuals[direct++];
        // A reuse has no new Local/Folding interval; the caller times all reuse work.
        if (!sink->on_ready(sink->user_data, event)) return false;
    }
    return true;
}
} // namespace

void PreparationCache::clear() {
    roles_ = 0;
    input_.clear();
    main_ = {};
    metadata_.reset();
}

PreparationCache::Result PreparationCache::prepare(const ggml_tensor * node,
                                                   ggml_gemmini_args_t & args) {
    const auto * src = node ? node->src[1] : nullptr;
    const auto fresh = [&] { return Result{quantize_activation(src, args), false}; };
#if CYCLE_SIM || EXSIA_OBSERVATION_ENABLED || GGML_GEMMINI_ACT_QUANT_METRICS || \
    GGML_GEMMINI_RESIDUAL_METRICS || GGML_GEMMINI_SCALE_METRICS
    clear();
    return fresh();
#else
    const auto family = projection_family(node);
    const int rounding = std::fegetround();
    const char * force = std::getenv("GGML_GEMMINI_EXSIA_FORCE_RECOMPUTE");
    GemminiGeometry geometry;
    if (config::CURRENT_ACTIVATION_QUANT != config::ActivationQuantAlgo::EXSIA ||
        rounding == -1 || (force && std::strcmp(force, "1") == 0) ||
        !family.second || !src || !src->data || src->type != GGML_TYPE_F32 ||
        !ggml_is_contiguous(src) || src->ne[2] != 1 || src->ne[3] != 1 ||
        size_t(src->ne[0]) != args.K || size_t(src->ne[1]) != args.I ||
        !args.activation_quant_geometry_matches(geometry) || !args.activation_rows_per_stripe ||
        !args.A.valid() || args.A.row_offset || args.A.row_stride_bytes != args.K ||
        args.A.bits != GGML_GEMMINI_ACTIVATION_BITS || args.A.rows != args.I || args.A.cols != args.K ||
        sensitivity::profiling() || sensitivity::shadow()) {
        clear();
        return fresh();
    }
    const Key key{family.first, src, src->data, args.I, args.K, args.activation_rows_per_stripe,
                  args.folding_anchor, static_cast<int>(args.residual_route), rounding};
    const size_t bytes = ggml_nbytes(src);
    const bool hit = roles_ && !(roles_ & family.second) && key_ == key &&
        input_.size() == bytes && std::memcmp(input_.data(), src->data, bytes) == 0;
    if (hit) {
        auto meta = metadata_;
        if (!rebind_columns(meta, args.J)) { clear(); return fresh(); }
        meta.run_id = act::exsia::next_exsia_run_id();
        if (args.exsia_stripe_ready_sink) {
            // The collector may already own this backing: copy, do not replace it.
            if (args.A.bytes->size() != main_.bytes->size()) { clear(); return fresh(); }
            std::memcpy(args.A.bytes->data(), main_.bytes->data(), main_.bytes->size());
        } else args.A = main_;
        args.act_quant.storage() = meta;
        args.gemmini_call_k_logical = args.K;
        args.gemmini_call_k_aligned = args.K;
        args.gemmini_call_tile_k_elems = args.tile_K > 0 ? args.tile_K*DIM : args.K;
        if (!publish(args, meta)) {
            clear();
            reset_activation_quant_state(args);
            return {false, true};
        }
        roles_ |= family.second;
        return {true, true};
    }
    clear();
    const auto result = fresh();
    if (!result.ok) return result;
    const auto * meta = std::get_if<act::exsia::Meta>(&args.act_quant.storage());
    if (!meta) return result;
    input_.assign(static_cast<const uint8_t *>(src->data), static_cast<const uint8_t *>(src->data)+bytes);
    key_ = key;
    main_ = args.A;
    metadata_ = *meta;
    roles_ = family.second;
    return result;
#endif
}
} // namespace ggml::gemmini::quants
