#include "ggml-gemmini-im2p.hpp"

#include <array>
#include <cstdint>
#include <cstdio>

namespace {

using ggml::gemmini::im2p_adapter::BuildIdentity;
using ggml::gemmini::im2p_adapter::Error;
using ggml::gemmini::im2p_adapter::ExsiaRouteRequest;
using ggml::gemmini::im2p_adapter::PublicMode;
using ggml::gemmini::im2p_adapter::ResidualBackend;
using ggml::gemmini::im2p_adapter::WeightFamily;
using ggml::gemmini::im2p_adapter::gate_route;

#if defined(IM2P_SIM_IMPLEMENTATION_GEMMINI_HP1) && !CYCLE_SIM
constexpr bool h1_supported = false;
#else
constexpr bool h1_supported = true;
#endif

constexpr std::array<std::uint8_t, 3> widths{{4, 8, 16}};
constexpr std::array<PublicMode, 2>   modes{{PublicMode::full, PublicMode::stripe_pipeline}};
constexpr std::array<WeightFamily, 3> supported_families{
    {WeightFamily::h0, WeightFamily::h1, WeightFamily::hp1}};
constexpr std::array<WeightFamily, 2>    deprecated_families{{WeightFamily::h2, WeightFamily::hp2}};
constexpr std::array<ResidualBackend, 2> residual_backends{
    {ResidualBackend::cpu_direct, ResidualBackend::compact_ws}};

bool expect(const char *              name,
            const ExsiaRouteRequest & request,
            Error                     expected,
            std::size_t &             accepted,
            std::size_t &             rejected) {
    const auto result = gate_route(request);
    if (result.error != expected) {
        std::fprintf(stderr,
                     "FAIL: %s returned %u, expected %u (%s)\n",
                     name,
                     static_cast<unsigned>(result.error),
                     static_cast<unsigned>(expected),
                     result.message);
        return false;
    }
    result.ok() ? ++accepted : ++rejected;
    return true;
}

ExsiaRouteRequest request(std::uint8_t    activation_bits,
                          std::uint8_t    weight_bits,
                          PublicMode      mode,
                          WeightFamily    family,
                          ResidualBackend backend) {
    ExsiaRouteRequest value{};
    value.exsia                    = true;
    value.activation_bits          = activation_bits;
    value.weight_bits              = weight_bits;
    value.artifact_activation_bits = activation_bits;
    value.artifact_weight_bits     = weight_bits;
    value.rmd_enabled              = true;
    value.mode                     = mode;
    value.family                   = family;
    value.residual_backend         = backend;
    value.build_identity           = BuildIdentity::im2p_sim_ws;
    return value;
}

} // namespace

int main() {
    std::size_t accepted = 0;
    std::size_t rejected = 0;

    for (const auto width : widths) {
        for (const auto mode : modes) {
            for (const auto family : supported_families) {
                for (const auto backend : residual_backends) {
                    const bool supported = (family != WeightFamily::h1 || h1_supported) &&
                                           (family != WeightFamily::h0 ||
                                            (width != 8 && backend == ResidualBackend::cpu_direct));
                    if (!expect("matched capability",
                                request(width, width, mode, family, backend),
                                supported ? Error::success : Error::unsupported_route,
                                accepted,
                                rejected)) {
                        return 1;
                    }
                }
            }
        }
    }
    if (accepted != (h1_supported ? 28u : 16u)) {
        std::fprintf(stderr, "FAIL: accepted %zu routes, unexpected capability count\n", accepted);
        return 1;
    }

    // Every ordered mixed A/W pair fails for every relevant family/backend in
    // both public modes.
    for (const auto activation_bits : widths) {
        for (const auto weight_bits : widths) {
            if (activation_bits == weight_bits)
                continue;
            for (const auto mode : modes) {
                for (const auto family : supported_families) {
                    for (const auto backend : residual_backends) {
                        if (!expect("ordered mixed width",
                                    request(activation_bits, weight_bits, mode, family, backend),
                                    Error::unsupported_route,
                                    accepted,
                                    rejected)) {
                            return 1;
                        }
                    }
                }
            }
        }
    }

    for (const auto width : widths) {
        for (const auto mode : modes) {
            for (const auto family : deprecated_families) {
                for (const auto backend : residual_backends) {
                    if (!expect("deprecated family",
                                request(width, width, mode, family, backend),
                                Error::unsupported_route,
                                accepted,
                                rejected)) {
                        return 1;
                    }
                }
            }
        }
    }

    for (const auto mode : modes) {
        auto invalid_mode = request(8, 8, mode, WeightFamily::h1, ResidualBackend::cpu_direct);
        invalid_mode.mode = static_cast<PublicMode>(0xff);
        if (gate_route(invalid_mode).error != Error::unsupported_route) {
            std::fprintf(stderr, "FAIL: unknown PublicMode underlying value was accepted\n");
            return 1;
        }

        for (const auto family : {WeightFamily::h1, WeightFamily::hp1}) {
            auto invalid_backend = request(8, 8, mode, family, ResidualBackend::cpu_direct);
            invalid_backend.residual_backend = static_cast<ResidualBackend>(0xff);
            if (gate_route(invalid_backend).error != Error::unsupported_route) {
                std::fprintf(stderr,
                             "FAIL: unknown ResidualBackend underlying value was accepted\n");
                return 1;
            }
        }

        auto os           = request(8, 8, mode, WeightFamily::h1, ResidualBackend::cpu_direct);
        os.build_identity = BuildIdentity::hardware_os;
        if (!expect("OS identity", os, Error::unsupported_route, accepted, rejected))
            return 1;

        auto unsupported = request(8, 8, mode, WeightFamily::h1, ResidualBackend::cpu_direct);
        unsupported.build_identity = BuildIdentity::unsupported;
        if (!expect("unsupported build identity",
                    unsupported,
                    Error::unsupported_route,
                    accepted,
                    rejected))
            return 1;

        auto artifact_activation =
            request(8, 8, mode, WeightFamily::h1, ResidualBackend::cpu_direct);
        artifact_activation.artifact_activation_bits = 4;
        if (!expect("activation artifact mismatch",
                    artifact_activation,
                    Error::invalid_contract,
                    accepted,
                    rejected))
            return 1;

        auto artifact_weight = request(8, 8, mode, WeightFamily::h1, ResidualBackend::cpu_direct);
        artifact_weight.artifact_weight_bits = 16;
        if (!expect("weight artifact mismatch",
                    artifact_weight,
                    Error::invalid_contract,
                    accepted,
                    rejected))
            return 1;

        auto disabled        = request(8, 8, mode, WeightFamily::h1, ResidualBackend::cpu_direct);
        disabled.rmd_enabled = false;
        if (!expect("RMD disabled", disabled, Error::unsupported_route, accepted, rejected))
            return 1;
    }

    std::size_t block_accepted = 0;
    std::size_t block_rejected = 0;
    auto block  = request(8, 8, PublicMode::full, WeightFamily::h1, ResidualBackend::compact_ws);
    block.exsia = false;
    if (block.block_activation)
        return 1;
    block.rmd_enabled = false;
    if (!expect("legacy non-BLOCK RMD-off",
                block,
                h1_supported ? Error::success : Error::unsupported_route,
                block_accepted,
                block_rejected))
        return 1;
    block.rmd_enabled = true;
    if (!expect("baseline FULL RMD",
                block,
                h1_supported ? Error::success : Error::unsupported_route,
                block_accepted,
                block_rejected))
        return 1;
    block.mode = PublicMode::stripe_pipeline;
    if (!expect("baseline RMD pipeline",
                block,
                Error::unsupported_route,
                block_accepted,
                block_rejected))
        return 1;
    block.mode   = PublicMode::full;
    block.family = WeightFamily::channel;
    if (!expect("baseline channel RMD", block, Error::success, block_accepted, block_rejected))
        return 1;
    block.block_activation = true;
    if (!expect(
            "BLOCK channel RMD", block, Error::unsupported_route, block_accepted, block_rejected))
        return 1;
    block.block_activation = false;
    block.exsia            = true;
    if (!expect("ExSIA channel", block, Error::unsupported_route, block_accepted, block_rejected))
        return 1;
    block.exsia            = false;
    block.family           = WeightFamily::h1;
    block.block_activation = true;
    if (!expect("BLOCK FULL",
                block,
                h1_supported ? Error::success : Error::unsupported_route,
                block_accepted,
                block_rejected))
        return 1;
    block.mode = PublicMode::stripe_pipeline;
    if (!expect("BLOCK pipeline", block, Error::unsupported_route, block_accepted, block_rejected))
        return 1;
    block.rmd_enabled = false;
    if (!expect("BLOCK RMD-off pipeline",
                block,
                Error::unsupported_route,
                block_accepted,
                block_rejected))
        return 1;
    block.mode = PublicMode::full;
    if (!expect("BLOCK RMD-off FULL",
                block,
                h1_supported ? Error::success : Error::unsupported_route,
                block_accepted,
                block_rejected))
        return 1;
    block.rmd_enabled    = true;
    block.build_identity = BuildIdentity::hardware_os;
    if (!expect("BLOCK OS", block, Error::unsupported_route, block_accepted, block_rejected))
        return 1;
    block.build_identity = BuildIdentity::im2p_sim_ws;
    block.family         = WeightFamily::h2;
    if (!expect("BLOCK H2", block, Error::unsupported_route, block_accepted, block_rejected))
        return 1;
    block.family = WeightFamily::h0;
    if (!expect(
            "BLOCK H0 compact", block, Error::unsupported_route, block_accepted, block_rejected))
        return 1;
    std::printf("BLOCK IM2P gate: PASS accepted=%zu rejected=%zu "
                "baseline_h1_full_rmd=%s pipeline=rejected os=rejected "
                "h2=rejected h0_compact=rejected\n",
                block_accepted,
                block_rejected,
                h1_supported ? "accepted" : "rejected");

    std::printf("IM2P RMD backend gate: PASS accepted=%zu rejected=%zu "
                "strict_invalid_mode=rejected strict_invalid_backend=rejected\n",
                accepted,
                rejected);
    return 0;
}
