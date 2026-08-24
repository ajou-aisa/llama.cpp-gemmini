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

constexpr std::array<std::uint8_t, 3> widths{{4, 8, 16}};
constexpr std::array<PublicMode, 2> modes{{PublicMode::full,
                                           PublicMode::stripe_pipeline}};
constexpr std::array<WeightFamily, 3> supported_families{{
    WeightFamily::h0, WeightFamily::h1, WeightFamily::hp1}};
constexpr std::array<WeightFamily, 2> deprecated_families{{
    WeightFamily::h2, WeightFamily::hp2}};
constexpr std::array<ResidualBackend, 2> residual_backends{{
    ResidualBackend::cpu_direct, ResidualBackend::compact_ws}};

bool expect(const char *name, const ExsiaRouteRequest &request, Error expected,
            std::size_t &accepted, std::size_t &rejected) {
    const auto result = gate_route(request);
    if (result.error != expected) {
        std::fprintf(stderr, "FAIL: %s returned %u, expected %u (%s)\n", name,
                     static_cast<unsigned>(result.error),
                     static_cast<unsigned>(expected), result.message);
        return false;
    }
    result.ok() ? ++accepted : ++rejected;
    return true;
}

ExsiaRouteRequest request(std::uint8_t activation_bits,
                          std::uint8_t weight_bits, PublicMode mode,
                          WeightFamily family, ResidualBackend backend) {
    ExsiaRouteRequest value{};
    value.exsia = true;
    value.activation_bits = activation_bits;
    value.weight_bits = weight_bits;
    value.artifact_activation_bits = activation_bits;
    value.artifact_weight_bits = weight_bits;
    value.rmd_enabled = true;
    value.mode = mode;
    value.family = family;
    value.residual_backend = backend;
    value.build_identity = BuildIdentity::im2p_sim_ws;
    return value;
}

} // namespace

int main() {
    std::size_t accepted = 0;
    std::size_t rejected = 0;

    // Exactly five capabilities per matched width and public mode:
    // H0 direct; H1 direct/compact; HP1 direct/compact.
    for (const auto width : widths) {
        for (const auto mode : modes) {
            for (const auto family : supported_families) {
                for (const auto backend : residual_backends) {
                    const bool supported = family != WeightFamily::h0 ||
                                           backend == ResidualBackend::cpu_direct;
                    if (!expect("matched capability", request(width, width, mode,
                                family, backend),
                                supported ? Error::success
                                          : Error::unsupported_route,
                                accepted, rejected)) {
                        return 1;
                    }
                }
            }
        }
    }
    if (accepted != 30) {
        std::fprintf(stderr, "FAIL: accepted %zu routes, expected exactly 30\n",
                     accepted);
        return 1;
    }

    // Every ordered mixed A/W pair fails for every relevant family/backend in
    // both public modes.
    for (const auto activation_bits : widths) {
        for (const auto weight_bits : widths) {
            if (activation_bits == weight_bits) continue;
            for (const auto mode : modes) {
                for (const auto family : supported_families) {
                    for (const auto backend : residual_backends) {
                        if (!expect("ordered mixed width",
                                    request(activation_bits, weight_bits, mode,
                                            family, backend),
                                    Error::unsupported_route, accepted,
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
                                Error::unsupported_route, accepted, rejected)) {
                        return 1;
                    }
                }
            }
        }
    }

    for (const auto mode : modes) {
        auto invalid_mode = request(8, 8, mode, WeightFamily::h1,
                                    ResidualBackend::cpu_direct);
        invalid_mode.mode = static_cast<PublicMode>(0xff);
        if (gate_route(invalid_mode).error != Error::unsupported_route) {
            std::fprintf(stderr,
                         "FAIL: unknown PublicMode underlying value was accepted\n");
            return 1;
        }

        for (const auto family : {WeightFamily::h1, WeightFamily::hp1}) {
            auto invalid_backend = request(8, 8, mode, family,
                                           ResidualBackend::cpu_direct);
            invalid_backend.residual_backend =
                static_cast<ResidualBackend>(0xff);
            if (gate_route(invalid_backend).error != Error::unsupported_route) {
                std::fprintf(stderr,
                             "FAIL: unknown ResidualBackend underlying value was accepted\n");
                return 1;
            }
        }

        auto os = request(8, 8, mode, WeightFamily::h1,
                          ResidualBackend::cpu_direct);
        os.build_identity = BuildIdentity::hardware_os;
        if (!expect("OS identity", os, Error::unsupported_route, accepted,
                    rejected)) return 1;

        auto unsupported = request(8, 8, mode, WeightFamily::h1,
                                   ResidualBackend::cpu_direct);
        unsupported.build_identity = BuildIdentity::unsupported;
        if (!expect("unsupported build identity", unsupported,
                    Error::unsupported_route, accepted, rejected)) return 1;

        auto artifact_activation = request(8, 8, mode, WeightFamily::h1,
                                           ResidualBackend::cpu_direct);
        artifact_activation.artifact_activation_bits = 4;
        if (!expect("activation artifact mismatch", artifact_activation,
                    Error::invalid_contract, accepted, rejected)) return 1;

        auto artifact_weight = request(8, 8, mode, WeightFamily::h1,
                                       ResidualBackend::cpu_direct);
        artifact_weight.artifact_weight_bits = 16;
        if (!expect("weight artifact mismatch", artifact_weight,
                    Error::invalid_contract, accepted, rejected)) return 1;

        auto disabled = request(8, 8, mode, WeightFamily::h1,
                                ResidualBackend::cpu_direct);
        disabled.rmd_enabled = false;
        if (!expect("RMD disabled", disabled, Error::unsupported_route,
                    accepted, rejected)) return 1;
    }

    std::printf("IM2P RMD backend gate: PASS accepted=%zu rejected=%zu "
                "strict_invalid_mode=rejected strict_invalid_backend=rejected\n",
                accepted, rejected);
    return 0;
}
