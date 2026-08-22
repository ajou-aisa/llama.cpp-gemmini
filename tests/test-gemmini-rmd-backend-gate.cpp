#include "ggml-gemmini-im2p.hpp"

#include <array>
#include <cstdint>
#include <cstdio>

namespace {

using ggml::gemmini::im2p_adapter::Error;
using ggml::gemmini::im2p_adapter::gate_route;

struct GateCase {
    const char * name;
    bool exsia;
    std::uint8_t activation_bits;
    bool rmd_enabled;
    bool cpu_direct_rmd;
    Error expected;
};

} // namespace

int main() {
    constexpr std::array<GateCase, 11> cases{{
        {"a8-exsia-cpu-direct", true, 8, true, true, Error::success},
        {"a8-exsia-rmd-disabled", true, 8, false, true, Error::unsupported_route},
        {"a8-exsia-compact", true, 8, true, false, Error::unsupported_route},
        {"a4-exsia", true, 4, true, true, Error::unsupported_route},
        {"a16-exsia", true, 16, true, true, Error::unsupported_route},
        {"unknown-width-exsia", true, 12, true, true, Error::unsupported_route},
        {"a4-non-exsia-no-rmd", false, 4, false, false, Error::success},
        {"a8-non-exsia-no-rmd", false, 8, false, false, Error::success},
        {"a16-non-exsia-no-rmd", false, 16, false, false, Error::success},
        {"non-exsia-cpu-rmd", false, 8, true, true, Error::unsupported_route},
        {"non-exsia-compact-rmd", false, 8, true, false, Error::unsupported_route},
    }};

    for (const auto & test : cases) {
        const auto result = gate_route(
            test.exsia, test.activation_bits, test.rmd_enabled,
            test.cpu_direct_rmd);
        if (result.error != test.expected) {
            std::fprintf(stderr, "FAIL: %s returned %u, expected %u\n", test.name,
                         static_cast<unsigned>(result.error),
                         static_cast<unsigned>(test.expected));
            return 1;
        }
    }

    std::printf("IM2P RMD backend gate: PASS (%zu cases)\n", cases.size());
    return 0;
}
