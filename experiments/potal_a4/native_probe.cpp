#include "../../ggml/src/ggml-gemmini/residual/rmd/rmd-bitmap-builder.hpp"
#include "../../ggml/src/ggml-gemmini/residual/rmd/rmd-builder.hpp"
#include "../../ggml/src/ggml-gemmini/quants/act/buffer.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sys/resource.h>

namespace rmd = ggml::gemmini::rmd;
namespace act = ggml::gemmini::quants::act;

static void require(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

static int32_t fixture(size_t index, const std::string & kind) {
    const int32_t ordinary = static_cast<int32_t>((index * 13) % 15) - 7;
    if (kind == "narrow") return ordinary;
    if (kind == "sparse") return index % 1000 == 0 ? 127 : ordinary;
    if (kind == "dense") return index % 2 ? 127 : -128;
    if (kind == "wide") return index % 2 ? 65535 : -65536;
    if (kind == "carry") return 0x77777778;
    throw std::runtime_error("Unknown fixture");
}

static void check_packet(const rmd::StripePacket & packet, const std::vector<int32_t> & expected) {
    std::vector<int64_t> actual(expected.size(), 0);
    for (const auto & block : packet.blocks) {
        for (const auto & group : block.groups) {
            std::vector<size_t> columns;
            for (size_t k = 0; k < 32; ++k)
                if (group.k_mask & (uint32_t{1} << k)) columns.push_back(block.global_k_begin + k);
            for (size_t lp = 0; lp < group.lane_positions.size(); ++lp) {
                const auto lane = block.lane_ids[group.lane_positions[lp]];
                for (size_t r = group.row_offsets[lp]; r < group.row_offsets[lp + 1]; ++r)
                    for (size_t k = 0; k < columns.size(); ++k)
                        actual[group.row_ids[r] * packet.logical_k + columns[k]] +=
                            packet.stacked_activation.signed_int8[group.activation_offset + r * group.padded_k_count + k] *
                            (int64_t{1} << (4 * lane));
            }
        }
    }
    require(std::equal(actual.begin(), actual.end(), expected.begin()), "Native packet reconstruction mismatch");
}

int main(int argc, char ** argv) {
    try {
        require(argc == 5, "usage: native-probe clipped|direct narrow|sparse|dense|wide|carry ROWS K");
        const std::string mode = argv[1], kind = argv[2];
        require(mode == "clipped" || mode == "direct", "Unknown mode");
        const size_t rows = std::stoul(argv[3]), k = std::stoul(argv[4]);
        require(rows && k && rows <= 8192 && k <= 8192 && k % 32 == 0, "Invalid dimensions");
        const auto begin = std::chrono::steady_clock::now();
        act::QuantizedActivationBuffer main;
        require(main.allocate(rows, k, 4), "Main allocation");
        std::vector<int32_t> dense;
        if (mode == "clipped") dense.resize(rows * k);
        std::vector<rmd::StripePacketHandle> packets;
        size_t packet_bytes = 0, metadata_bytes = 0, nnz = 0;
        for (size_t first = 0; first < rows; first += 32) {
            const size_t count = std::min(size_t{32}, rows - first);
            std::vector<uint64_t> selected((count * k + 63) / 64, ~uint64_t{0});
            std::vector<int32_t> expected(count * k);
            rmd::RmdBitmapBuilder builder;
            builder.reset(first / 32, first, count, k, 4096, 4, selected, k);
            for (size_t r = 0; r < count; ++r) for (size_t col = 0; col < k; ++col) {
                const size_t index = (first + r) * k + col;
                const int32_t u = fixture(index, kind);
                rmd::NativeBalancedDigits ds;
                require(rmd::decompose_balanced_radix(u, 4, ds) == rmd::RmdStatus::success, "Digit decomposition");
                const int32_t digit = mode == "direct" ? ds.digits[0] : std::clamp(u, -8, 7);
                require(main.set(first + r, col, digit), "Main write");
                const int64_t upper = int64_t{u} - digit;
                require(upper >= INT32_MIN && upper <= INT32_MAX, "Fixture exceeds legacy scalar bridge");
                const auto residual = static_cast<int32_t>(upper);
                if (mode == "clipped") dense[index] = residual;
                expected[r * k + col] = residual;
                if (residual) require(builder.emit(r, col, residual), "Bitmap emission");
            }
            const auto packet = builder.finish();
            require(builder.status() == rmd::RmdStatus::success, "Packet finish");
            if (!packet) {
                require(std::all_of(expected.begin(), expected.end(), [](int32_t value) { return value == 0; }), "Missing nonzero packet");
                continue;
            }
            check_packet(*packet, expected);
            packet_bytes += packet->stacked_activation.signed_int8.size();
            nnz += packet->digit_nnz;
            metadata_bytes += sizeof(*packet) + packet->k_indices.size() * sizeof(uint32_t) + packet->blocks.size() * sizeof(rmd::BlockDescriptor);
            for (const auto & block : packet->blocks) for (const auto & group : block.groups)
                metadata_bytes += sizeof(group) + group.row_ids.size() * sizeof(uint16_t) + group.lane_positions.size();
            packets.push_back(packet);
        }
        rusage usage{};
        require(getrusage(RUSAGE_SELF, &usage) == 0, "getrusage");
        std::cout << "{\"mode\":\"" << mode << "\",\"fixture\":\"" << kind
                  << "\",\"rows\":" << rows << ",\"k\":" << k << ",\"dim\":" << DIM
                  << ",\"main_bytes\":" << rows * k << ",\"dense_residual_bytes\":" << dense.size() * sizeof(int32_t)
                  << ",\"packet_digit_bytes\":" << packet_bytes << ",\"packet_metadata_bytes\":" << metadata_bytes
                  << ",\"digit_nnz\":" << nnz << ",\"peak_rss_bytes\":" << usage.ru_maxrss
                  << ",\"seconds\":" << std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count()
                  << "}" << std::endl;
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
