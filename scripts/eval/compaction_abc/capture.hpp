#pragma once

// Included only in an isolated evaluation copy of direct-executor.cpp.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <set>
#include <string>

namespace ggml::gemmini::residual {
inline void ablation_dump(const ggml_gemmini_args_t &args,
                          const DirectStripePayload &payload) {
    const char *directory = std::getenv("GGML_GEMMINI_ABLATION_DUMP");
    if (!directory || payload.events.empty()) return;
    static std::mutex mutex;
    static std::set<std::string> weights_written;
    static uint64_t next_id = 0;
    const std::lock_guard<std::mutex> lock(mutex);
    std::string layer = args.matmul_layer;
    for (char &c : layer) if (c == '/' || c == '\\') c = '_';
    const std::filesystem::path root(directory);
    const auto write = [](FILE *file, const void *data, size_t bytes) {
        if (std::fwrite(data, 1, bytes, file) != bytes) std::abort();
    };
    if (weights_written.insert(layer).second) {
        FILE *file = std::fopen((root / (layer + ".wbin")).c_str(), "wb");
        if (!file) std::abort();
        const uint64_t header[] = {GGML_GEMMINI_WEIGHT_BITS, args.J, args.K};
        write(file, header, sizeof(header));
#if GGML_GEMMINI_WEIGHT_BITS == 4
        write(file, args.q4_hp1_blocks, args.J * (args.K / 32) * sizeof(block_q4_hp1));
#elif GGML_GEMMINI_WEIGHT_BITS == 8
        write(file, args.q8_hp1_blocks, args.J * (args.K / 32) * sizeof(block_q8_hp1));
#else
#error "Compaction evaluation supports A4/W4 and A8/W8 HP1"
#endif
        if (std::fclose(file) != 0) std::abort();
    }
    const std::string name = std::to_string(next_id++) + ".rbin";
    FILE *file = std::fopen((root / name).c_str(), "wb");
    if (!file) std::abort();
    const uint64_t header[] = {0x524d444142430001ULL, GGML_GEMMINI_ACTIVATION_BITS,
        payload.row_count, args.J, args.K, payload.stripe_id, payload.row_begin,
        payload.events.size(), layer.size(), args.I};
    write(file, header, sizeof(header));
    write(file, layer.data(), layer.size());
    for (const ResidualEvent &event : payload.events) {
        const uint32_t coordinates[] = {static_cast<uint32_t>(event.local_row),
                                        static_cast<uint32_t>(event.original_k)};
        write(file, coordinates, sizeof(coordinates));
        write(file, &event.residual, sizeof(event.residual));
    }
    if (std::fclose(file) != 0) std::abort();
}
}
