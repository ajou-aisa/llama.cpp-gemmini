#pragma once
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace abc {
struct Event { uint32_t row, k; int32_t residual; };
static_assert(sizeof(Event) == 12);
struct Input {
    uint64_t bits, m, n, k, stripe, row_begin, graph_m;
    std::string layer;
    std::vector<Event> events;
};
struct Weight {
    uint64_t bits, n, k;
    std::vector<uint8_t> bytes;
    std::vector<uint64_t> block_bounds;
    size_t block_bytes() const { return bits == 4 ? 24 : 40; }
    int code(size_t k_index, size_t col) const {
        const size_t block = (col * (k / 32) + k_index / 32) * block_bytes();
        const size_t local = k_index % 32;
        if (bits == 4) return ((bytes[block + local % 16] >> (local < 16 ? 0 : 4)) & 15) - 8;
        return static_cast<int8_t>(bytes[block + local]);
    }
    int16_t exponent(size_t block, size_t col) const {
        int16_t value;
        std::memcpy(&value, bytes.data() + (col * (k / 32) + block) * block_bytes() + (bits == 4 ? 16 : 32), 2);
        return value;
    }
};
struct Digit { uint32_t row, k; uint8_t lane; int8_t value; };
struct Digits { size_t lanes = 0; std::vector<Digit> entries; };
struct Run { uint32_t block, mask, begin, count; };
enum class Variant { A, B, C };
inline char name(Variant v) { return static_cast<char>('A' + static_cast<int>(v)); }
struct Packed {
    size_t m = 0, k = 0, lanes = 0, bits = 4, array_dim = 16;
    Variant variant = Variant::A;
    std::vector<size_t> rows, columns;
    std::vector<int8_t> a;
    std::vector<int32_t> w;
    std::vector<Run> runs;
    size_t row_identity(size_t index) const { return variant == Variant::C ? index : rows[index]; }
    size_t column_identity(size_t index) const { return variant == Variant::A ? columns[index] : index; }
};
inline uint64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
inline void read_bytes(std::ifstream &file, void *destination, size_t bytes) {
    if (!file.read(static_cast<char *>(destination), bytes)) throw std::runtime_error("truncated dump");
}
inline Input read_input(const std::filesystem::path &path) {
    std::ifstream file(path, std::ios::binary);
    std::array<uint64_t, 10> h{};
    read_bytes(file, h.data(), sizeof(h));
    if (h[0] != 0x524d444142430001ULL || (h[1] != 4 && h[1] != 8) || !h[2] || !h[3] || !h[4] || h[4] % 32 || h[8] > 4096)
        throw std::runtime_error("invalid residual header");
    Input input{h[1],h[2],h[3],h[4],h[5],h[6],h[9],std::string(h[8], '\0'),std::vector<Event>(h[7])};
    read_bytes(file, input.layer.data(), input.layer.size());
    read_bytes(file, input.events.data(), input.events.size() * sizeof(Event));
    for (const Event &event : input.events)
        if (event.row >= input.m || event.k >= input.k || !event.residual) throw std::runtime_error("invalid residual event");
    return input;
}
inline Weight read_weight(const std::filesystem::path &path) {
    std::ifstream file(path, std::ios::binary);
    std::array<uint64_t, 3> h{};
    read_bytes(file, h.data(), sizeof(h));
    if ((h[0] != 4 && h[0] != 8) || !h[1] || !h[2] || h[2] % 32) throw std::runtime_error("invalid weight header");
    Weight weight{h[0],h[1],h[2],{},std::vector<uint64_t>(h[2]/32,0)};
    weight.bytes.resize(h[1] * (h[2] / 32) * weight.block_bytes());
    read_bytes(file, weight.bytes.data(), weight.bytes.size());
    for (size_t block = 0; block < weight.k / 32; ++block)
        for (size_t col = 0; col < weight.n; ++col) {
            const int16_t exponent = weight.exponent(block,col);
            if (exponent == INT16_MIN) continue;
            if (exponent < 0 || exponent > 31) throw std::runtime_error("weight exponent unsupported by reference bound");
            weight.block_bounds[block] = std::max(weight.block_bounds[block], (uint64_t{1} << (weight.bits - 1)) * (uint64_t{1} << exponent));
        }
    return weight;
}
inline Digits decompose(const Input &input) {
    Digits result;
    result.entries.reserve(input.events.size() * 3);
    const int64_t base = int64_t{1} << input.bits;
    for (const Event &event : input.events) {
        int64_t quotient = event.residual;
        uint8_t lane = 0;
        while (quotient) {
            int64_t digit = static_cast<uint64_t>(quotient) & (base - 1);
            if (digit >= base / 2) digit -= base;
            quotient = (quotient - digit) / base;
            if (digit) result.entries.push_back({event.row,event.k,lane,static_cast<int8_t>(digit)});
            ++lane;
        }
        result.lanes = std::max(result.lanes, size_t{lane});
    }
    return result;
}
}
