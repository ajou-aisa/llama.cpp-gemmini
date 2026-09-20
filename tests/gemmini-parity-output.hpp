#pragma once
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>

template <class T>
void save_gemmini_parity_output(const std::vector<T> &values, uint64_t domain, uint64_t stride) {
    const char *directory = std::getenv("GEMMINI_LOG_DIR");
    if (!directory || !*directory) throw std::runtime_error("parity output directory missing");
    const auto path = std::filesystem::path(directory) / "numerical-output.bin";
    if (std::filesystem::exists(path)) throw std::runtime_error("parity output already exists");
    std::ofstream output(path, std::ios::binary);
    const uint64_t header[] = {1, sizeof(T), domain, stride, values.size()};
    output.write(reinterpret_cast<const char *>(header), sizeof(header));
    output.write(reinterpret_cast<const char *>(values.data()), values.size() * sizeof(T));
    output.close();
    if (!output) throw std::runtime_error("parity output write failed");
}
