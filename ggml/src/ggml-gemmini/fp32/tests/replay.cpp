#include "../dump.hpp"
#include "golden.hpp"
#include "gguf.h"
#include <filesystem>
#include <fstream>
#include <map>
#include <string>

namespace {
std::vector<float> read_values(std::ifstream & in, size_t n) {
    std::vector<float> values(n);
    in.read(reinterpret_cast<char *>(values.data()), std::streamsize(n*4));
    golden_fp32::check(bool(in), "truncated dump");
    return values;
}
}
int main(int argc, char ** argv) {
    golden_fp32::check(argc == 3, "usage: golden_replay_fp32 MODEL.gguf DUMP_DIRECTORY");
    ggml_context * tensors = nullptr;
    auto * gguf = gguf_init_from_file(argv[1], {true, &tensors});
    golden_fp32::check(gguf && tensors, "GGUF metadata");
    std::ifstream model(argv[1], std::ios::binary);
    std::vector<std::filesystem::path> files;
    for (const auto & f : std::filesystem::directory_iterator(argv[2]))
        if (f.path().extension() == ".bin") files.push_back(f.path());
    std::sort(files.begin(), files.end());
    golden_fp32::check(files.size() == 24, "exactly 24 dumps required");
    std::map<std::string, std::vector<float>> cache;
    uint64_t total = 0;
    for (const auto & path : files) {
        std::ifstream in(path, std::ios::binary);
        ggml::gemmini::fp32::DumpHeader h;
        in.read(reinterpret_cast<char *>(&h), sizeof(h));
        golden_fp32::check(bool(in) && h.magic == 0x3144383233504647ull && h.role <= 2 &&
            h.order <= 2 && h.m && h.n && h.k && h.heads && (h.graph == 0 || h.graph == 1 || h.graph == 8),
            "dump header");
        std::vector<float> * weight = nullptr;
        if (h.role == 0) {
            const std::string name(h.weight, strnlen(h.weight, sizeof(h.weight)));
            const int64_t id = gguf_find_tensor(gguf, name.c_str());
            auto * tensor = ggml_get_tensor(tensors, name.c_str());
            golden_fp32::check(id >= 0 && tensor && tensor->ne[0] == int64_t(h.k) &&
                              tensor->ne[1] == int64_t(h.n), "GGUF weight identity and dimensions");
            auto it = cache.find(name);
            if (it == cache.end()) {
                model.seekg(std::streamoff(gguf_get_data_offset(gguf) + gguf_get_tensor_offset(gguf, id)));
                std::vector<float> b(h.k*h.n);
                if (tensor->type == GGML_TYPE_F16) {
                    std::vector<uint16_t> raw(h.k*h.n);
                    model.read(reinterpret_cast<char *>(raw.data()), std::streamsize(raw.size()*2));
                    for (size_t col = 0; col < h.n; ++col)
                        for (size_t z = 0; z < h.k; ++z) b[z*h.n+col] = golden_fp32::widen(raw[col*h.k+z]);
                } else {
                    golden_fp32::check(tensor->type == GGML_TYPE_F32, "GGUF FP weight type");
                    const auto raw = read_values(model, h.k*h.n);
                    for (size_t col = 0; col < h.n; ++col)
                        for (size_t z = 0; z < h.k; ++z) b[z*h.n+col] = raw[col*h.k+z];
                }
                golden_fp32::check(bool(model), "GGUF weight read");
                it = cache.emplace(name, std::move(b)).first;
            }
            weight = &it->second;
        }
        uint64_t node_values = 0;
        for (size_t head = 0; head < h.heads; ++head) {
            const auto a = read_values(in, h.m*h.k);
            const auto b = h.role ? read_values(in, h.k*h.n) : std::vector<float>{};
            const auto c = read_values(in, h.m*h.n);
            const auto stats = golden_fp32::compare(a.data(), h.role ? b.data() : weight->data(),
                                                   c.data(), h.m,h.n,h.k,unsigned(h.order));
            golden_fp32::report(path.filename().c_str(), stats);
            node_values += stats.values;
        }
        golden_fp32::check(in.peek() == EOF, "trailing dump bytes");
        total += node_values;
        std::printf("FP32_REPLAY PASS node=%s graph=%llu values=%llu\n", path.filename().c_str(),
                    (unsigned long long)h.graph, (unsigned long long)node_values);
    }
    ggml_free(tensors); gguf_free(gguf);
    std::printf("golden_replay_fp32: PASS nodes=24 values=%llu\n", (unsigned long long)total);
}
