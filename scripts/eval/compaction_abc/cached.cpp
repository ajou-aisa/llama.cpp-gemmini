#include "packing.hpp"
#include <iostream>
#include <memory>

namespace abc {
uint64_t median(std::vector<uint64_t> values) {
    std::sort(values.begin(),values.end());
    return values[values.size()/2];
}
std::vector<int32_t> decode_full(const Weight &weight) {
    std::vector<int32_t> result(weight.k * weight.n);
    for (size_t col = 0; col < weight.n; ++col)
        for (size_t k = 0; k < weight.k; ++k)
            result[k * weight.n + col] = weight.code(k,col);
    return result;
}
std::vector<int32_t> copy_compact(const Packed &packed, const Weight &weight,
                                  const std::vector<int32_t> &decoded) {
    std::vector<int32_t> result(packed.k * weight.n);
    for (size_t k = 0; k < packed.k; ++k)
        std::memcpy(result.data() + k * weight.n,
                    decoded.data() + packed.column_identity(k) * weight.n,
                    weight.n * sizeof(int32_t));
    asm volatile("" : : "r"(result.data()) : "memory");
    return result;
}
}

int main(int argc, char **argv) {
    using namespace abc;
    try {
        if (argc == 2 && std::string(argv[1]) == "--help") {
            std::cout << "Usage: compaction-cached DUMP_DIR OUTPUT.csv REPEATS\n"; return 0;
        }
        if (argc != 4) throw std::runtime_error("use --help for arguments");
        const std::filesystem::path root(argv[1]);
        const size_t repeats = std::stoul(argv[3]);
        if (repeats < 3) throw std::runtime_error("REPEATS must be >= 3");
        std::vector<std::filesystem::path> files;
        for (const auto &entry : std::filesystem::directory_iterator(root))
            if (entry.path().extension() == ".rbin") files.push_back(entry.path());
        std::sort(files.begin(),files.end(),[](const auto &a, const auto &b) {
            const Input left = read_input(a), right = read_input(b);
            return left.layer != right.layer ? left.layer < right.layer : std::stoul(a.stem()) < std::stoul(b.stem());
        });
        if (files.empty()) throw std::runtime_error("no real residual dumps");
        std::ofstream csv(argv[2]);
        if (!csv) throw std::runtime_error("cannot create CSV");
        csv << "case,layer,phase,bits,n,original_k,compact_k,full_cache_bytes,copy_p10_ns,copy_p50_ns,copy_p90_ns,layer_decode_ns\n";
        std::string previous_layer;
        std::unique_ptr<Weight> weight;
        std::vector<int32_t> decoded;
        uint64_t layer_decode = 0, checksum = 0;
        size_t completed = 0;
        for (const auto &file : files) {
            const Input input = read_input(file);
            if (previous_layer != input.layer) {
                weight = std::make_unique<Weight>(read_weight(root / (input.layer + ".wbin")));
                const uint64_t start = now_ns();
                decoded = decode_full(*weight);
                asm volatile("" : : "r"(decoded.data()) : "memory");
                layer_decode = now_ns() - start;
                previous_layer = input.layer;
            }
            const Packed packed = pack(input,decompose(input),Variant::A);
            std::vector<uint64_t> samples;
            for (size_t repeat = 0; repeat < repeats + 2; ++repeat) {
                const uint64_t start = now_ns();
                const auto compact = copy_compact(packed,*weight,decoded);
                const uint64_t elapsed = now_ns() - start;
                checksum += compact.size();
                if (repeat >= 2) samples.push_back(elapsed);
                if (repeat == 0)
                    for (size_t k = 0; k < packed.k; ++k)
                        for (size_t col = 0; col < weight->n; ++col)
                            if (compact[k * weight->n + col] != weight->code(packed.column_identity(k),col))
                                throw std::runtime_error("cached copy mismatch");
            }
            std::sort(samples.begin(),samples.end());
            csv << file.stem().string() << ',' << input.layer << ',' << (input.graph_m > 1 ? "prefill" : "decode") << ','
                << input.bits << ',' << input.n << ',' << input.k << ',' << packed.k << ',' << decoded.size() * sizeof(int32_t) << ','
                << samples[(samples.size()-1)/10] << ',' << median(samples) << ',' << samples[(samples.size()-1)*9/10] << ',' << layer_decode << '\n';
            if (++completed % 100 == 0) std::cerr << "processed " << completed << '/' << files.size() << " cached-copy stripes\n";
        }
        if (!checksum) throw std::runtime_error("benchmark result was discarded");
        std::cerr << "PASS: " << completed << " cached-copy stripes; every copied weight checked\n";
        return 0;
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
