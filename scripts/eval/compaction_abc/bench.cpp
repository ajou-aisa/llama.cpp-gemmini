#include "production.hpp"
#include "device.hpp"
#include "numerical.hpp"
#include <iostream>
#include <map>
#include <random>

namespace abc {
struct Samples { std::vector<uint64_t> decompose, pack, gather, restore, total; };
template<class T> void materialize(const std::vector<T> &buffer) {
    // The opaque memory use keeps every generated element observable to the optimizer.
    asm volatile("" : : "r"(buffer.data()) : "memory");
}
uint64_t percentile(std::vector<uint64_t> values, double fraction) {
    std::sort(values.begin(), values.end());
    return values[static_cast<size_t>((values.size() - 1) * fraction)];
}
std::string geometry_key(const Input &input, const Packed &p, size_t dim) {
    std::string key = std::to_string(dim) + ":" + std::to_string(input.bits) + ":" +
        std::to_string(p.m) + ":" + std::to_string(input.n) + ":" + std::to_string(p.k) + ":" + std::to_string(input.k);
    // Mask positions are admission metadata; timing uses block identity and compact count.
    for (const Run &run : p.runs) key += ":" + std::to_string(run.block) + "." + std::to_string(run.count);
    return key;
}
void compare_production(const ggml::gemmini::rmd::RunAwareRequest &actual, const Packed &expected) {
    if (actual.m != expected.m || actual.k != expected.k || actual.activations != expected.a || actual.weights != expected.w || actual.runs.size() != expected.runs.size())
        throw std::runtime_error("simple A disagrees with current production arrays");
    for (size_t r = 0; r < actual.rows.size(); ++r)
        if (size_t{actual.rows[r].original_lane_id} * actual.source_row_count + actual.rows[r].source_row != expected.rows[r])
            throw std::runtime_error("production row identity mismatch");
    for (size_t r = 0; r < actual.runs.size(); ++r)
        if (actual.runs[r].original_block_id != expected.runs[r].block || actual.runs[r].union_k_mask != expected.runs[r].mask)
            throw std::runtime_error("production K-run mismatch");
}
}

int main(int argc, char **argv) {
    using namespace abc;
    try {
        if (argc == 2 && std::string(argv[1]) == "--self-test") {
            self_test(); std::cout << "PASS: saturation boundary and INT32 balanced carry\n"; return 0;
        }
        if (argc == 2 && std::string(argv[1]) == "--help") {
            std::cout << "Usage: compaction-abc DUMP_DIR OUTPUT.csv DIM REPEATS [MAX_CASES]\n"; return 0;
        }
        if (argc < 5 || argc > 6) throw std::runtime_error("use --help for arguments");
        const std::filesystem::path root(argv[1]);
        const size_t dim = std::stoul(argv[3]), repeats = std::stoul(argv[4]);
        const size_t limit = argc == 6 ? std::stoul(argv[5]) : SIZE_MAX;
        if ((dim != 16 && dim != 64) || repeats < 3) throw std::runtime_error("DIM must be 16/64 and REPEATS >= 3");
        std::vector<std::filesystem::path> files;
        for (const auto &entry : std::filesystem::directory_iterator(root)) if (entry.path().extension() == ".rbin") files.push_back(entry.path());
        std::sort(files.begin(),files.end(),[](const auto &a, const auto &b) { return std::stoul(a.stem()) < std::stoul(b.stem()); });
        if (files.empty()) throw std::runtime_error("no real residual dumps");
        std::ofstream csv(argv[2]);
        std::ofstream geometry(std::string(argv[2]) + ".geometry.tsv");
        const bool cpu_only = std::getenv("ABC_CPU_ONLY") != nullptr;
        if (!csv) throw std::runtime_error("cannot create CSV");
        csv << "case,layer,phase,bits,dim,source_m,graph_m,n,original_k,lanes,variant,m,k,runs,residual_nnz,digit_nnz,logical_macs,padded_macs,decompose_ns,pack_ns,gather_ns,restore_ns,total_ns,total_p10_ns,total_p90_ns,production_prepare_ns,device_cycles,fragments,loads,stores,scales,numeric_mismatches,numeric_max_error,saturations,all_column_bound,cycle_key\n";
        std::map<std::string,DeviceResult> cycle_cache;
        std::mt19937 random(20261004);
        std::string previous_layer;
        size_t completed = 0;
        // Process one layer at a time so its real quantized weights are loaded once.
        std::stable_sort(files.begin(),files.end(),[](const auto &a, const auto &b) { return read_input(a).layer < read_input(b).layer; });
        std::unique_ptr<Weight> weight;
        std::unique_ptr<ProductionFixture> fixture;
        for (const auto &file : files) {
            if (completed == limit) break;
            const Input input = read_input(file);
            if (input.bits != GGML_GEMMINI_ACTIVATION_BITS) throw std::runtime_error("dump precision does not match executable");
            if (previous_layer != input.layer) {
                weight = std::make_unique<Weight>(read_weight(root / (input.layer + ".wbin")));
                fixture = std::make_unique<ProductionFixture>(*weight);
                previous_layer = input.layer;
            }
            const Digits digits = decompose(input);
            const uint64_t bound = all_column_bound(input,digits,*weight);
            std::array<Packed,3> prepared;
            std::array<Check,3> checks;
            std::array<std::vector<int32_t>,3> outputs;
            std::array<Samples,3> samples;
            for (size_t v = 0; v < 3; ++v) {
                prepared[v] = pack(input,digits,static_cast<Variant>(v));
                prepared[v].array_dim = dim;
                gather_weights(prepared[v],*weight,dim);
                checks[v] = numerical_check(input,prepared[v],*weight);
                outputs[v].resize(prepared[v].m * input.n, 1);
            }
            compare_production(fixture->prepare(input),prepared[0]);
            std::vector<uint64_t> production_samples;
            uint64_t checksum = 0;
            for (size_t repeat = 0; repeat < repeats + 2; ++repeat) {
                std::array<size_t,4> order{0,1,2,3};
                std::shuffle(order.begin(),order.end(),random);
                for (size_t v : order) {
                    if (v == 3) {
                        const uint64_t start = now_ns();
                        const auto request = fixture->prepare(input);
                        materialize(request.activations); materialize(request.weights);
                        const uint64_t elapsed = now_ns() - start;
                        checksum += request.activations.size() + request.weights.size();
                        if (repeat >= 2) production_samples.push_back(elapsed);
                        continue;
                    }
                    const uint64_t t0 = now_ns();
                    const auto d = decompose(input);
                    materialize(d.entries);
                    const uint64_t t1 = now_ns();
                    auto p = pack(input,d,static_cast<Variant>(v));
                    materialize(p.a);
                    const uint64_t t2 = now_ns();
                    gather_weights(p,*weight,dim);
                    materialize(p.w);
                    const uint64_t t3 = now_ns();
                    const auto restored = restore(input,p,outputs[v]);
                    materialize(restored);
                    const uint64_t t4 = now_ns();
                    checksum += restored.front() + p.a.size() + p.w.size();
                    if (repeat >= 2) {
                        samples[v].decompose.push_back(t1-t0); samples[v].pack.push_back(t2-t1);
                        samples[v].gather.push_back(t3-t2); samples[v].restore.push_back(t4-t3); samples[v].total.push_back(t4-t0);
                    }
                }
            }
            if (!checksum) throw std::runtime_error("benchmark result was discarded");
            for (size_t v = 0; v < 3; ++v) {
                const Packed &p = prepared[v];
                const auto key = geometry_key(input,p,dim);
                auto found = cycle_cache.find(key);
                if (found == cycle_cache.end()) {
                    geometry << key << ' ' << input.bits << ' ' << dim << ' ' << p.m << ' ' << input.n << ' ' << p.k << ' ' << input.k << ' ' << p.runs.size();
                    for (const Run &run : p.runs) geometry << ' ' << run.block << ' ' << run.mask << ' ' << run.begin << ' ' << run.count;
                    geometry << '\n';
                    const DeviceResult value = cpu_only ? DeviceResult{} : estimate(input,p,dim);
                    found = cycle_cache.emplace(key,value).first;
                }
                const auto &device = found->second;
                uint64_t padded_k = 0;
                for (const Run &run : p.runs) padded_k += ((run.count + dim - 1) / dim) * dim;
                const uint64_t padded_m = ((p.m + dim - 1) / dim) * dim, padded_n = ((input.n + dim - 1) / dim) * dim;
                csv << file.stem().string() << ',' << input.layer << ',' << (input.graph_m > 1 ? "prefill" : "decode") << ','
                    << input.bits << ',' << dim << ',' << input.m << ',' << input.graph_m << ',' << input.n << ',' << input.k << ',' << digits.lanes << ','
                    << name(p.variant) << ',' << p.m << ',' << p.k << ',' << p.runs.size() << ',' << input.events.size() << ',' << digits.entries.size() << ','
                    << p.m * input.n * p.k << ',' << padded_m * padded_n * padded_k << ','
                    << percentile(samples[v].decompose,.5) << ',' << percentile(samples[v].pack,.5) << ',' << percentile(samples[v].gather,.5) << ','
                    << percentile(samples[v].restore,.5) << ',' << percentile(samples[v].total,.5) << ',' << percentile(samples[v].total,.1) << ','
                    << percentile(samples[v].total,.9) << ',' << percentile(production_samples,.5) << ',' << device.cycles << ',' << device.fragments << ','
                    << device.loads << ',' << device.stores << ',' << device.scales << ',' << checks[v].mismatches << ',' << checks[v].max_error << ',' << checks[v].saturations << ',' << bound << ',' << key << '\n';
            }
            csv.flush();
            if (++completed % 25 == 0) std::cerr << "processed " << completed << '/' << std::min(files.size(),limit) << " stripes; cycle geometries " << cycle_cache.size() << '\n';
        }
        std::cerr << "PASS: " << completed << " real stripes; production arrays verified; " << cycle_cache.size() << " cycle geometries\n";
        return 0;
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
