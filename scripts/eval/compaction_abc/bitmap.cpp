#include "types.hpp"
#include <iostream>
#include <map>
#include <sstream>

namespace abc {
uint64_t percentile(std::vector<uint64_t> samples, double fraction) {
    std::sort(samples.begin(), samples.end());
    return samples[static_cast<size_t>((samples.size() - 1) * fraction)];
}
std::map<size_t, std::string> phases(const std::filesystem::path &path) {
    std::ifstream stream(path);
    if (!stream) throw std::runtime_error("missing passes.tsv; run label_phases.py");
    std::string line;
    std::getline(stream, line);
    std::map<size_t, std::string> result;
    while (std::getline(stream, line)) {
        std::istringstream record(line);
        size_t id, pass;
        std::string phase;
        if (!(record >> id >> pass >> phase) || (phase != "prefill" && phase != "decode"))
            throw std::runtime_error("invalid pass record");
        if (!result.emplace(id, phase).second) throw std::runtime_error("duplicate pass record");
    }
    if (result.empty()) throw std::runtime_error("empty passes.tsv");
    return result;
}
}

int main(int argc, char **argv) {
    using namespace abc;
    try {
        if (argc != 3) throw std::runtime_error("Usage: compaction-bitmap RESULTS_DIR OUTPUT.csv");
        std::ofstream csv(argv[2]);
        if (!csv) throw std::runtime_error("cannot create bitmap CSV");
        csv << "model,bits,case,phase,events,bitmap_ns,bitmap_p10_ns,bitmap_p90_ns\n";
        size_t completed = 0;
        uint64_t checksum = 0;
        for (const std::string dataset : {"gpt2", "llama", "gpt2-a8", "llama-a8"}) {
            const auto root = std::filesystem::path(argv[1]) / dataset;
            for (const auto &[id, phase] : phases(root / "passes.tsv")) {
                const auto input = read_input(root / (std::to_string(id) + ".rbin"));
                std::vector<uint64_t> samples;
                for (size_t repeat = 0; repeat < 9; ++repeat) {
                    const uint64_t start = now_ns();
                    std::vector<uint64_t> selection((input.m * input.k + 63) / 64, 0);
                    for (const Event &event : input.events) {
                        const size_t cell = event.row * input.k + event.k;
                        selection[cell / 64] |= uint64_t{1} << (cell % 64);
                    }
                    asm volatile("" : : "r"(selection.data()) : "memory");
                    const uint64_t elapsed = now_ns() - start;
                    if (repeat >= 2) samples.push_back(elapsed);
                    checksum += selection.front();
                    for (const Event &event : input.events) {
                        const size_t cell = event.row * input.k + event.k;
                        if (!(selection[cell / 64] & (uint64_t{1} << (cell % 64))))
                            throw std::runtime_error("selection omitted an event");
                    }
                }
                csv << dataset.substr(0, dataset.find('-')) << ',' << input.bits << ',' << id << ',' << phase << ','
                    << input.events.size() << ',' << percentile(samples, .5) << ','
                    << percentile(samples, .1) << ',' << percentile(samples, .9) << '\n';
                ++completed;
            }
        }
        std::cerr << "PASS: " << completed << " bitmap cases; checksum " << checksum << '\n';
        return 0;
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
