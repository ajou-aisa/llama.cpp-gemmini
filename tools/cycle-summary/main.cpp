#include <gemmini/performance.hpp>

#include <cstdio>
#include <cstring>

int main(int argc, char ** argv) {
    const bool json = argc == 3 && std::strcmp(argv[1], "--json") == 0;
    const bool help = argc == 2 && std::strcmp(argv[1], "--help") == 0;
    if (help || (argc != 2 && !json) || (!json && argv[1][0] == '-')) {
        std::fprintf(help ? stdout : stderr, "usage: %s [--json] cycle-log.jsonl\n", argv[0]);
        return help ? 0 : 2;
    }
    const auto summary = ggml::gemmini::performance::read_summary(argv[json ? 2 : 1]);
    if (json) std::printf("%s\n", summary.serialize().c_str());
    else summary.print(stdout);
    return summary.available ? 0 : 1;
}
