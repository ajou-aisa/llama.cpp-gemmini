#include "gemmini/log.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>

static std::string read_file(const std::filesystem::path & path) {
    std::ifstream input(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

static void write_file(const std::filesystem::path & path, const char * content) {
    std::ofstream(path, std::ios::binary) << content;
}

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::cerr << "usage: test-gemmini-log PATH_PREFIX\n";
        return 1;
    }

    const std::filesystem::path cycle_path = std::string(argv[1]) + ".cycle.jsonl";
    const std::filesystem::path debug_path = std::string(argv[1]) + ".debug.jsonl";
    const std::string sentinel = "sentinel\n";
    bool failed = false;

    write_file(cycle_path, sentinel.c_str());
    ggml::gemmini::log::cycle(ggml::gemmini::log::file(cycle_path.c_str()), "test", "append", 1, 2);
    if (read_file(cycle_path).rfind(sentinel, 0) != 0) {
        std::cerr << "cycle output truncated existing records\n";
        failed = true;
    }

#if !EXPECT_LOG_DEBUG
    std::filesystem::remove(debug_path);
    if (!ggml::gemmini::log::debug.set_output_path(debug_path.c_str())) {
        std::cerr << "disabled debug output setup failed\n";
        failed = true;
    }
    ggml::gemmini::log::debug.set_output(stderr);
    if (std::filesystem::exists(debug_path)) {
        std::cerr << "disabled debug output created a file\n";
        failed = true;
    }
#endif

    std::filesystem::remove(cycle_path);
    std::filesystem::remove(debug_path);
    return failed ? 2 : 0;
}
