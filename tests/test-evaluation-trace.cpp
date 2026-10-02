#include "evaluation-trace.h"
#include <gemmini/log.hpp>
#include <cassert>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>

int main(int argc, char **argv) {
    assert(argc == 2);
    const auto directory = std::filesystem::absolute(argv[1]);
    std::filesystem::create_directories(directory);
#ifdef _WIN32
    assert(_putenv_s("GEMMINI_LOG_DIR", directory.string().c_str()) == 0);
#else
    assert(setenv("GEMMINI_LOG_DIR", directory.string().c_str(), 1) == 0);
#endif
    common_params params;
    {
        evaluation_trace trace(params, "sink-fixture", 256, 0);
        ggml::gemmini::log::cycle.write_json("{\"sink_probe\":true}");
        trace.finish(false);
    }
#if LOG_CYCLE
    std::ifstream input(directory / "cycle-log.jsonl");
    assert(input);
    const std::string bytes((std::istreambuf_iterator<char>(input)), {});
    assert(bytes.find("\"sink_probe\":true") != std::string::npos);
#else
    assert(!std::filesystem::exists(directory / "cycle-log.jsonl"));
#endif
#if CYCLE_SIM
    assert(!std::filesystem::exists(directory / "npu-cycle-trace.jsonl"));
    const auto prefill_directory = directory / "prefill";
    std::filesystem::create_directory(prefill_directory);
#ifdef _WIN32
    assert(_putenv_s("GEMMINI_LOG_DIR", prefill_directory.string().c_str()) == 0);
#else
    assert(setenv("GEMMINI_LOG_DIR", prefill_directory.string().c_str(), 1) == 0);
#endif
    evaluation_trace prefill(params, "prefill-cycle-fixture", 256, 0, false, true);
    prefill.finish(false);
    std::ifstream npu(prefill_directory / "npu-cycle-trace.jsonl");
    assert(npu);
    const std::string npu_bytes((std::istreambuf_iterator<char>(npu)), {});
    assert(npu_bytes.find("\"requested_generated_tokens\":0") != std::string::npos);
    assert(npu_bytes.find("RUN_END") != std::string::npos);
#endif
}
