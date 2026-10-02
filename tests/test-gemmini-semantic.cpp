#include <gemmini/semantic.hpp>
#include <gemmini/cpu_log_context.hpp>
#include <cassert>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace semantic = ggml::gemmini::semantic;
template<class Action> void rejected(Action action) {
    bool failed = false;
    try { action(); } catch (const std::runtime_error &) { failed = true; }
    assert(failed);
}
std::string read(const std::filesystem::path &path) {
    std::ifstream input(path);
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}
int main(int argc, char **argv) {
    assert(argc == 2);
    const std::filesystem::path root(argv[1]);
    const int32_t tokens[] = {7, 11};
    int first = 0, second = 0, unknown = 0;
    std::string first_identity, second_identity;
    for (int run = 0; run != 2; ++run) {
        const auto directory = root / std::to_string(run);
        std::filesystem::create_directories(directory);
        setenv("GEMMINI_LOG_DIR", directory.c_str(), 1);
        auto session = semantic::Session::start(semantic::Source::FullCpu, "{}", "{}", true);
        rejected([&] { session->graph({}, "[]"); });
        session->phase("prefill", std::nullopt, tokens, 2);
        assert(session->completed_graph_count() == 0);
        session->graph({{&first, "{\"op\":\"ADD\"}", false}, {&second, "{\"op\":\"ADD\"}", false}}, "[]");
        rejected([&] { session->completed_graph_count(); });
        const auto a = semantic::context_for(&first), b = semantic::context_for(&second);
        assert(a->identity.node_ordinal == 0 && b->identity.node_ordinal == 1);
        assert(a->identity.graph_occurrence == 0 && b->identity.graph_occurrence == 0);
        if (!run) { first_identity = semantic::identity_fields(a->identity); second_identity = semantic::identity_fields(b->identity); }
        else { assert(first_identity == semantic::identity_fields(a->identity)); assert(second_identity == semantic::identity_fields(b->identity)); }
        rejected([&] { semantic::context_for(&unknown); });
        {
            semantic::ScopedNode scope(&second, 3);
            assert(semantic::current_context()->identity.node_ordinal == 1);
            assert(ggml::gemmini::log::current_cpu_correlation().worker_count == 3);
        }
        assert(!semantic::current_context());
        session->execution(&first, run ? "CPU-split-a" : "CPU-combined", "ORDINARY_CPU", true);
        session->execution(&second, run ? "CPU-split-b" : "CPU-combined", "ORDINARY_CPU", true);
        assert(session->completed_graph_count() == 1);
        rejected([&] { session->execution(&first, "CPU", "ORDINARY_CPU", true); });
        rejected([&] { session->phase("prefill", std::nullopt, tokens, 2); });
        session->phase("decode", 0, tokens, 1);
        rejected([&] { semantic::context_for(&first); });
        session->graph({{&first, "{\"op\":\"ADD\"}", false}}, "[]");
        assert(semantic::context_for(&first)->identity.graph_occurrence == 0);
        session->execution(&first, "CPU", "ORDINARY_CPU", true);
        session->finish(true);
    }
    const auto contents = read(root / "0" / "semantic-graph.jsonl");
    assert(contents.find("\"expected_node_count\":3,\"executed_node_count\":3") != std::string::npos);
    assert(contents.find("\"cpu_only_proven\":true") != std::string::npos);
    assert(contents.find("0x") == std::string::npos);
    std::cout << "semantic identity: repeated operations, split-independent keys, phase ownership, exact coverage PASS\n";
}
