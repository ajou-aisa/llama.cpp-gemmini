#include <gemmini/semantic.hpp>
#include <gemmini/semantic.h>
#include <gemmini/cpu_log_context.hpp>
#include <gemmini/log.hpp>
#include <cstdio>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace ggml::gemmini::semantic {
bool compiled_cpu_only_build() noexcept {
#if GEMMINI_SEMANTIC_CPU_ONLY_BUILD
    return true;
#else
    return false;
#endif
}
namespace {
std::mutex active_mutex;
std::weak_ptr<Session> active;
void require(bool condition, const char *reason) {
    if (!condition) throw std::runtime_error(std::string("semantic metadata: ") + reason);
}
std::string number(std::optional<uint64_t> value) {
    return value ? std::to_string(*value) : "null";
}
std::string phase_fields(const Identity &identity) {
    return ",\"phase_kind\":" + quote(identity.phase_kind) +
        ",\"decode_index\":" + number(identity.decode_index);
}
}
std::string quote(const std::string &value) {
    std::string out = "\"";
    constexpr char hex[] = "0123456789abcdef";
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') { out += '\\'; out += static_cast<char>(c); }
        else if (c < 32) { out += "\\u00"; out += hex[c >> 4]; out += hex[c & 15]; }
        else out += static_cast<char>(c);
    }
    return out + '"';
}
const char *source_name(Source source) noexcept {
    switch (source) {
        case Source::FullCpu: return "FULL_CPU";
        case Source::PotalCollection: return "POTAL_COLLECTION";
        case Source::Unspecified: return "OBSERVATION_ONLY";
    }
    return "OBSERVATION_ONLY";
}
std::string identity_fields(const Identity &identity) {
    return ",\"semantic_phase_kind\":" + quote(identity.phase_kind) +
        ",\"semantic_decode_index\":" + number(identity.decode_index) +
        ",\"semantic_graph_occurrence\":" + std::to_string(identity.graph_occurrence) +
        ",\"semantic_node_ordinal\":" + std::to_string(identity.node_ordinal);
}
std::string serialize_context_fields(const Context &context) {
    return identity_fields(context.identity) + ",\"run_config_id\":" + quote(context.run_config_id) +
        ",\"source_role\":" + quote(source_name(context.duration_source));
}
std::shared_ptr<const Context> current_context() noexcept {
    return log::current_cpu_correlation().semantic_context;
}
struct Session::Impl {
    struct Entry { std::shared_ptr<const Context> context; bool excluded, completed = false; };
    FILE *file = nullptr;
    mutable std::mutex mutex;
    Source source = Source::Unspecified;
    Identity phase;
    bool has_phase = false, finished = false, cpu_only_build = false, cpu_only_execution = true;
    uint64_t sequence = 0, next_graph = 0, next_decode = 0, graphs = 0, expected = 0, executed = 0;
    std::string failure;
    std::unordered_map<const void *, Entry> nodes;
    ~Impl() { if (file) std::fclose(file); }
    void check() const { require(failure.empty(), failure.c_str()); require(!finished, "run already closed"); }
    void emit(const char *kind, const std::string &fields) {
        check();
        const auto line = "{\"schema\":\"im2p-semantic-graph\",\"version\":1,\"kind\":" + quote(kind) +
            ",\"sequence\":" + std::to_string(sequence++) + fields + "}\n";
        if (std::fwrite(line.data(), 1, line.size(), file) != line.size() || std::fflush(file)) {
            failure = "metadata write failed";
            throw std::runtime_error(failure);
        }
    }
};
Session::Session(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Session::~Session() noexcept = default;
std::shared_ptr<Session> active_session() {
    std::lock_guard<std::mutex> lock(active_mutex);
    return active.lock();
}
std::shared_ptr<Session> Session::start(Source source, const std::string &workload,
                                     const std::string &producer, bool cpu_only_build) {
    std::lock_guard<std::mutex> lock(active_mutex);
    require(active.expired(), "another run is active");
    require(source != Source::FullCpu || cpu_only_build, "FullCPU role requires CPU-only build proof");
    auto impl = std::make_unique<Impl>();
    impl->source = source;
    impl->cpu_only_build = cpu_only_build;
    const auto path = log::resolve_output_path("log/semantic-graph.jsonl");
    require(!path.empty() && log::prepare_output_parent(path), "unsafe metadata path");
    impl->file = std::fopen(path.string().c_str(), "wx");
    require(impl->file != nullptr, "cannot exclusively create semantic-graph.jsonl");
    impl->emit("RUN", ",\"source_role\":" + quote(source_name(source)) +
        ",\"run_config_id\":\"workload-0\",\"workload\":" + workload + ",\"producer\":" + producer +
        ",\"cpu_only_build\":" + (cpu_only_build ? "true" : "false"));
    auto result = std::shared_ptr<Session>(new Session(std::move(impl)));
    active = result;
    return result;
}
void Session::phase(const std::string &kind, std::optional<uint64_t> decode, const int32_t *tokens, size_t count) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->check();
    require(impl_->expected == impl_->executed, "phase changed before all graph nodes completed");
    require((kind == "prefill" && !decode) || (kind == "decode" && decode), "invalid phase");
    require(!impl_->has_phase ? kind == "prefill" : kind == "decode", "repeated/missing prefill phase");
    require(kind != "decode" || *decode == impl_->next_decode, "duplicate/nonconsecutive decode phase");
    require(tokens || !count, "missing token input");
    uint64_t hash = UINT64_C(14695981039346656037);
    for (size_t i = 0; i < count; ++i) for (unsigned shift = 0; shift != 32; shift += 8) {
        hash ^= (static_cast<uint32_t>(tokens[i]) >> shift) & 255;
        hash *= UINT64_C(1099511628211);
    }
    std::ostringstream fingerprint;
    fingerprint << "fnv1a64-le-i32:" << std::hex << std::setfill('0') << std::setw(16) << hash;
    impl_->phase = {kind, decode, 0, 0};
    if (kind == "decode") ++impl_->next_decode;
    impl_->has_phase = true;
    impl_->next_graph = 0;
    impl_->nodes.clear();
    impl_->emit("PHASE", phase_fields(impl_->phase) + ",\"input_tokens\":" + std::to_string(count) +
                ",\"token_fingerprint\":" + quote(fingerprint.str()));
}
void Session::graph(const std::vector<Node> &nodes, const std::string &leaves) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->check();
    require(impl_->has_phase, "graph outside actual execution phase");
    require(impl_->expected == impl_->executed, "previous graph has incomplete nodes");
    impl_->nodes.clear();
    const auto occurrence = impl_->next_graph++;
    std::string payload = "[";
    for (size_t i = 0; i < nodes.size(); ++i) {
        auto context = std::make_shared<Context>();
        context->identity = impl_->phase;
        context->identity.graph_occurrence = occurrence;
        context->identity.node_ordinal = i;
        context->duration_source = impl_->source;
        require(nodes[i].key && impl_->nodes.emplace(nodes[i].key, Impl::Entry{context, nodes[i].excluded}).second,
                "duplicate/null original graph node");
        if (i) payload += ',';
        payload += "{\"node_ordinal\":" + std::to_string(i) + ",\"payload\":" + nodes[i].payload_json +
            ",\"excluded\":" + (nodes[i].excluded ? "true" : "false") + '}';
    }
    impl_->expected += nodes.size();
    ++impl_->graphs;
    impl_->emit("GRAPH", phase_fields(impl_->phase) + ",\"graph_occurrence\":" + std::to_string(occurrence) +
                ",\"nodes\":" + payload + "],\"leaves\":" + leaves);
}
std::shared_ptr<const Context> context_for(const void *node) {
    const auto session = active_session();
    if (!session) return {};
    std::lock_guard<std::mutex> lock(session->impl_->mutex);
    session->impl_->check();
    const auto it = session->impl_->nodes.find(node);
    require(it != session->impl_->nodes.end(), "executed node was not in original graph");
    return it->second.context;
}
void Session::execution(const void *node, const std::string &backend, const std::string &kind, bool success) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->check();
    const auto it = impl_->nodes.find(node);
    require(it != impl_->nodes.end() && !it->second.completed, "unknown or duplicate execution");
    require(kind == "ORDINARY_CPU" || kind == "TARGET_NPU" || kind == "UNSUPPORTED" || kind == "EXCLUDED",
            "invalid execution classification");
    const std::string actual = it->second.excluded ? "EXCLUDED" : kind;
    if (actual != "ORDINARY_CPU" && actual != "EXCLUDED") impl_->cpu_only_execution = false;
    impl_->emit("NODE_EXECUTION", serialize_context_fields(*it->second.context) +
        ",\"actual_backend\":" + quote(backend) + ",\"execution_class\":" + quote(actual) +
        ",\"success\":" + (success ? "true" : "false"));
    it->second.completed = true;
    ++impl_->executed;
    if (!success) impl_->failure = "backend execution failed";
}
void Session::ensure_healthy() const { std::lock_guard<std::mutex> lock(impl_->mutex); impl_->check(); }
uint64_t Session::completed_graph_count() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->check();
    require(impl_->expected == impl_->executed, "dispatch has incomplete graph nodes");
    return impl_->graphs;
}
void Session::fail(std::string_view reason) noexcept {
    try { std::lock_guard<std::mutex> lock(impl_->mutex); if (impl_->failure.empty()) impl_->failure = reason; }
    catch (...) { std::terminate(); }
}
void Session::finish(bool success) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->check();
    require(!success || impl_->expected == impl_->executed, "missing graph node executions");
    require(!success || impl_->expected != 0, "empty execution corpus");
    const bool proof = impl_->cpu_only_build && impl_->cpu_only_execution;
    require(!success || impl_->source != Source::FullCpu || proof, "FullCPU execution used a non-CPU path");
    impl_->emit("RUN_END", std::string(",\"success\":") + (success ? "true" : "false") +
        ",\"expected_node_count\":" + std::to_string(impl_->expected) + ",\"executed_node_count\":" +
        std::to_string(impl_->executed) + ",\"graph_count\":" + std::to_string(impl_->graphs) +
        ",\"cpu_only_proven\":" + (proof ? "true" : "false"));
    impl_->finished = true;
    auto *file = std::exchange(impl_->file, nullptr);
    require(std::fclose(file) == 0, "metadata close failed");
}
struct ScopedNode::Impl { log::ScopedCpuCorrelation scope; explicit Impl(log::CpuCorrelation value) : scope(std::move(value)) {} };
ScopedNode::ScopedNode(const void *node, uint64_t workers) {
    auto value = log::current_cpu_correlation();
    value.semantic_context = context_for(node);
    value.worker_count = workers;
    impl_ = std::make_unique<Impl>(std::move(value));
}
ScopedNode::~ScopedNode() noexcept = default;
}
extern "C" void *gemmini_semantic_enter(const void *node, uint64_t workers) {
    try { return new ggml::gemmini::semantic::ScopedNode(node, workers); }
    catch (const std::exception &error) {
        if (const auto session = ggml::gemmini::semantic::active_session()) session->fail(error.what());
        return nullptr;
    }
}
extern "C" void gemmini_semantic_exit(void *scope) { delete static_cast<ggml::gemmini::semantic::ScopedNode *>(scope); }
