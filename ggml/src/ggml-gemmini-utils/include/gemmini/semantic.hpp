#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace ggml::gemmini::semantic {
enum class Source { Unspecified, FullCpu, PotalCollection };
struct Identity {
    std::string phase_kind;
    std::optional<uint64_t> decode_index;
    uint64_t graph_occurrence = 0, node_ordinal = 0;
};
struct Context {
    Identity identity;
    Source duration_source = Source::Unspecified;
    std::string run_config_id = "workload-0";
};
std::string quote(const std::string &value);
std::string identity_fields(const Identity &identity);
std::string serialize_context_fields(const Context &context);
std::shared_ptr<const Context> current_context() noexcept;
std::shared_ptr<const Context> context_for(const void *node);
const char *source_name(Source source) noexcept;
bool compiled_cpu_only_build() noexcept;

// Source keys are process-local lookup handles, never serialized identities.
struct Node {
    const void *key = nullptr;
    std::string payload_json;
    bool excluded = false;
};
class Session {
public:
    static std::shared_ptr<Session> start(Source source, const std::string &workload_json,
                                           const std::string &producer_json, bool cpu_only_build);
    ~Session() noexcept;
    void phase(const std::string &kind, std::optional<uint64_t> decode_index,
               const int32_t *tokens, size_t count);
    void graph(const std::vector<Node> &nodes, const std::string &leaves_json);
    void execution(const void *node, const std::string &backend,
                   const std::string &execution_class, bool success);
    void finish(bool success);
    void fail(std::string_view reason) noexcept;
    void ensure_healthy() const;
    Session(const Session &) = delete;
    Session &operator=(const Session &) = delete;
private:
    struct Impl;
    explicit Session(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
    friend std::shared_ptr<const Context> context_for(const void *);
};
std::shared_ptr<Session> active_session();
class ScopedNode {
public:
    explicit ScopedNode(const void *node, uint64_t workers = UINT64_MAX);
    ~ScopedNode() noexcept;
    ScopedNode(const ScopedNode &) = delete;
    ScopedNode &operator=(const ScopedNode &) = delete;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
