#include <gemmini/optrace.hpp>
#include <gemmini/log.hpp>

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <tuple>
#include <utility>

#if defined(GEMMINI_OPTRACE_BUILD_CONFIG)
#include "optrace-build-config.hpp"
#endif

namespace ggml::gemmini::optrace {
namespace {
thread_local Context bound_context;
using CountKey = std::tuple<uint64_t, std::string, std::string>;
using Counts = std::map<CountKey, uint64_t>;

void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(std::string("optrace: ") + message);
}
std::string quote(const std::string &s) {
    std::string out = "\"";
    for (unsigned char c : s) {
        if (c == '"' || c == '\\') { out += '\\'; out += char(c); }
        else if (c < 0x20) {
            const char hex[] = "0123456789abcdef";
            out += "\\u00"; out += hex[c >> 4]; out += hex[c & 15];
        } else out += char(c);
    }
    return out + '"';
}
std::string number(uint64_t v) { return std::to_string(v); }
std::string number(std::optional<uint64_t> v) { return v ? number(*v) : "null"; }
std::string boolean(bool v) { return v ? "true" : "false"; }
void field(std::string &s, const char *name, const std::string &value) {
    s += ','; s += quote(name); s += ':'; s += value;
}
std::string string_map(const std::map<std::string, std::string> &values) {
    std::string out = "{";
    for (const auto &v : values) {
        if (out.size() != 1) out += ',';
        out += quote(v.first) + ':' + quote(v.second);
    }
    return out + '}';
}
bool hex_digest(const std::string &s, size_t length) {
    return s.size() == length && std::all_of(s.begin(), s.end(), [](char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
    });
}
void add_count(Counts &counts, const CountKey &key, uint64_t count) {
    auto &value = counts[key];
    require(count <= UINT64_MAX - value, "dispatch counter overflow");
    value += count;
}
std::string serialize_counts(const Counts &counts) {
    std::string out = "[";
    for (const auto &item : counts) {
        if (out.size() != 1) out += ',';
        out += "{\"phase_id\":" + number(std::get<0>(item.first));
        field(out, "layer", quote(std::get<1>(item.first)));
        field(out, "provenance", quote(std::get<2>(item.first)));
        field(out, "count", number(item.second)); out += '}';
    }
    return out + ']';
}

void validate(const RunInfo &r) {
    require(!r.run_id.empty() && !r.model.empty(), "run/model identity is required");
    require((r.activation_bits == 4 || r.activation_bits == 8) &&
            r.activation_bits == r.weight_bits &&
            (r.dim == 16 || r.dim == 32 || r.dim == 64), "unsupported profile");
    require(r.profile == "a" + number(r.activation_bits) + "w" + number(r.weight_bits) +
            "-d" + number(r.dim) + "-hp1", "profile identity mismatch");
    require(r.backend == "IM2P_SIM/GEMMINI_HP1", "only production HP1 simulator tracing is supported");
    require(!r.mode.empty(), "mode identity is required");
    require(hex_digest(r.hardware_contract_sha256, 64) &&
            hex_digest(r.runtime_manifest_sha256, 64), "hardware/runtime build contract is required");
    require(r.source_commits.size() == 3 && r.source_worktree_sha256.size() == 3,
            "three source identities are required");
    for (const char *name : {"IM2P.sim", "llama.cpp-gemmini", "headers"}) {
        auto commit = r.source_commits.find(name), digest = r.source_worktree_sha256.find(name);
        require(commit != r.source_commits.end() && hex_digest(commit->second, 40), "invalid source commit");
        require(digest != r.source_worktree_sha256.end() && hex_digest(digest->second, 64), "invalid source digest");
    }
}
void validate(const Work &w, const RunInfo &r) {
    require(!w.layer.empty() && !w.operation.empty(), "work layer/operation is required");
    require(w.activation_bits == r.activation_bits && w.weight_bits == r.weight_bits && w.dim == r.dim,
            "work profile mismatch");
    require(w.m && w.n && w.k && w.tile_i_count && w.tile_j_count && w.tile_k_count,
            "missing or zero shape/final tile factor");
    require(w.tile_i_count <= UINT16_MAX / w.dim && w.tile_j_count <= UINT16_MAX / w.dim &&
            w.tile_k_count <= UINT32_MAX / w.dim, "tile factor exceeds hardware domain");
    require(w.m == w.row_count && w.row_begin <= w.geometry_m &&
            w.row_count <= w.geometry_m - w.row_begin, "invalid accepted row range");
    require(w.activation_stride_bytes >= w.k && w.weight_stride_bytes >= w.n &&
            w.n <= UINT64_MAX / 4 && w.output_stride_bytes >= w.n * 4 &&
            w.output_stride_bytes % 4 == 0 && w.scale_stride_elements >= w.n,
            "invalid descriptor strides");
    require(w.block_size == 32 && w.vector_op == 5 && w.output_domain == 2 &&
            w.production_geometry_version == 1, "non-HP1 or unsupported production contract");
    require(!w.rmd_raw && !w.host_integer_block_multiply, "raw/host-integer residual paths are not production HP1");
    require(!w.host_slot || *w.host_slot < 2, "host slot must be 0 or 1");
    if (w.provenance == "residual") {
        require(r.residual_enabled, "residual work in residual-disabled run");
        require(w.scope == "residual_compact" && w.k <= 32 && w.original_block_id.has_value(),
                "residual needs compact geometry and original block provenance");
    } else {
        require(w.provenance == "dense_main" && !w.original_block_id,
                "invalid dense provenance");
        require(w.scope == "full" || w.scope == "stripe", "invalid dense scope");
    }
    if (w.scope == "stripe") require(w.stripe_id && w.host_slot && *w.host_slot < 2,
                                      "stripe identity/host slot 0 or 1 required");
    else require(w.row_begin == 0 && w.geometry_m == w.m, "non-stripe geometry must describe full compact work");
}
} // namespace

struct Session::Impl {
    struct Parent {
        Work descriptor;
        uint64_t next_row = 0, next_stripe = 0, works = 0;
    };
    RunInfo info;
    FILE *file = nullptr;
    std::mutex mutex;
    uint64_t sequence = 0, work_count = 0, phase_id = 0, next_decode = 0;
    bool has_phase = false, closed = false, io_failed = false;
    Counts trace_counts, independent_counts;
    uint64_t next_parent = 0;
    std::map<uint64_t, Parent> parents;

    ~Impl() { if (file) std::fclose(file); }
    std::string base(const char *kind) const {
        return "{\"kind\":" + quote(kind) + ",\"sequence\":" + number(sequence) +
            ",\"run_id\":" + quote(info.run_id);
    }
    void emit(std::string value) {
        require(!closed && !io_failed && file, "output is closed or failed");
        require(sequence != UINT64_MAX, "sequence overflow");
        value += "}\n";
        if (std::fwrite(value.data(), 1, value.size(), file) != value.size() || std::fflush(file) != 0) {
            io_failed = true;
            throw std::runtime_error(std::string("optrace: write/flush failed: ") + std::strerror(errno));
        }
        ++sequence;
    }
};

Context current_context() { return bound_context; }
ScopedContext::ScopedContext(Context context) : previous_(std::move(bound_context)) {
    bound_context = std::move(context);
}
ScopedContext::~ScopedContext() noexcept { bound_context = std::move(previous_); }

Session::Session(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
std::shared_ptr<Session> Session::start(const char *path, const RunInfo &info) {
    if (!path || !*path) return {};
#if CYCLE_SIM
    throw std::runtime_error("optrace: CPU-functional collection cannot claim production RTL acceptance");
#endif
    validate(info);
    auto impl = std::make_unique<Impl>();
    impl->info = info;
    const auto resolved = log::resolve_output_path(path);
    require(!resolved.empty() && log::prepare_output_parent(resolved), "invalid output path/parent");
    // Unlike best-effort cycle telemetry, provenance may not silently append,
    // replace an old run, or disable itself after a write failure.
    impl->file = std::fopen(resolved.string().c_str(), "wx");
    if (!impl->file) throw std::runtime_error(std::string("optrace: exclusive open failed: ") + std::strerror(errno));
    auto s = impl->base("run");
    field(s, "schema", quote("im2p-production-optrace")); field(s, "version", "2");
    field(s, "model", quote(info.model)); field(s, "profile", quote(info.profile));
    field(s, "activation_bits", number(info.activation_bits)); field(s, "weight_bits", number(info.weight_bits));
    field(s, "dim", number(info.dim)); field(s, "backend", quote(info.backend)); field(s, "mode", quote(info.mode));
    field(s, "residual_enabled", boolean(info.residual_enabled));
    field(s, "prompt_tokens", number(info.prompt_tokens));
    field(s, "requested_generated_tokens", number(info.requested_generated_tokens));
    field(s, "source_commits", string_map(info.source_commits));
    field(s, "source_worktree_sha256", string_map(info.source_worktree_sha256));
    field(s, "hardware_contract_sha256", quote(info.hardware_contract_sha256));
    field(s, "runtime_manifest_sha256", quote(info.runtime_manifest_sha256));
    impl->emit(std::move(s));
    return std::shared_ptr<Session>(new Session(std::move(impl)));
}
Session::~Session() noexcept {
    try { if (!impl_->closed) finish(false, "session ended without successful finalization"); }
    catch (...) { /* The triggering I/O error is reported to the caller. */ }
}
Context Session::phase(const std::string &kind, std::optional<uint64_t> decode_index,
                       uint64_t input_tokens) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    auto &x = *impl_;
    require(!x.closed, "phase after run end");
    require(x.parents.empty(), "phase changed with incomplete parent");
    if (!x.has_phase) require(kind == "prefill" && !decode_index, "first phase must be prefill");
    else require(kind == "decode" && decode_index && *decode_index == x.next_decode,
                 "decode phases must be contiguous from zero");
    const uint64_t id = x.has_phase ? x.phase_id + 1 : 0;
    auto s = x.base("phase"); field(s, "phase_id", number(id));
    field(s, "phase_kind", quote(kind)); field(s, "decode_index", number(decode_index));
    field(s, "input_tokens", number(input_tokens)); x.emit(std::move(s));
    x.phase_id = id; x.has_phase = true;
    if (decode_index) ++x.next_decode;
    return {shared_from_this(), id, std::nullopt};
}
Context Session::parent_begin(const Context &phase, const Work &w) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    auto &x = *impl_;
    require(!x.closed && phase.session.get() == this && x.has_phase &&
            phase.phase_id == x.phase_id && !phase.parent_invocation_id,
            "parent belongs to missing/stale/foreign phase");
    Work full = w;
    if (full.scope == "stripe") full.scope = "full";
    validate(full, x.info);
    require(w.row_begin == 0 && w.m == w.geometry_m && w.row_count == w.m,
            "parent descriptor must contain complete shape");
    require(x.next_parent != UINT64_MAX, "parent identity overflow");
    const uint64_t id = x.next_parent++;
    auto s = x.base("parent_begin");
    field(s, "phase_id", number(phase.phase_id));
    field(s, "parent_invocation_id", number(id));
    field(s, "layer", quote(w.layer)); field(s, "operation", quote(w.operation));
    field(s, "provenance", quote(w.provenance)); field(s, "scope", quote(w.scope));
#define NUM(name) field(s, #name, number(w.name))
    NUM(activation_bits); NUM(weight_bits); NUM(dim); NUM(m); NUM(n); NUM(k);
    NUM(tile_i_count); NUM(tile_j_count); NUM(tile_k_count);
    NUM(activation_stride_bytes); NUM(weight_stride_bytes); NUM(output_stride_bytes); NUM(scale_stride_elements);
    NUM(block_size); NUM(vector_op); NUM(output_domain); NUM(production_geometry_version);
#undef NUM
    x.emit(std::move(s));
    x.parents.emplace(id, Impl::Parent{w});
    return {shared_from_this(), phase.phase_id, id};
}
void Session::parent_end(const Context &parent) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    auto &x = *impl_;
    require(parent.session.get() == this && parent.phase_id == x.phase_id &&
            parent.parent_invocation_id, "invalid parent completion context");
    const auto found = x.parents.find(*parent.parent_invocation_id);
    require(found != x.parents.end(), "parent missing or already completed");
    const auto &p = found->second;
    require(p.works && p.next_row == p.descriptor.m,
            "parent completed with incomplete row coverage");
    auto s = x.base("parent_end");
    field(s, "phase_id", number(parent.phase_id));
    field(s, "parent_invocation_id", number(parent.parent_invocation_id));
    field(s, "status", quote("success"));
    x.emit(std::move(s));
    x.parents.erase(found);
}
void Session::accepted(const Context &context, const Work &w) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    auto &x = *impl_;
    require(context.session.get() == this && x.has_phase && context.phase_id == x.phase_id,
            "work belongs to missing/stale/foreign phase");
    validate(w, x.info);
    require(context.parent_invocation_id.has_value(), "work has no declared parent");
    const auto found = x.parents.find(*context.parent_invocation_id);
    require(found != x.parents.end(), "work parent missing or completed");
    auto &p = found->second;
    const auto &d = p.descriptor;
    require(w.layer == d.layer && w.operation == d.operation && w.provenance == d.provenance &&
            w.scope == d.scope && w.geometry_m == d.m && w.n == d.n && w.k == d.k &&
            w.activation_stride_bytes == d.activation_stride_bytes &&
            w.weight_stride_bytes == d.weight_stride_bytes &&
            w.output_stride_bytes == d.output_stride_bytes &&
            w.scale_stride_elements == d.scale_stride_elements,
            "work differs from declared parent descriptor/geometry");
    require(w.scope == "stripe" || (w.tile_i_count == d.tile_i_count &&
            w.tile_j_count == d.tile_j_count && w.tile_k_count == d.tile_k_count),
            "FULL/compact work differs from parent final tile geometry");
    require(w.row_begin == p.next_row && (w.scope == "stripe"
            ? w.stripe_id == p.next_stripe : p.works == 0),
            "duplicate, overlapping, gapped, or unordered parent work");
    auto s = x.base("npu_work"); field(s, "phase_id", number(context.phase_id));
    field(s, "parent_invocation_id", number(context.parent_invocation_id));
    field(s, "layer", quote(w.layer)); field(s, "operation", quote(w.operation));
    field(s, "provenance", quote(w.provenance)); field(s, "numerical_datapath", quote("hp1_scu"));
    field(s, "scope", quote(w.scope));
#define NUM(name) field(s, #name, number(w.name))
    NUM(activation_bits); NUM(weight_bits); NUM(dim); NUM(m); NUM(n); NUM(k);
    NUM(tile_i_count); NUM(tile_j_count); NUM(tile_k_count); NUM(geometry_m);
    NUM(row_begin); NUM(row_count); NUM(stripe_id); NUM(host_slot); NUM(original_block_id);
    NUM(activation_stride_bytes); NUM(weight_stride_bytes); NUM(output_stride_bytes); NUM(scale_stride_elements);
    NUM(block_size); NUM(vector_op); NUM(output_domain); NUM(production_geometry_version);
    NUM(work_context); NUM(source_row_begin); NUM(source_row_count); NUM(column_begin); NUM(group_index);
#undef NUM
    field(s, "logical_work_id", number(x.sequence));
    field(s, "rmd_raw", boolean(w.rmd_raw)); field(s, "host_integer_block_multiply", boolean(w.host_integer_block_multiply));
    x.emit(std::move(s));
    p.next_row += w.row_count; ++p.next_stripe; ++p.works;
    add_count(x.trace_counts, {context.phase_id, w.layer, w.provenance}, 1);
    ++x.work_count;
}
void Session::independent_count(const Context &context, const std::string &layer,
                                const std::string &provenance, uint64_t count) {
    if (!count) return;
    std::lock_guard<std::mutex> lock(impl_->mutex);
    require(!impl_->closed && context.session.get() == this && impl_->has_phase &&
            context.phase_id == impl_->phase_id, "counter belongs to missing/stale/foreign phase");
    require(!layer.empty() && (provenance == "dense_main" || provenance == "residual"), "invalid independent counter key");
    add_count(impl_->independent_counts, {context.phase_id, layer, provenance}, count);
}
void Session::finish(bool success, const std::string &reason) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    auto &x = *impl_;
    require(!x.closed, "run already finalized");
    const bool counts_match = x.trace_counts == x.independent_counts;
    const bool valid = success && counts_match && x.has_phase && !x.io_failed && x.parents.empty();
    auto s = x.base("run_end"); field(s, "status", quote(valid ? "success" : "failed"));
    field(s, "reason", quote(success && !counts_match ? "independent accepted-dispatch count mismatch" :
                            success && !x.parents.empty() ? "incomplete parent invocation" : reason));
    field(s, "work_count", number(x.work_count));
    field(s, "independent_counts", serialize_counts(x.independent_counts));
    x.emit(std::move(s));
    x.closed = true;
    FILE *file = std::exchange(x.file, nullptr);
    require(std::fclose(file) == 0, "close failed");
    require(!success || valid, "run failed final phase/count integrity check");
}

RunInfo compiled_run_info(const std::string &model, uint64_t prompt_tokens,
                          uint64_t requested_generated_tokens) {
    RunInfo r;
#if defined(GEMMINI_OPTRACE_BUILD_CONFIG)
    fill_compiled_optrace_info(r);
#else
    throw std::runtime_error("optrace: compiled source/profile identity unavailable");
#endif
    r.model = model; r.prompt_tokens = prompt_tokens;
    r.requested_generated_tokens = requested_generated_tokens;
    return r;
}
} // namespace ggml::gemmini::optrace
