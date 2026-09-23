#include <gemmini/evaluation_metrics.hpp>
#include <gemmini/log.hpp>
#include <gemmini/semantic.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>
#include <utility>

namespace ggml::gemmini::evaluation {
namespace {
std::mutex active_mutex;
std::weak_ptr<Session> active;
void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(std::string("evaluation metrics: ") + message);
}
std::string field(const char *name, uint64_t value) {
    return ",\"" + std::string(name) + "\":" + std::to_string(value);
}
std::string text_field(const char *name, const std::string &value) {
    return ",\"" + std::string(name) + "\":" + semantic::quote(value);
}
}

struct Session::Impl {
    Config config;
    FILE *activation = nullptr, *residual = nullptr;
    mutable std::mutex mutex;
    uint64_t chunk = 0, next_invocation = 0, next_record = 0, completed_quantization = 0;
    bool chunk_set = false, finished = false, reference_complete = true;
    std::string failure;
    ~Impl() {
        if (activation) std::fclose(activation);
        if (residual) std::fclose(residual);
    }
    void check() const {
        require(failure.empty(), failure.c_str());
        require(!finished, "session already finished");
    }
    void emit(FILE *file, const char *kind, const std::string &fields) {
        check();
        if (!file) return;
        const std::string schema = file == activation
            ? "im2p-activation-quant-metrics" : "im2p-residual-path-metrics";
        const auto line = "{\"schema\":" + semantic::quote(schema) +
            ",\"version\":1" + text_field("kind", kind) +
            field("sequence", next_record++) + text_field("run_id", config.run_id) +
            text_field("workload_id", config.workload_id) + fields + "}\n";
        if (std::fwrite(line.data(), 1, line.size(), file) != line.size() || std::fflush(file)) {
            failure = "sink write failed";
            throw std::runtime_error("evaluation metrics: " + failure);
        }
    }
};
struct Invocation::Impl {
    std::shared_ptr<Session> session;
    std::string identity;
    size_t m = 0, k = 0;
    std::mutex mutex;
    std::vector<uint8_t> fp_selected, observed;
    std::set<std::pair<size_t, size_t>> requantized;
    std::map<size_t, std::pair<size_t, size_t>> stripes;
    std::set<size_t> compact_stripes;
    uint64_t finite = 0, fp_count = 0, selected = 0, intersection = 0, residual = 0;
    uint64_t requant_events = 0, observed_count = 0;
    bool activation_finished = false, reference_complete = false;
};

Session::Session(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Session::~Session() = default;
Invocation::Invocation(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Invocation::~Invocation() = default;
std::shared_ptr<Session> active_session() {
#if GGML_GEMMINI_ACT_QUANT_METRICS || GGML_GEMMINI_RESIDUAL_METRICS
    std::lock_guard<std::mutex> lock(active_mutex);
    return active.lock();
#else
    return {};
#endif
}
std::shared_ptr<Session> Session::start(const Config &config) {
    require(!config.activation_reference_candidate,
            "invocation-wide ACT reference removed; signed row/B_K=32 reference is authoritative");
    require(GGML_GEMMINI_ACT_QUANT_METRICS || config.activation_path.empty(),
            "ACT metrics compiled out");
    require(GGML_GEMMINI_RESIDUAL_METRICS || config.residual_path.empty(),
            "RES metrics compiled out");
    if (config.activation_path.empty() && config.residual_path.empty()) return {};
    require(!config.run_id.empty() && !config.workload_id.empty(), "missing run/workload identity");
    std::lock_guard<std::mutex> lock(active_mutex);
    require(active.expired(), "another metrics session is active");
    auto impl = std::make_unique<Impl>();
    impl->config = config;
    const auto act = config.activation_path.empty() ? std::filesystem::path{}
        : log::resolve_output_path(config.activation_path.c_str());
    const auto res = config.residual_path.empty() ? std::filesystem::path{}
        : log::resolve_output_path(config.residual_path.c_str());
    require(act.empty() || res.empty() || act != res, "metric sinks must differ");
    const auto open = [](const std::string &requested, const std::filesystem::path &path) {
        if (requested.empty()) return static_cast<FILE *>(nullptr);
        require(!path.empty() && log::prepare_output_parent(path), "unsafe metric path");
        FILE *file = std::fopen(path.string().c_str(), "wx");
        require(file != nullptr, "metric sink exists or cannot be created");
        return file;
    };
    impl->activation = open(config.activation_path, act);
    impl->residual = open(config.residual_path, res);
    impl->emit(impl->activation, "RUN",
        text_field("definition_status", "CONFIRMED_BY_USER") +
        text_field("metric_revision", "act-original-fp-observer-v1") +
        text_field("reference_revision", "signed-row-original-bk32-population-2sigma-v1") +
        text_field("reference_population", "logical_row_original_bk32_all_valid_positions") +
        text_field("variance_divisor", "N_valid_positions") +
        text_field("comparison", "signed_x_strictly_greater_than_mean_plus_2sigma") +
        text_field("nonfinite_policy", "unspecified_invalidates_invocation_reference") +
        text_field("selection_stage", "exsia.final_folding_outlier_mask"));
    impl->emit(impl->residual, "RUN", text_field("metric_revision", "residual-logical-shapes-v1") +
        text_field("weighting_status", "PROPOSED_NOT_CONFIRMED") +
        text_field("weighting_revision", "proposed-v3-DTR-v1"));
    auto session = std::shared_ptr<Session>(new Session(std::move(impl)));
    active = session;
    return session;
}
void Session::chunk(uint64_t chunk) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->check();
    require(!impl_->chunk_set || chunk > impl_->chunk, "non-increasing chunk identity");
    impl_->chunk = chunk;
    impl_->chunk_set = true;
}
std::shared_ptr<Invocation> Session::invocation(const std::string &layer, size_t m,
                                                size_t k, const float *original) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->check();
    require(impl_->chunk_set && m && k && m <= SIZE_MAX / k, "invalid invocation");
    auto data = std::make_unique<Invocation::Impl>();
    data->session = shared_from_this();
    data->m = m;
    data->k = k;
    data->identity = field("chunk_id", impl_->chunk) +
        field("invocation_id", impl_->next_invocation++) + text_field("layer", layer);
    if (const auto semantic_context = semantic::current_context())
        data->identity += semantic::identity_fields(semantic_context->identity);
#if GGML_GEMMINI_ACT_QUANT_METRICS
    if (impl_->activation) {
        require(original, "missing original FP view");
        data->observed.assign(m * k, 0);
        for (size_t i = 0; i < m * k; ++i) data->finite += std::isfinite(original[i]);
        data->reference_complete = data->finite == m * k;
        impl_->reference_complete = impl_->reference_complete && data->reference_complete;
        if (data->reference_complete) {
            data->fp_selected.assign(m * k, 0);
            for (size_t row = 0; row < m; ++row) {
                for (size_t block = 0; block < k; block += 32) {
                    const size_t count = std::min(size_t{32}, k - block);
                    const size_t offset = row * k + block;
                    long double mean = 0, squared_deviations = 0;
                    for (size_t index = 0; index < count; ++index) mean += original[offset + index];
                    mean /= count;
                    for (size_t index = 0; index < count; ++index) {
                        const long double delta = original[offset + index] - mean;
                        squared_deviations += delta * delta;
                    }
                    const long double threshold = mean + 2 * std::sqrt(squared_deviations / count);
                    for (size_t index = 0; index < count; ++index) {
                        data->fp_selected[offset + index] = original[offset + index] > threshold;
                        data->fp_count += data->fp_selected[offset + index];
                    }
                }
            }
        }
    }
#else
    (void) original;
#endif
    return std::shared_ptr<Invocation>(new Invocation(std::move(data)));
}
void Session::ensure_healthy() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->check();
}
void Session::finish(bool success) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    require(!success || impl_->completed_quantization == impl_->next_invocation,
            "incomplete invocation coverage");
    const auto fields = std::string(",\"success\":") + (success ? "true" : "false") +
        field("invocation_count", impl_->next_invocation);
    impl_->emit(impl_->activation, "RUN_END", fields +
        ",\"reference_complete\":" + (impl_->reference_complete ? "true" : "false"));
    impl_->emit(impl_->residual, "RUN_END", fields);
    impl_->finished = true;
}
void Invocation::requantized(size_t row, size_t block) {
#if GGML_GEMMINI_ACT_QUANT_METRICS
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (!impl_->session->impl_->activation) return;
    require(!impl_->activation_finished && row < impl_->m && block <= (impl_->k - 1) / 32,
            "requantization outside original logical block");
    impl_->requantized.emplace(row, block);
    ++impl_->requant_events;
#else
    (void) row; (void) block;
#endif
}
void Invocation::position(size_t row, size_t column, bool selected, bool residual) {
#if GGML_GEMMINI_ACT_QUANT_METRICS
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (!impl_->session->impl_->activation) return;
    require(!impl_->activation_finished && row < impl_->m && column < impl_->k,
            "position outside original activation");
    const size_t index = row * impl_->k + column;
    require(!impl_->observed[index], "duplicate activation coordinate");
    impl_->observed[index] = 1;
    ++impl_->observed_count;
    impl_->selected += selected;
    impl_->residual += residual;
    impl_->intersection += selected && !impl_->fp_selected.empty() && impl_->fp_selected[index];
#else
    (void) row; (void) column; (void) selected; (void) residual;
#endif
}
void Invocation::finish_activation() {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    auto &session = *impl_->session->impl_;
    require(!impl_->activation_finished, "duplicate quantization completion");
#if GGML_GEMMINI_RESIDUAL_METRICS
    if (session.residual) {
        std::map<size_t, size_t> by_row;
        for (const auto &stripe : impl_->stripes)
            require(by_row.emplace(stripe.second.first, stripe.second.second).second,
                    "duplicate stripe row range");
        size_t next = 0;
        for (const auto &range : by_row) {
            require(range.first == next, "main stripe overlap/gap");
            next += range.second;
        }
        require(next == impl_->m, "incomplete main stripe coverage");
    }
#endif
#if GGML_GEMMINI_ACT_QUANT_METRICS
    if (session.activation) {
    require(impl_->observed_count == impl_->m * impl_->k,
            "incomplete/duplicate activation coverage");
    const auto optional = [&](const char *name, uint64_t value) {
        return impl_->reference_complete ? field(name, value)
            : ",\"" + std::string(name) + "\":null";
    };
    const auto counts = field("m", impl_->m) + field("k", impl_->k) +
        field("valid_positions", impl_->observed_count) + field("finite_positions", impl_->finite) +
        field("nonfinite_positions", impl_->observed_count - impl_->finite) +
        ",\"reference_complete\":" + (impl_->reference_complete ? "true" : "false") +
        ",\"reference_invalid_reason\":" +
            (impl_->reference_complete ? "null" : "\"NONFINITE_INPUT_UNSPECIFIED\"") +
        optional("fp_selected", impl_->fp_count) + field("potal_selected", impl_->selected) +
        optional("intersection", impl_->intersection) +
        optional("union", impl_->fp_count + impl_->selected - impl_->intersection) +
        field("residual_nnz", impl_->residual) +
        field("eligible_logical_blocks", impl_->m * ((impl_->k - 1) / 32 + 1)) +
        field("unique_actual_requantized_blocks", impl_->requantized.size()) +
        field("p3_requantization_events", impl_->requant_events);
    std::lock_guard<std::mutex> session_lock(session.mutex);
    session.emit(session.activation, "COUNTS", impl_->identity + counts);
    }
#endif
    impl_->activation_finished = true;
    std::lock_guard<std::mutex> session_lock(session.mutex);
    ++session.completed_quantization;
}
void Invocation::main_stripe(size_t stripe, size_t begin, size_t m, size_t n, size_t k) {
#if GGML_GEMMINI_RESIDUAL_METRICS
    std::lock_guard<std::mutex> lock(impl_->mutex);
    auto &session = *impl_->session->impl_;
    if (!session.residual) return;
    require(m && n && k == impl_->k && begin <= impl_->m && m <= impl_->m - begin,
            "invalid main stripe");
    require(impl_->stripes.emplace(stripe, std::make_pair(begin, m)).second, "duplicate main stripe");
    std::lock_guard<std::mutex> session_lock(session.mutex);
    session.emit(session.residual, "MAIN_STRIPE", impl_->identity + field("stripe_id", stripe) +
        field("row_begin", begin) + field("row_count", m) + field("m", m) + field("n", n) + field("k", k));
#else
    (void) stripe; (void) begin; (void) m; (void) n; (void) k;
#endif
}
void Invocation::compact_work(size_t stripe, size_t m, size_t n, size_t k, size_t original_k,
                              size_t tile_i, size_t tile_j, size_t tile_k,
                              const std::vector<Run> &runs, const std::vector<Row> &rows) {
#if GGML_GEMMINI_RESIDUAL_METRICS
    std::lock_guard<std::mutex> lock(impl_->mutex);
    auto &session = *impl_->session->impl_;
    if (!session.residual) return;
    require(m && n && k && original_k == impl_->k && tile_i && tile_j && tile_k &&
            rows.size() == m && !runs.empty(), "invalid compact work");
    const auto parent = impl_->stripes.find(stripe);
    require(parent != impl_->stripes.end() && impl_->compact_stripes.insert(stripe).second,
            "missing/duplicate compact parent");
    size_t cursor = 0;
    uint32_t previous = 0;
    std::string metadata = ",\"runs\":[";
    for (size_t index = 0; index < runs.size(); ++index) {
        const auto &run = runs[index];
        require(run.original_k_mask && run.compact_k_begin == cursor &&
                run.compact_k_count == static_cast<size_t>(__builtin_popcount(run.original_k_mask)) &&
                (index == 0 || run.original_block_id > previous) &&
                static_cast<uint64_t>(run.original_block_id) * 32 +
                    (31 - __builtin_clz(run.original_k_mask)) < original_k,
                "invalid ordered run view");
        require(run.compact_k_count <= k - cursor, "run exceeds compact K");
        cursor += run.compact_k_count;
        previous = run.original_block_id;
        if (index) metadata += ',';
        metadata += "{\"original_block_id\":" + std::to_string(run.original_block_id) +
            field("original_k_mask", run.original_k_mask) + field("compact_k_begin", run.compact_k_begin) +
            field("compact_k_count", run.compact_k_count) + '}';
    }
    require(cursor == k, "incomplete run coverage");
    metadata += "],\"row_map\":[";
    std::set<std::pair<uint32_t, uint32_t>> unique_rows;
    for (size_t index = 0; index < rows.size(); ++index) {
        const auto &row = rows[index];
        require(row.source_row < parent->second.second &&
                unique_rows.emplace(row.original_lane_id, row.source_row).second, "invalid row/lane map");
        if (index) metadata += ',';
        metadata += "{\"original_lane_id\":" + std::to_string(row.original_lane_id) +
            field("source_row", row.source_row) + '}';
    }
    std::lock_guard<std::mutex> session_lock(session.mutex);
    session.emit(session.residual, "COMPACT_WORK", impl_->identity + field("stripe_id", stripe) +
        field("m", m) + field("n", n) + field("k", k) + field("original_k", original_k) +
        field("tile_i_count", tile_i) + field("tile_j_count", tile_j) + field("tile_k_count", tile_k) +
        field("source_row_begin", parent->second.first) + field("source_row_count", parent->second.second) +
        metadata + ']');
#else
    (void) stripe; (void) m; (void) n; (void) k; (void) original_k;
    (void) tile_i; (void) tile_j; (void) tile_k; (void) runs; (void) rows;
#endif
}

}
