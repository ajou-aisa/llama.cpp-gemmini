#include <gemmini/evaluation_metrics.hpp>
#include <gemmini/log.hpp>
#include <gemmini/semantic.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <limits>
#include <locale>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <utility>
#ifndef _WIN32
#include <unistd.h>
#endif

namespace ggml::gemmini::evaluation {
namespace {
std::mutex active_mutex;
std::weak_ptr<Session> active;
[[noreturn]] void fail(const char *message) {
    throw std::runtime_error(std::string("evaluation metrics: ") + message);
}
inline void require(bool condition, const char *message) {
    if (!condition) fail(message);
}
std::string field(const char *name, uint64_t value) {
    return ",\"" + std::string(name) + "\":" + std::to_string(value);
}
std::string text_field(const char *name, const std::string &value) {
    return ",\"" + std::string(name) + "\":" + semantic::quote(value);
}
#if GGML_GEMMINI_SCALE_METRICS
std::string real_field(const char *name, double value) {
    std::ostringstream stream;
    stream.imbue(std::locale::classic());
    stream << ",\"" << name << "\":" << std::setprecision(std::numeric_limits<double>::max_digits10) << value;
    return stream.str();
}
#endif
// SCU sums are integer sufficient statistics: every addition is checked, an overflow fails instead of wrapping.
void add(uint64_t &sum, uint64_t value) {
    require(value <= std::numeric_limits<uint64_t>::max() - sum, "SCU aggregate overflow");
    sum += value;
}
struct ScaleSums {
    uint64_t delta_w_sum = 0, max_delta_w = 0, alignment_count = 0,
             updated_partial_sum_count = 0, total_partial_sum_count = 0, zero_weight_count = 0;
    void merge(const ScaleSums &other) {
        add(delta_w_sum, other.delta_w_sum);
        max_delta_w = std::max(max_delta_w, other.max_delta_w);
        add(alignment_count, other.alignment_count);
        add(updated_partial_sum_count, other.updated_partial_sum_count);
        add(total_partial_sum_count, other.total_partial_sum_count);
        add(zero_weight_count, other.zero_weight_count);
    }
};
const char *work_type_name(size_t type) {
    return type == static_cast<size_t>(ScaleWorkType::Dense) ? "DENSE" : "RESIDUAL";
}
// The SCU coordinates (work type, stripe, original block, column) an invocation has seen: one exact column bitmap
// per (work type, stripe, original block) row, bit index == column. The producer visits a row's columns in order,
// so a coordinate is one bit test-and-set; columns from kBitmapColumns up go to an exact ordered set instead.
class ScaleCoordinates {
public:
    bool insert(ScaleWorkType type, size_t stripe, size_t original_block, size_t column) {
        if (column >= kBitmapColumns) return wide_.emplace(type, stripe, original_block, column).second;
        const Row key{type, stripe, original_block};
        if (!row_ || !(key == key_)) {
            row_ = &rows_[key];  // stable: unordered_map never moves its values
            key_ = key;
        }
        const size_t word = column / 64;
        if (word >= row_->size()) row_->resize(word + 1);
        const uint64_t bit = uint64_t{1} << (column % 64);
        if ((*row_)[word] & bit) return false;
        (*row_)[word] |= bit;
        return true;
    }
private:
    static constexpr size_t kBitmapColumns = size_t{1} << 20;
    struct Row {
        ScaleWorkType type;
        size_t stripe, original_block;
        bool operator==(const Row &other) const {
            return type == other.type && stripe == other.stripe && original_block == other.original_block;
        }
    };
    struct RowHash {
        size_t operator()(const Row &row) const noexcept {
            return std::hash<size_t>{}((row.stripe * 0x9e3779b97f4a7c15ULL) ^
                                       (row.original_block << 1 | static_cast<uint8_t>(row.type)));
        }
    };
    std::unordered_map<Row, std::vector<uint64_t>, RowHash> rows_;
    Row key_{};
    std::vector<uint64_t> *row_ = nullptr;
    std::set<std::tuple<ScaleWorkType, size_t, size_t, size_t>> wide_;
};
// Aggregate SCU mode: the integer sums of one invocation, merged into its chunk's AGGREGATE records exactly once,
// at the chunk boundary that follows it (Session::chunk/finish): residual SCU work runs after the invocation's
// quantization completes, so the chunk boundary is the first point every SCU alignment of the invocation precedes.
struct ScaleAccumulator {
    std::mutex mutex;  // the invocation's SCU lock: guards sums, merged and the invocation's coordinate set
    std::string layer;
    std::array<ScaleSums, 2> sums;  // by ScaleWorkType
    bool merged = false;
    bool quantized = false;  // guarded by the session mutex, set with the session's completed quantization count
};
std::string profile_fields() {
#if GGML_GEMMINI_ACT_QUANT_METRICS || GGML_GEMMINI_RESIDUAL_METRICS || GGML_GEMMINI_SCALE_METRICS
    return text_field("precision", "A" + std::to_string(GGML_GEMMINI_ACTIVATION_BITS) +
        "W" + std::to_string(GGML_GEMMINI_WEIGHT_BITS)) + field("dim", GGML_GEMMINI_DIM);
#else
    return {};
#endif
}
}

struct Session::Impl {
    Config config;
    FILE *activation = nullptr, *residual = nullptr, *scale = nullptr;
    mutable std::mutex mutex;
    uint64_t chunk = 0, next_invocation = 0, next_record = 0, completed_quantization = 0;
    uint64_t activation_observations = 0, residual_observations = 0, scale_observations = 0;
    // Aggregate SCU mode: the invocations of the current chunk, plus run totals.
    std::vector<std::shared_ptr<ScaleAccumulator>> scale_open;
    uint64_t scale_alignments = 0, scale_invocations = 0;
    bool chunk_set = false, finished = false, reference_complete = true;
    std::string failure;
    ~Impl() {
        if (activation) std::fclose(activation);
        if (residual) std::fclose(residual);
        if (scale) std::fclose(scale);
    }
    void check() const {
        require(failure.empty(), failure.c_str());
        require(!finished, "session already finished");
    }
    void emit(FILE *file, const char *kind, const std::string &fields) {
        check();
        if (!file) return;
        const std::string schema = file == activation
            ? "im2p-activation-quant-metrics" : file == residual
                ? "im2p-residual-path-metrics" : config.scale_aggregate
                    ? "im2p-scale-alignment-aggregate" : "im2p-scale-alignment-metrics";
        const auto line = "{\"schema\":" + semantic::quote(schema) +
            ",\"version\":1" + text_field("kind", kind) +
            field("sequence", next_record++) + text_field("run_id", config.run_id) +
            text_field("workload_id", config.workload_id) +
            text_field("manifest_sha256", config.manifest_sha256) + profile_fields() + fields + "}\n";
        if (std::fwrite(line.data(), 1, line.size(), file) != line.size() || std::fflush(file)) {
            failure = "sink write failed";
            throw std::runtime_error("evaluation metrics: " + failure);
        }
        if (std::string(kind) != "RUN" && std::string(kind) != "RUN_END")
            ++(file == activation ? activation_observations :
               file == residual ? residual_observations : scale_observations);
    }
    // Seal and merge every invocation of the closing chunk exactly once, then emit one AGGREGATE per observed
    // (layer, work type), ordered by layer, then DENSE before RESIDUAL. An invocation whose quantization never
    // completed is incomplete: it rejects the boundary of a successful run and is left out of a failed one.
    void flush_scale_sums(bool success) {
        const auto invocations = std::move(scale_open);
        scale_open.clear();
        std::map<std::string, std::array<ScaleSums, 2>> layers;
        uint64_t alignments = 0, observed = 0;
        bool incomplete = false;
        for (const auto &invocation : invocations) {
            std::lock_guard<std::mutex> lock(invocation->mutex);
            require(!invocation->merged, "duplicate SCU invocation merge");
            invocation->merged = true;
            if (!invocation->quantized) {
                incomplete = true;
                continue;
            }
            for (size_t type = 0; type < invocation->sums.size(); ++type)
                if (invocation->sums[type].alignment_count)
                    layers[invocation->layer][type].merge(invocation->sums[type]);
            add(alignments, invocation->sums[0].alignment_count);
            add(alignments, invocation->sums[1].alignment_count);
            observed += invocation->sums[0].alignment_count || invocation->sums[1].alignment_count;
        }
        require(!success || !incomplete, "SCU aggregate of an incomplete invocation");
        add(scale_alignments, alignments);
        scale_invocations += observed;
        for (const auto &[layer, sums] : layers)
            for (size_t type = 0; type < sums.size(); ++type)
                if (sums[type].alignment_count)
                    emit(scale, "AGGREGATE", field("chunk_id", chunk) + text_field("layer", layer) +
                        text_field("work_type", work_type_name(type)) + field("delta_w_sum", sums[type].delta_w_sum) +
                        field("max_delta_w", sums[type].max_delta_w) +
                        field("alignment_count", sums[type].alignment_count) +
                        field("updated_partial_sum_count", sums[type].updated_partial_sum_count) +
                        field("total_partial_sum_count", sums[type].total_partial_sum_count) +
                        field("zero_weight_count", sums[type].zero_weight_count));
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
    std::set<size_t> radix_stripes;
    ScaleCoordinates scale_coordinates;
    std::shared_ptr<ScaleAccumulator> scale;  // aggregate SCU mode only
    uint64_t finite = 0, fp_count = 0, selected = 0, intersection = 0, residual = 0;
    uint64_t requant_events = 0, observed_count = 0;
    bool activation_finished = false, reference_complete = false;
};

Session::Session(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Session::~Session() = default;
Invocation::Invocation(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Invocation::~Invocation() = default;
bool Invocation::scale_enabled() const { return impl_->session->impl_->scale != nullptr; }
bool Invocation::residual_enabled() const { return impl_->session->impl_->residual != nullptr; }
std::shared_ptr<Session> active_session() {
#if GGML_GEMMINI_ACT_QUANT_METRICS || GGML_GEMMINI_RESIDUAL_METRICS || GGML_GEMMINI_SCALE_METRICS
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
    require(GGML_GEMMINI_SCALE_METRICS || (config.scale_path.empty() && config.scale_fd < 0), "SCALE metrics compiled out");
    require(config.scale_fd < 0 || config.scale_path.empty(), "SCALE path and descriptor are mutually exclusive");
    if (config.activation_path.empty() && config.residual_path.empty() && config.scale_path.empty() && config.scale_fd < 0) return {};
    require(!config.run_id.empty() && !config.workload_id.empty(), "missing run/workload identity");
    require(config.manifest_sha256.size() == 64 &&
            config.manifest_sha256.find_first_not_of("0123456789abcdef") == std::string::npos,
            "validated manifest SHA256 required");
    std::lock_guard<std::mutex> lock(active_mutex);
    require(active.expired(), "another metrics session is active");
    auto impl = std::make_unique<Impl>();
    impl->config = config;
    const auto act = config.activation_path.empty() ? std::filesystem::path{}
        : log::resolve_output_path(config.activation_path.c_str());
    const auto res = config.residual_path.empty() ? std::filesystem::path{}
        : log::resolve_output_path(config.residual_path.c_str());
    const auto scale = config.scale_path.empty() ? std::filesystem::path{}
        : log::resolve_output_path(config.scale_path.c_str());
    require(act.empty() || res.empty() || act != res, "metric sinks must differ");
    require(scale.empty() || ((act.empty() || act != scale) && (res.empty() || res != scale)),
            "metric sinks must differ");
    const auto open = [](const std::string &requested, const std::filesystem::path &path) {
        if (requested.empty()) return static_cast<FILE *>(nullptr);
        require(!path.empty() && log::prepare_output_parent(path), "unsafe metric path");
        FILE *file = std::fopen(path.string().c_str(), "wx");
        require(file != nullptr, "metric sink exists or cannot be created");
        return file;
    };
    impl->activation = open(config.activation_path, act);
    impl->residual = open(config.residual_path, res);
    impl->scale = open(config.scale_path, scale);
    if (config.scale_fd >= 0) {
#ifdef _WIN32
        throw std::runtime_error("evaluation metrics: inherited SCALE descriptor requires POSIX");
#else
        const int fd = dup(config.scale_fd);
        require(fd >= 0, "invalid inherited SCALE descriptor");
        impl->scale = fdopen(fd, "w");
        if (!impl->scale) close(fd);
        require(impl->scale != nullptr, "cannot open inherited SCALE descriptor");
#endif
    }
    impl->emit(impl->activation, "RUN",
        text_field("definition_status", "CONFIRMED_BY_USER") +
        text_field("metric_revision", "act-original-fp-observer-v1") +
        text_field("reference_revision", "signed-row-original-bk32-population-2sigma-v1") +
        text_field("reference_population", "logical_row_original_bk32_all_valid_positions") +
        text_field("variance_divisor", "N_valid_positions") +
        text_field("comparison", "signed_x_strictly_greater_than_mean_plus_2sigma") +
        text_field("nonfinite_policy", "unspecified_invalidates_invocation_reference") +
        text_field("selection_stage", "exsia.final_folding_outlier_mask"));
    impl->emit(impl->residual, "RUN", text_field("metric_revision", "residual-producer-shapes-v2") +
        text_field("weighting_status", "CONFIRMED_BY_USER") +
        text_field("weighting_revision", "integer-count-mac-capacity-v1"));
    impl->emit(impl->scale, "RUN", text_field("metric_revision", "hp1-scu-offset-v1") +
        text_field("scale_domain", "hp1_block_pot_to_channel_anchor") +
        text_field("update_definition", "partial_sums_requiring_nonzero_scu_shift") +
        (config.scale_aggregate ? text_field("collection_mode", "aggregate") : std::string()));
    auto session = std::shared_ptr<Session>(new Session(std::move(impl)));
    active = session;
    return session;
}
void Session::chunk(uint64_t chunk) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->check();
    require(!impl_->chunk_set || chunk > impl_->chunk, "non-increasing chunk identity");
    impl_->flush_scale_sums(true);  // aggregates of the previous chunk; empty in detailed mode
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
    if (impl_->scale && impl_->config.scale_aggregate) {
        data->scale = std::make_shared<ScaleAccumulator>();
        data->scale->layer = layer;
        impl_->scale_open.push_back(data->scale);
    }
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
        ",\"reference_complete\":" + (impl_->reference_complete ? "true" : "false") +
        field("observation_count", impl_->activation_observations));
    impl_->emit(impl_->residual, "RUN_END", fields + field("observation_count", impl_->residual_observations));
    if (impl_->config.scale_aggregate) {
        impl_->flush_scale_sums(success);
        impl_->emit(impl_->scale, "RUN_END", fields + field("observation_count", impl_->scale_observations) +
            field("alignment_count", impl_->scale_alignments) +
            field("scale_invocation_count", impl_->scale_invocations));
    } else {
        impl_->emit(impl_->scale, "RUN_END", fields + field("observation_count", impl_->scale_observations));
    }
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
        require(impl_->radix_stripes.size() == impl_->stripes.size(), "incomplete radix stripe coverage");
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
    if (impl_->scale) impl_->scale->quantized = true;
}
void Invocation::main_stripe(size_t stripe, size_t begin, size_t m, size_t n, size_t k) {
#if GGML_GEMMINI_RESIDUAL_METRICS
    std::lock_guard<std::mutex> lock(impl_->mutex);
    auto &session = *impl_->session->impl_;
    if (!session.residual) return;
    require(m && n && k == impl_->k && begin <= impl_->m && m <= impl_->m - begin,
            "invalid main stripe");
    require(impl_->stripes.emplace(stripe, std::make_pair(begin, m)).second, "duplicate main stripe");
    size_t k_fragments = 0;
    for (size_t block = 0; block < k; block += 32)
        k_fragments += (std::min(size_t{32}, k - block) + GGML_GEMMINI_DIM - 1) / GGML_GEMMINI_DIM;
    const size_t physical_fragments = ((m - 1) / GGML_GEMMINI_DIM + 1) *
        ((n - 1) / GGML_GEMMINI_DIM + 1) * k_fragments;
    std::lock_guard<std::mutex> session_lock(session.mutex);
    session.emit(session.residual, "MAIN_STRIPE", impl_->identity + field("stripe_id", stripe) +
        field("row_begin", begin) + field("row_count", m) + field("m", m) + field("n", n) + field("k", k) +
        field("physical_fragments", physical_fragments));
#else
    (void) stripe; (void) begin; (void) m; (void) n; (void) k;
#endif
}
void Invocation::radix_stripe(size_t stripe, size_t radix_limb_count) {
#if GGML_GEMMINI_RESIDUAL_METRICS
    std::lock_guard<std::mutex> lock(impl_->mutex);
    auto &session = *impl_->session->impl_;
    if (!session.residual) return;
    const auto parent = impl_->stripes.find(stripe);
    require(parent != impl_->stripes.end() && radix_limb_count <= 9 &&
            impl_->radix_stripes.insert(stripe).second, "invalid/duplicate radix stripe");
    const size_t original_rows = parent->second.second * radix_limb_count;
    require(radix_limb_count == 0 || original_rows / radix_limb_count == parent->second.second,
            "radix row count overflow");
    std::lock_guard<std::mutex> session_lock(session.mutex);
    session.emit(session.residual, "RADIX_STRIPE", impl_->identity + field("stripe_id", stripe) +
        field("radix_limb_count", radix_limb_count) + field("original_rows", original_rows) +
        field("original_radix_rows", original_rows) + field("main_original_rows", parent->second.second) +
        field("original_k", impl_->k));
#else
    (void) stripe; (void) radix_limb_count;
#endif
}
void Invocation::compact_work(size_t stripe, size_t m, size_t n, size_t k, size_t original_k,
                              size_t tile_i, size_t tile_j, size_t tile_k,
                              const std::vector<Run> &runs, const std::vector<Row> &rows,
                              size_t radix_limb_count, size_t physical_fragments) {
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
    const size_t original_rows = parent->second.second * radix_limb_count;
    require(radix_limb_count == 0 || (original_rows / radix_limb_count == parent->second.second &&
            original_rows >= rows.size()), "invalid radix row count");
    std::lock_guard<std::mutex> session_lock(session.mutex);
    session.emit(session.residual, "COMPACT_WORK", impl_->identity + field("stripe_id", stripe) +
        field("m", m) + field("n", n) + field("k", k) + field("original_k", original_k) +
        field("tile_i_count", tile_i) + field("tile_j_count", tile_j) + field("tile_k_count", tile_k) +
        field("source_row_begin", parent->second.first) + field("source_row_count", parent->second.second) +
        field("radix_limb_count", radix_limb_count) + field("original_rows", original_rows) +
        field("original_radix_rows", original_rows) + field("main_original_rows", parent->second.second) +
        field("retained_rows", m) + field("retained_k", k) + field("compact_k", k) +
        field("zero_limb_pruned_count", radix_limb_count ? original_rows - m : 0) +
        field("physical_fragments", physical_fragments) + metadata + ']');
#else
    (void) stripe; (void) m; (void) n; (void) k; (void) original_k;
    (void) tile_i; (void) tile_j; (void) tile_k; (void) runs; (void) rows;
    (void) radix_limb_count; (void) physical_fragments;
#endif
}

void Invocation::scale_alignment(size_t stripe, ScaleWorkType work_type, size_t column,
                                  size_t original_block, double original_weight_scale,
                                  double aligned_pot_scale, uint32_t scu_shift_offset,
                                  uint64_t updated_partial_sum_count,
                                  uint64_t total_partial_sum_count, bool zero_weight) {
#if GGML_GEMMINI_SCALE_METRICS
    auto &session = *impl_->session->impl_;
    if (!session.scale) return;
    // Aggregate mode takes only the invocation's SCU lock; detailed mode the invocation lock, then the session's.
    std::lock_guard<std::mutex> lock(impl_->scale ? impl_->scale->mutex : impl_->mutex);
    require(!impl_->scale || !impl_->scale->merged, "SCU alignment after its chunk aggregate");
    require((work_type == ScaleWorkType::Dense || work_type == ScaleWorkType::Residual) &&
            std::isfinite(original_weight_scale) && std::isfinite(aligned_pot_scale) &&
            total_partial_sum_count && updated_partial_sum_count <= total_partial_sum_count &&
            original_block <= (impl_->k - 1) / 32 && scu_shift_offset <= 32767,
            "invalid SCU alignment record");
    require(zero_weight ? original_weight_scale == 0 && aligned_pot_scale >= 0 &&
                scu_shift_offset == 0 && updated_partial_sum_count == 0
            : original_weight_scale > 0 && aligned_pot_scale > 0 &&
                std::ldexp(aligned_pot_scale, scu_shift_offset) == original_weight_scale &&
                updated_partial_sum_count == (scu_shift_offset ? total_partial_sum_count : 0),
            "SCU scale/offset/count mismatch");
    require(impl_->scale_coordinates.insert(work_type, stripe, original_block, column),
            "duplicate SCU alignment coordinate");
    if (impl_->scale) {
        auto &sums = impl_->scale->sums[static_cast<size_t>(work_type)];
        auto next = sums;  // checked as a whole: an overflowing alignment changes no sum
        next.merge({scu_shift_offset, scu_shift_offset, 1, updated_partial_sum_count, total_partial_sum_count,
                    zero_weight});
        sums = next;
        return;
    }
    std::lock_guard<std::mutex> session_lock(session.mutex);
    session.emit(session.scale, "SCALE_ALIGNMENT", impl_->identity + field("stripe_id", stripe) +
        text_field("work_type", work_type_name(static_cast<size_t>(work_type))) + field("column", column) +
        field("original_block", original_block) + real_field("original_weight_scale", original_weight_scale) +
        real_field("aligned_pot_scale", aligned_pot_scale) + field("scu_shift_offset", scu_shift_offset) +
        field("updated_partial_sum_count", updated_partial_sum_count) +
        field("total_partial_sum_count", total_partial_sum_count) +
        ",\"zero_weight\":" + (zero_weight ? "true" : "false"));
#else
    (void) stripe; (void) work_type; (void) column; (void) original_block;
    (void) original_weight_scale; (void) aligned_pot_scale; (void) scu_shift_offset;
    (void) updated_partial_sum_count; (void) total_partial_sum_count; (void) zero_weight;
#endif
}

}
