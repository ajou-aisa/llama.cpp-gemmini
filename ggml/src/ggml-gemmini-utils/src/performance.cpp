#include <gemmini/performance.hpp>
#include <gemmini/log.hpp>
#include <gemmini/host-timing.hpp>
#include <gemmini/trace-context.hpp>
#include "json.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace ggml::gemmini::performance {
namespace {
using Json = nlohmann::json;
struct Total {
    uint64_t value = 0;
    uint64_t samples = 0;
    std::string reason;

    void fail(const char * why) {
        if (reason.empty()) reason = why ? why : "unavailable";
    }
    void add(uint64_t amount) {
        ++samples;
        if (amount > std::numeric_limits<uint64_t>::max() - value) fail("aggregate_overflow");
        else value += amount;
    }
    bool valid() const { return samples != 0 && reason.empty(); }
};

struct CpuSpan {
    gemmini_cpu_sample start, end;
    uint64_t cycles, thread_ns;
    std::string cycles_reason, thread_reason;
};
struct WallSpan { uint64_t start, end; };
using NpuKey = std::tuple<std::string, std::string, std::string>;
struct NpuSample {
    NpuKey key;
    uint64_t cycles;
    bool collected;
    std::string reason;
};
struct PhaseTotals {
    uint64_t operations = 0, failures = 0;
    Total elapsed, cycles, thread_ns, work_wall, measured_wall;
    std::map<NpuKey, Total> npu;
};
struct Operation {
    bool active = false;
    bool success = false;
    Phase phase = Phase::prefill;
    uint64_t id = 0, start = 0, end = 0;
    std::vector<CpuSpan> cpu;
    std::vector<WallSpan> wall;
    std::vector<NpuSample> npu;
    std::string wall_reason;
    std::vector<uint64_t> sequences;
    std::vector<uint64_t> cpu_interval_sequences;
    uint64_t cpu_interval_records = 0;
};
struct Request {
    bool active = false;
    uint64_t start = 0, first_token = 0, last_token = 0, tokens = 0, last_operation_end = 0;
    std::string token_reason;
};
struct State {
    std::array<PhaseTotals, 2> phases;
    Operation operation;
    Request request;
    uint64_t requests = 0, completed_requests = 0, operations = 0, tokens = 0, frequency_hz = 0;
    Total request_elapsed, ttft, token_gaps;
    bool cpu_interval_coverage_verified = true;
    bool latency_boundary_verified = true;
};

void reduce_operation(State & state, Operation & operation) {
    const uint64_t end_ns = operation.end;
    auto & phase = state.phases[operation.phase == Phase::prefill ? 0 : 1];
    ++phase.operations;
    if (!operation.success) ++phase.failures;
    if (end_ns < operation.start) phase.elapsed.fail("clock_regression");
    else phase.elapsed.add(end_ns - operation.start);

    std::sort(operation.cpu.begin(), operation.cpu.end(), [](const CpuSpan & a, const CpuSpan & b) {
        if (a.start.tid != b.start.tid) return a.start.tid < b.start.tid;
        if (a.start.ns != b.start.ns) return a.start.ns < b.start.ns;
        return a.end.ns > b.end.ns;
    });
    const CpuSpan * previous = nullptr;
    for (const auto & span : operation.cpu) {
        if (span.start.tid == 0 || span.start.tid != span.end.tid || span.end.ns < span.start.ns ||
            span.start.ns < operation.start || span.end.ns > end_ns) {
            phase.cycles.fail(span.cycles_reason.empty() ? "invalid_cpu_span" : span.cycles_reason.c_str());
            phase.thread_ns.fail(span.thread_reason.empty() ? "invalid_cpu_span" : span.thread_reason.c_str());
            continue;
        }
        // A containing scope avoids double-counting, but cannot erase a failed sample.
        if (!span.cycles_reason.empty()) phase.cycles.fail(span.cycles_reason.c_str());
        if (!span.thread_reason.empty()) phase.thread_ns.fail(span.thread_reason.c_str());
        if (previous && previous->start.tid == span.start.tid && span.start.ns <= previous->end.ns) {
            if (span.end.ns <= previous->end.ns) {
                if (span.start.native_source != previous->start.native_source ||
                    span.start.owner_token != previous->start.owner_token ||
                    span.start.generation != previous->start.generation) {
                    phase.cycles.fail("nested_cpu_counter_domain_mismatch");
                } else if (span.cycles_reason.empty() && previous->cycles_reason.empty() &&
                    (span.start.counter < previous->start.counter || span.end.counter > previous->end.counter)) {
                    phase.cycles.fail("nested_cpu_counter_value_mismatch");
                }
                if (span.thread_reason.empty() && previous->thread_reason.empty() &&
                    (span.start.thread_cpu_ns < previous->start.thread_cpu_ns ||
                     span.end.thread_cpu_ns > previous->end.thread_cpu_ns))
                    phase.thread_ns.fail("nested_thread_counter_value_mismatch");
                continue;
            }
            if (span.start.ns < previous->end.ns) {
                phase.cycles.fail("partially_overlapping_cpu_spans");
                phase.thread_ns.fail("partially_overlapping_cpu_spans");
            }
        }
        previous = &span;
        if (span.cycles_reason.empty()) phase.cycles.add(span.cycles);
        if (span.thread_reason.empty()) phase.thread_ns.add(span.thread_ns);
    }
    if (operation.cpu.empty()) {
        phase.cycles.fail("no_cpu_samples");
        phase.thread_ns.fail("no_cpu_samples");
    }

    std::sort(operation.wall.begin(), operation.wall.end(), [](WallSpan a, WallSpan b) {
        return a.start < b.start;
    });
    uint64_t wall_ns = 0, union_end = operation.start;
    for (const auto span : operation.wall) {
        if (span.end < span.start || span.start < operation.start || span.end > end_ns) {
            if (operation.wall_reason.empty()) operation.wall_reason = "invalid_cpu_work_span";
            continue;
        }
        const uint64_t start = std::max(span.start, union_end);
        if (span.end > start) wall_ns += span.end - start;
        union_end = std::max(union_end, span.end);
    }
    phase.measured_wall.add(wall_ns);
    if (operation.wall.empty() && operation.wall_reason.empty()) operation.wall_reason = "no_cpu_work_spans";
    if (!operation.wall_reason.empty()) phase.work_wall.fail(operation.wall_reason.c_str());
    else phase.work_wall.add(wall_ns);
    for (const auto & sample : operation.npu) {
        auto & total = phase.npu[sample.key];
        if (sample.collected) total.add(sample.cycles);
        else total.fail(sample.reason.c_str());
    }
}

void finish_request_replay(State & state, uint64_t end_ns) {
    auto & request = state.request;
    if (!request.active) return;
    ++state.completed_requests;
    if (end_ns < request.start || end_ns < request.last_operation_end ||
        (request.tokens && end_ns < request.last_token)) {
        state.request_elapsed.fail("clock_regression");
        request.token_reason = "clock_regression";
    } else state.request_elapsed.add(end_ns - request.start);
    if (!request.token_reason.empty()) {
        state.ttft.fail(request.token_reason.c_str());
        state.token_gaps.fail(request.token_reason.c_str());
    } else if (request.tokens) {
        state.ttft.add(request.first_token - request.start);
        if (request.tokens > 1) {
            state.token_gaps.add(request.last_token - request.first_token);
            state.token_gaps.samples += request.tokens - 2;
        }
    }
    request.active = false;
}

std::string quote(const std::string & value) {
    std::string result = "\"";
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') {
            result += '\\';
            result += static_cast<char>(c);
        } else if (c < 0x20) {
            const char hex[] = "0123456789abcdef";
            result += "\\u00";
            result += hex[c >> 4];
            result += hex[c & 15];
        } else result += static_cast<char>(c);
    }
    return result + '"';
}

bool valid(const Total & total) {
    return total.valid();
}

const char * reason(const Total & total, const char * empty = "not_collected") {
    if (!total.reason.empty()) return total.reason.c_str();
    return total.samples ? nullptr : empty;
}

void metric_json(std::ostream & out, const char * name, const Total & total,
                 bool average = false, const char * empty = "not_collected") {
    out << ',' << quote(name) << ':';
    if (valid(total)) {
        if (average) out << static_cast<long double>(total.value) / total.samples;
        else out << total.value;
    } else out << "null";
    out << ',' << quote(std::string(name) + "_reason") << ':';
    const char * why = reason(total, empty);
    out << (why ? quote(why) : "null");
}

void print_metric(FILE * output, const char * name, const Total & total,
                  bool milliseconds = false, bool average = false,
                  const char * empty = "not_collected") {
    if (valid(total)) {
        long double value = total.value;
        if (average) value /= total.samples;
        if (milliseconds) std::fprintf(output, "%s=%.3Lf ms", name, value / 1000000.0L);
        else std::fprintf(output, "%s=%llu", name, static_cast<unsigned long long>(total.value));
    } else std::fprintf(output, "%s=n/a (%s)", name, reason(total, empty));
}

std::string serialize_state(const State & state) {
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::setprecision(18);
    out << "{\"schema\":\"gemmini.cycle\",\"version\":2,\"record_type\":\"FINAL_INFERENCE_SUMMARY\""
        << ",\"available\":true,\"reason\":null,\"requests\":" << state.requests << ",\"completed_requests\":" << state.completed_requests
        << ",\"tokens\":" << state.tokens;
    out << ",\"cpu_interval_coverage\":\"" << (state.cpu_interval_coverage_verified ?
        "verified" : "unverified_legacy_log") << '"';
    out << ",\"latency_boundary_status\":\"" << (state.latency_boundary_verified ?
        "verified" : "unverified_legacy_log") << '"'
        << ",\"latency_source\":\"host_monotonic_request_events\""
        << ",\"request_start_boundary\":" << (state.latency_boundary_verified ?
            "\"tokenized_prompt_before_inference\"" : "null")
        << ",\"token_ready_boundary\":" << (state.latency_boundary_verified ?
            "\"sampling_and_accept_complete\"" : "null")
        << ",\"ttft_definition\":\"first_token_ready-request_start\",\"ttft_aggregation\":\"request_mean\""
        << ",\"tpot_definition\":\"(last_token_ready-first_token_ready)/(tokens-1), tokens>1\""
        << ",\"tpot_aggregation\":\"token_gap_weighted_mean\"";
    metric_json(out, "request_elapsed_ns", state.request_elapsed);
    metric_json(out, "ttft_ns", state.ttft, true, "no_generated_tokens");
    metric_json(out, "tpot_ns", state.token_gaps, true, "fewer_than_two_tokens_per_request");
    out << ",\"ttft_samples\":" << state.ttft.samples << ",\"tpot_gaps\":" << state.token_gaps.samples
        << ",\"npu_frequency_hz\":" << (state.frequency_hz ? std::to_string(state.frequency_hz) : "null")
        << ",\"cpu_work_wall_definition\":\"union_of_instrumented_cpu_stages\",\"phases\":[";
    for (size_t index = 0; index < state.phases.size(); ++index) {
        const auto & phase = state.phases[index];
        if (index) out << ',';
        out << "{\"phase\":\"" << (index == 0 ? "prefill" : "decode") << "\",\"operations\":"
            << phase.operations << ",\"failed_operations\":" << phase.failures;
        metric_json(out, "elapsed_ns", phase.elapsed);
        metric_json(out, "cpu_cycles", phase.cycles);
        metric_json(out, "thread_cpu_ns", phase.thread_ns);
        metric_json(out, "cpu_work_wall_ns", phase.work_wall);
        metric_json(out, "cpu_work_wall_measured_ns", phase.measured_wall);
        out << ",\"npu\":[";
        bool first = true;
        for (const auto & entry : phase.npu) {
            if (!first) out << ',';
            first = false;
            out << "{\"backend\":" << quote(std::get<0>(entry.first))
                << ",\"domain\":" << quote(std::get<1>(entry.first))
                << ",\"metric\":" << quote(std::get<2>(entry.first));
            metric_json(out, "cycles", entry.second);
            out << ",\"time_ns\":";
            if (valid(entry.second) && state.frequency_hz) {
                out << static_cast<long double>(entry.second.value) * 1000000000.0L / state.frequency_hz;
            } else out << "null";
            out << ",\"time_reason\":";
            const char * why = reason(entry.second);
            if (!why && !state.frequency_hz) why = "npu_frequency_unavailable";
            out << (why ? quote(why) : "null")
                << ",\"time_source\":\"cycles/configured_npu_frequency\",\"time_kind\":\"cycle_derived\"}";
        }
        out << "],\"npu_reason\":" << (phase.npu.empty() ? "\"no_device_counter_records\"" : "null") << '}';
    }
    out << "]}";
    return out.str();
}

void print_state(FILE * output, const State & state) {
    if (!output) return;
    std::fprintf(output, "\nInference performance (%llu request%s, %llu tokens)\n",
                 static_cast<unsigned long long>(state.requests), state.requests == 1 ? "" : "s",
                 static_cast<unsigned long long>(state.tokens));
    for (size_t index = 0; index < state.phases.size(); ++index) {
        const auto & phase = state.phases[index];
        std::fprintf(output, "  %s: ", index == 0 ? "prefill" : "decode");
        print_metric(output, "elapsed", phase.elapsed, true);
        std::fprintf(output, ", ");
        print_metric(output, "CPU cycles", phase.cycles);
        std::fprintf(output, ", ");
        print_metric(output, "worker CPU", phase.thread_ns, true);
        std::fprintf(output, ", ");
        print_metric(output, "CPU work wall", phase.work_wall, true);
        if (!valid(phase.work_wall) && valid(phase.measured_wall)) {
            std::fprintf(output, " [measured stages %.3Lf ms]",
                         static_cast<long double>(phase.measured_wall.value) / 1000000.0L);
        }
        std::fprintf(output, ", failures=%llu\n", static_cast<unsigned long long>(phase.failures));
        for (const auto & entry : phase.npu) {
            std::fprintf(output, "    NPU %s/%s/%s: ", std::get<0>(entry.first).c_str(),
                         std::get<1>(entry.first).c_str(), std::get<2>(entry.first).c_str());
            print_metric(output, "cycles", entry.second);
            if (valid(entry.second) && state.frequency_hz) {
                std::fprintf(output, ", %.3Lf ms (cycles / %llu Hz)\n",
                    static_cast<long double>(entry.second.value) * 1000.0L / state.frequency_hz,
                    static_cast<unsigned long long>(state.frequency_hz));
            } else {
                std::fprintf(output, ", time=n/a (%s)\n", valid(entry.second) ?
                             "npu_frequency_unavailable" : reason(entry.second));
            }
        }
        if (phase.npu.empty()) std::fprintf(output, "    NPU: n/a (no_device_counter_records)\n");
    }
    std::fprintf(output, "  ");
    print_metric(output, "TTFT", state.ttft, true, true, "no_generated_tokens");
    std::fprintf(output, ", ");
    print_metric(output, "TPOT", state.token_gaps, true, true, "fewer_than_two_tokens_per_request");
    std::fprintf(output, "\n");
}


std::mutex control_mutex;
std::atomic<uint64_t> context_version{0}, active_request{0}, active_operation{0};
std::atomic<Phase> active_phase{Phase::prefill};
std::atomic<uint64_t> resource_sequence{0};
std::atomic<uint64_t> cpu_interval_sequence{0};
uint64_t request_sequence = 0, operation_sequence = 0, event_sequence = 0, frequency_hz = 0;
uint64_t token_sequence = 0;

void publish_context(Context context) {
    std::lock_guard<std::mutex> lock(control_mutex);
    context_version.fetch_add(1);
    active_request.store(context.request_id);
    active_operation.store(context.operation_id);
    active_phase.store(context.phase);
    context_version.fetch_add(1);
}

void emit_event(const char * event, uint64_t timestamp, bool success = true,
                std::optional<int32_t> token_id = {}) noexcept {
#if LOG_CYCLE
    try {
        Json record = {{"schema", "gemmini.cycle"}, {"version", 2},
            {"record_type", "INFERENCE_EVENT"}, {"event", event},
            {"execution_id", cycle::host_execution_id()}, {"timestamp_ns", timestamp},
            {"event_sequence", ++event_sequence}, {"npu_frequency_hz", frequency_hz}};
        if (std::string_view(event) == "request_start" || std::string_view(event) == "token_ready") {
            record["clock"] = "steady_clock";
            record["boundary"] = std::string_view(event) == "request_start" ?
                "tokenized_prompt_before_inference" : "sampling_and_accept_complete";
        }
        if (std::string_view(event) == "token_ready") {
            record["token_index"] = token_sequence++;
            record["token_id"] = token_id ? Json(*token_id) : Json();
        }
        if (std::string_view(event) == "operation_start" || std::string_view(event) == "operation_end")
            record["token_step"] = token_sequence;
        if (std::string_view(event) == "operation_end") {
            record["success"] = success;
            record["resource_samples"] = resource_sequence.load();
            record["cpu_interval_samples"] = cpu_interval_sequence.load();
        }
        if (std::string_view(event) == "session_end") record["log_healthy"] = log::cycle.healthy();
        log::cycle.write_json(record.dump());
        log::cycle.flush();
    } catch (...) {
        log::cycle.report_failure("inference event");
    }
#else
    (void) event; (void) timestamp; (void) success; (void) token_id;
#endif
}

void enqueue(Measurement measurement) noexcept {
#if LOG_CYCLE
    if (!(measurement.trace.flags & GEMMINI_TRACE_CAPTURED)) measurement.trace = gemmini_trace_capture();
    const auto origin = trace::inference_context(measurement.trace);
    if (origin.operation_id == 0) return;
    try {
        const auto active = capture_context();
        // Late resources retain their captured origin; do not inflate the next
        // operation's cardinality. The legacy reducer may reject a late log.
        measurement.sequence = origin.request_id == active.request_id &&
            origin.operation_id == active.operation_id ? resource_sequence.fetch_add(1) + 1 : 0;
        log::cycle.write_measurement(measurement);
    } catch (...) {
        log::cycle.report_failure("performance sample");
    }
#else
    (void) measurement;
#endif
}

Json cpu_sample_json(const gemmini_cpu_sample & sample) {
    return {{"ns", sample.ns}, {"tid", sample.tid}, {"thread_cpu_ns", sample.thread_cpu_ns},
        {"counter", sample.counter}, {"owner_token", sample.owner_token},
        {"generation", sample.generation}, {"thread_cpu_valid", sample.thread_cpu_valid},
        {"native_valid", sample.native_valid}, {"native_reason", sample.native_reason},
        {"native_source", sample.native_source}};
}

uint64_t unsigned_value(const Json & value) {
    if (!value.is_number_unsigned()) throw std::runtime_error("invalid_unsigned_value");
    return value.get<uint64_t>();
}

uint64_t number(const Json & object, const char * key) {
    return unsigned_value(object.at(key));
}

std::string text_value(const Json & object, const char * key) {
    const auto & value = object.at(key);
    if (!value.is_string() || value.get_ref<const std::string &>().empty())
        throw std::runtime_error("invalid_string_value");
    return value.get<std::string>();
}

bool boolean(const Json & object, const char * key) {
    const auto & value = object.at(key);
    if (!value.is_boolean()) throw std::runtime_error("invalid_boolean_value");
    return value.get<bool>();
}

gemmini_cpu_sample parse_cpu_sample(const Json & sample) {
    gemmini_cpu_sample value{};
    value.ns = number(sample, "ns");
    value.tid = number(sample, "tid");
    value.thread_cpu_ns = number(sample, "thread_cpu_ns");
    value.counter = number(sample, "counter");
    value.owner_token = number(sample, "owner_token");
    value.generation = number(sample, "generation");
    const auto byte = [&](const char * key, uint64_t limit) {
        const auto v = number(sample, key);
        if (v > limit) throw std::runtime_error("invalid_cpu_sample_flag");
        return static_cast<uint8_t>(v);
    };
    value.thread_cpu_valid = byte("thread_cpu_valid", 1);
    value.native_valid = byte("native_valid", 1);
    value.native_reason = byte("native_reason", 255);
    value.native_source = byte("native_source", GEMMINI_CPU_COUNTER_THREAD_PERF);
    return value;
}

std::pair<uint64_t, std::string> nullable_counter(const Json & record, const char * key,
                                                const char * reason_key) {
    const auto & reason = record.at(reason_key);
    if (reason.is_null()) return {number(record, key), {}};
    if (!record.at(key).is_null()) throw std::runtime_error("invalid_counter_null_semantics");
    return {0, text_value(record, reason_key)};
}

void match_context(const Json & record, const State & state) {
    const auto & context = record.at("inference_context");
    if (!state.request.active || number(context, "request_id") != state.requests ||
        !boolean(context, "included")) throw std::runtime_error("request_context_mismatch");
    if (record.contains("token_step") && number(record, "token_step") != state.request.tokens)
        throw std::runtime_error("token_step_mismatch");
    if (state.operation.active) {
        if (number(context, "operation_id") != state.operation.id ||
            text_value(context, "phase") != (state.operation.phase == Phase::prefill ? "prefill" : "decode"))
            throw std::runtime_error("operation_context_mismatch");
    } else if (!context.at("operation_id").is_null() || !context.at("phase").is_null()) {
        throw std::runtime_error("operation_context_mismatch");
    }
}

void check_latency_boundary(const Json & record, const char * expected, State & state) {
    if (!record.contains("boundary") || !record.contains("clock")) state.latency_boundary_verified = false;
    if ((record.contains("boundary") && text_value(record, "boundary") != expected) ||
        (record.contains("clock") && text_value(record, "clock") != "steady_clock"))
        throw std::runtime_error("unsupported_latency_boundary");
}

void consume_measurement(const Json & record, State & state) {
    if (text_value(record, "role") != "summary_resource")
        throw std::runtime_error("invalid_resource_role");
    if (!state.operation.active) throw std::runtime_error("resource_outside_operation");
    match_context(record, state);
    auto & operation = state.operation;
    const auto sequence = number(record, "sequence");
    if (!sequence) throw std::runtime_error("invalid_resource_sequence");
    operation.sequences.push_back(sequence);
    const auto kind = text_value(record, "kind");
    if (kind == "cpu") {
        const auto start = parse_cpu_sample(record.at("start"));
        const auto end = parse_cpu_sample(record.at("end"));
        auto cycles = nullable_counter(record, "cycles", "cycles_reason");
        auto thread = nullable_counter(record, "thread_ns", "thread_reason");
        if (cycles.second.empty() && (!start.native_valid || !end.native_valid ||
            start.native_source != GEMMINI_CPU_COUNTER_THREAD_PERF ||
            end.native_source != GEMMINI_CPU_COUNTER_THREAD_PERF || !start.owner_token ||
            start.owner_token != end.owner_token || !start.generation || start.generation != end.generation ||
            end.counter < start.counter || end.counter - start.counter != cycles.first))
            throw std::runtime_error("invalid_cpu_counter_value");
        if (thread.second.empty() && (!start.thread_cpu_valid || !end.thread_cpu_valid ||
            end.thread_cpu_ns < start.thread_cpu_ns || end.thread_cpu_ns - start.thread_cpu_ns != thread.first))
            throw std::runtime_error("invalid_thread_counter_value");
        operation.cpu.push_back({start, end, cycles.first, thread.first,
                                 std::move(cycles.second), std::move(thread.second)});
    } else if (kind == "wall") {
        operation.wall.push_back({number(record, "begin_ns"), number(record, "end_ns")});
    } else if (kind == "wall_gap") {
        const auto reason = text_value(record, "reason");
        if (operation.wall_reason.empty()) operation.wall_reason = reason;
    } else if (kind == "npu") {
        auto counter = nullable_counter(record, "cycles", "reason");
        const auto collected = boolean(record, "collected");
        if (collected != counter.second.empty()) throw std::runtime_error("invalid_npu_counter_status");
        operation.npu.push_back({{text_value(record, "backend"), text_value(record, "domain"),
            text_value(record, "metric")}, counter.first, collected, std::move(counter.second)});
    } else throw std::runtime_error("unknown_resource_kind");
}

Json parse_line(const std::string & line) {
    std::vector<std::set<std::string>> keys;
    return Json::parse(line, [&](int, Json::parse_event_t event, Json & parsed) {
        if (event == Json::parse_event_t::object_start) keys.emplace_back();
        else if (event == Json::parse_event_t::object_end) keys.pop_back();
        else if (event == Json::parse_event_t::key && !keys.back().insert(parsed.get<std::string>()).second)
            throw std::runtime_error("duplicate_json_key");
        return true;
    });
}

} // namespace

Context capture_context() noexcept {
    Context result;
    for (;;) {
        const auto before = context_version.load();
        if (before & 1) continue;
        result = {active_request.load(), active_operation.load(), active_phase.load()};
        if (before == context_version.load()) return result;
    }
}

std::string serialize_context(Context context) {
    if (!context.request_id) return {};
    return "{\"request_id\":" + std::to_string(context.request_id) +
        ",\"operation_id\":" + (context.operation_id ? std::to_string(context.operation_id) : "null") +
        ",\"phase\":" + (context.operation_id ?
            (context.phase == Phase::prefill ? "\"prefill\"" : "\"decode\"") : "null") +
        ",\"included\":true}";
}

uint64_t next_cpu_interval_sequence(Context context) noexcept {
    const auto active = capture_context();
    return context.operation_id && context.request_id == active.request_id &&
        context.operation_id == active.operation_id ?
        cpu_interval_sequence.fetch_add(1, std::memory_order_relaxed) + 1 : 0;
}

std::string log_context() { return serialize_context(capture_context()); }

std::string serialize_measurement(const Measurement & measurement) {
    Json record = {{"schema", "gemmini.cycle"}, {"version", 2},
        {"record_type", "RESOURCE_SAMPLE"}, {"role", "summary_resource"},
        {"sequence", measurement.sequence}, {"execution_id", cycle::host_execution_id()}};
    switch (measurement.kind) {
        case Measurement::Kind::cpu:
            record["kind"] = "cpu";
            record["start"] = cpu_sample_json(measurement.start);
            record["end"] = cpu_sample_json(measurement.end);
            record["cycles"] = measurement.cycles_reason.empty() ? Json(measurement.cycles) : Json();
            record["cycles_reason"] = measurement.cycles_reason.empty() ? Json() : Json(measurement.cycles_reason);
            record["thread_ns"] = measurement.thread_reason.empty() ? Json(measurement.thread_ns) : Json();
            record["thread_reason"] = measurement.thread_reason.empty() ? Json() : Json(measurement.thread_reason);
            break;
        case Measurement::Kind::wall:
            record["kind"] = "wall";
            record["begin_ns"] = measurement.begin_ns;
            record["end_ns"] = measurement.end_ns;
            break;
        case Measurement::Kind::wall_gap:
            record["kind"] = "wall_gap";
            record["reason"] = measurement.reason;
            break;
        case Measurement::Kind::npu:
            record["kind"] = "npu";
            record["backend"] = measurement.backend;
            record["domain"] = measurement.domain;
            record["metric"] = measurement.metric;
            record["cycles"] = measurement.collected ? Json(measurement.cycles) : Json();
            record["collected"] = measurement.collected;
            record["reason"] = measurement.collected ? Json() : Json(measurement.reason);
            break;
    }
    return record.dump();
}

void reset() {
    log::cycle.flush();
    publish_context({});
    request_sequence = operation_sequence = event_sequence = frequency_hz = 0;
    token_sequence = 0;
    resource_sequence.store(0);
    cpu_interval_sequence.store(0);
    emit_event("session_start", 0);
}

void finish_recording() {
    log::cycle.flush();
    emit_event("session_end", 0);
}

void start_request(uint64_t start_ns) {
    if (capture_context().request_id) finish_request(start_ns);
    token_sequence = 0;
    publish_context({++request_sequence, 0, Phase::prefill});
    emit_event("request_start", start_ns);
}

void begin_operation(Phase phase, uint64_t start_ns) {
    auto context = capture_context();
    if (!context.request_id) return;
    if (context.operation_id) end_operation(start_ns, false);
    context.operation_id = ++operation_sequence;
    context.phase = phase;
    resource_sequence.store(0);
    cpu_interval_sequence.store(0);
    publish_context(context);
    emit_event("operation_start", start_ns);
}

void end_operation(uint64_t end_ns, bool success) {
    auto context = capture_context();
    if (!context.operation_id) return;
    log::cycle.flush();
    emit_event("operation_end", end_ns, success);
    context.operation_id = 0;
    publish_context(context);
}

void token_ready(uint64_t ready_ns, std::optional<int32_t> token_id) {
    if (!capture_context().request_id) return;
    log::cycle.flush();
    emit_event("token_ready", ready_ns, true, token_id);
}

void finish_request(uint64_t end_ns) {
    if (!capture_context().request_id) return;
    end_operation(end_ns, false);
    log::cycle.flush();
    emit_event("request_end", end_ns);
    publish_context({});
}

void set_npu_frequency(uint64_t hz) {
    frequency_hz = hz;
    emit_event("configuration", 0);
}

void record_cpu(const gemmini_cpu_sample & start, const gemmini_cpu_sample & end,
                const gemmini_cpu_totals & interval) noexcept {
#if LOG_CYCLE
    if (!trace::inference_context(start.trace).operation_id) return;
    try {
        Measurement measurement;
        measurement.kind = Measurement::Kind::cpu;
        measurement.trace = start.trace;
        measurement.start = start;
        measurement.end = end;
        measurement.cycles = interval.cycles;
        measurement.thread_ns = interval.thread_cpu_ns;
        measurement.cycles_reason = interval.cycles_reason ? interval.cycles_reason :
            (interval.interval_count == 1 && interval.cycles_valid_count == 1 ? "" : "invalid_cpu_interval");
        measurement.thread_reason = interval.thread_cpu_reason ? interval.thread_cpu_reason :
            (interval.interval_count == 1 && interval.thread_cpu_valid_count == 1 ? "" : "invalid_cpu_interval");
        enqueue(std::move(measurement));
    } catch (...) { log::cycle.report_failure("CPU resource capture"); }
#else
    (void) start; (void) end; (void) interval;
#endif
}

void record_cpu_wall(uint64_t begin_ns, uint64_t end_ns) noexcept {
    Measurement measurement;
    measurement.kind = Measurement::Kind::wall;
    measurement.begin_ns = begin_ns;
    measurement.end_ns = end_ns;
    enqueue(std::move(measurement));
}

void incomplete_cpu_wall(const char * reason) noexcept {
#if LOG_CYCLE
    if (!trace::inference_context(gemmini_trace_capture()).operation_id) return;
    try {
        Measurement measurement;
        measurement.kind = Measurement::Kind::wall_gap;
        measurement.reason = reason ? reason : "cpu_stage_coverage_incomplete";
        enqueue(std::move(measurement));
    } catch (...) { log::cycle.report_failure("CPU coverage capture"); }
#else
    (void) reason;
#endif
}

void record_npu(const char * backend, const char * domain, const char * metric,
                uint64_t cycles, bool collected, const char * reason) noexcept {
#if LOG_CYCLE
    if (!trace::inference_context(gemmini_trace_capture()).operation_id) return;
    try {
        Measurement measurement;
        measurement.kind = Measurement::Kind::npu;
        measurement.backend = backend ? backend : "unknown";
        measurement.domain = domain ? domain : "unknown";
        measurement.metric = metric ? metric : "unknown";
        measurement.cycles = cycles;
        measurement.collected = collected;
        if (!collected) measurement.reason = reason ? reason : "unavailable_device_cycles";
        enqueue(std::move(measurement));
    } catch (...) { log::cycle.report_failure("NPU resource capture"); }
#else
    (void) backend; (void) domain; (void) metric; (void) cycles; (void) collected; (void) reason;
#endif
}

Summary read_summary(std::istream & input) {
    Summary result;
    State state;
    bool started = false, finished = false;
    uint64_t events = 0;
    std::string execution;
    try {
        if (!input.good()) throw std::runtime_error("log_read_failure");
        std::string line;
        while (std::getline(input, line)) {
            if (input.eof()) throw std::runtime_error("truncated_jsonl");
            if (line.empty()) throw std::runtime_error("empty_jsonl_record");
            const auto record = parse_line(line);
            if (!record.is_object()) throw std::runtime_error("invalid_jsonl_record");
            const auto type = record.value("record_type", std::string());
            if (type == "LOG_ERROR" || type == "LOG_FAILURE") throw std::runtime_error("log_collection_failure");
            const bool event = type == "INFERENCE_EVENT";
            const bool resource = type == "RESOURCE_SAMPLE";
            if (!event && !resource) {
                const bool included = record.contains("inference_context") && !record.at("inference_context").is_null();
                if (included) match_context(record, state);
                if (type == "CPU_INTERVAL" && included && state.operation.active) {
                    ++state.operation.cpu_interval_records;
                    if (record.contains("cpu_interval_sequence"))
                        state.operation.cpu_interval_sequences.push_back(number(record, "cpu_interval_sequence"));
                }
                if (started && record.contains("execution_id") && text_value(record, "execution_id") != execution)
                    throw std::runtime_error("mixed_execution_id");
                if (started && record.contains("host_timing")) {
                    const auto & timing = record.at("host_timing");
                    if (timing.contains("execution_id") && text_value(timing, "execution_id") != execution)
                        throw std::runtime_error("mixed_execution_id");
                }
                continue;
            }
            if (record.at("schema") != "gemmini.cycle" || number(record, "version") != 2)
                throw std::runtime_error("unsupported_record_schema");
            const auto id = text_value(record, "execution_id");
            if (execution.empty()) execution = id;
            if (id != execution) throw std::runtime_error("mixed_execution_id");
            if (finished) throw std::runtime_error("records_after_session_end");
            if (resource) {
                if (!started) throw std::runtime_error("missing_session_start");
                consume_measurement(record, state);
                result.peak_operation_samples = std::max(result.peak_operation_samples, state.operation.sequences.size());
                continue;
            }
            if (number(record, "event_sequence") != ++events) throw std::runtime_error("missing_or_duplicate_event");
            const auto name = text_value(record, "event");
            const auto timestamp = number(record, "timestamp_ns");
            const auto hz = number(record, "npu_frequency_hz");
            if (state.frequency_hz && hz != state.frequency_hz) throw std::runtime_error("changed_npu_frequency");
            state.frequency_hz = hz;
            if (name == "session_start") {
                if (started) throw std::runtime_error("duplicate_session_start");
                started = true;
            } else if (!started) throw std::runtime_error("missing_session_start");
            else if (name == "configuration") {
                if (state.operation.active) throw std::runtime_error("configuration_during_operation");
            } else if (name == "request_start") {
                if (state.request.active) throw std::runtime_error("missing_request_end");
                check_latency_boundary(record, "tokenized_prompt_before_inference", state);
                state.request = {};
                state.request.active = true;
                state.request.start = timestamp;
                ++state.requests;
                match_context(record, state);
            } else if (name == "operation_start") {
                if (!state.request.active || state.operation.active) throw std::runtime_error("invalid_operation_start");
                const auto phase = text_value(record.at("inference_context"), "phase");
                if (phase != "prefill" && phase != "decode") throw std::runtime_error("invalid_phase");
                state.operation.active = true;
                state.operation.id = ++state.operations;
                state.operation.phase = phase == "prefill" ? Phase::prefill : Phase::decode;
                state.operation.start = timestamp;
                if (timestamp < state.request.start || timestamp < state.request.last_operation_end)
                    throw std::runtime_error("invalid_operation_order");
                match_context(record, state);
            } else if (name == "operation_end") {
                if (!state.operation.active) throw std::runtime_error("missing_operation_start");
                match_context(record, state);
                auto & operation = state.operation;
                const auto expected = number(record, "resource_samples");
                if (expected != operation.sequences.size()) throw std::runtime_error("missing_resource_samples");
                std::sort(operation.sequences.begin(), operation.sequences.end());
                for (size_t index = 0; index < operation.sequences.size(); ++index) {
                    if (operation.sequences[index] != index + 1) throw std::runtime_error("missing_or_duplicate_resource");
                }
                if (record.contains("cpu_interval_samples")) {
                    const auto expected_cpu = number(record, "cpu_interval_samples");
                    if (expected_cpu != operation.cpu_interval_records ||
                        expected_cpu != operation.cpu_interval_sequences.size())
                        throw std::runtime_error("missing_cpu_interval_samples");
                    std::sort(operation.cpu_interval_sequences.begin(), operation.cpu_interval_sequences.end());
                    for (size_t index = 0; index < operation.cpu_interval_sequences.size(); ++index) {
                        if (operation.cpu_interval_sequences[index] != index + 1)
                            throw std::runtime_error("missing_or_duplicate_cpu_interval");
                    }
                } else state.cpu_interval_coverage_verified = false;
                operation.end = timestamp;
                operation.success = boolean(record, "success");
                reduce_operation(state, operation);
                state.request.last_operation_end = std::max(operation.start, operation.end);
                state.operation = {};
            } else if (name == "token_ready") {
                if (!state.request.active || state.operation.active) throw std::runtime_error("invalid_token_event");
                match_context(record, state);
                auto & request = state.request;
                check_latency_boundary(record, "sampling_and_accept_complete", state);
                if (record.contains("token_index") && number(record, "token_index") != request.tokens)
                    throw std::runtime_error("invalid_token_index");
                if (record.contains("token_id") && !record.at("token_id").is_null()) {
                    const auto & token_id = record.at("token_id");
                    if (!token_id.is_number_integer() ||
                        (token_id.is_number_unsigned() && token_id.get<uint64_t>() > std::numeric_limits<int32_t>::max()) ||
                        (!token_id.is_number_unsigned() && (token_id.get<int64_t>() < std::numeric_limits<int32_t>::min() ||
                            token_id.get<int64_t>() > std::numeric_limits<int32_t>::max())))
                        throw std::runtime_error("invalid_token_id");
                }
                if (timestamp < request.start || timestamp < request.last_operation_end ||
                    (request.tokens && timestamp < request.last_token))
                    request.token_reason = "clock_regression";
                if (!request.tokens) request.first_token = timestamp;
                request.last_token = timestamp;
                ++request.tokens;
                ++state.tokens;
            } else if (name == "request_end") {
                if (!state.request.active || state.operation.active) throw std::runtime_error("invalid_request_end");
                match_context(record, state);
                finish_request_replay(state, timestamp);
            } else if (name == "session_end") {
                if (!boolean(record, "log_healthy")) throw std::runtime_error("log_collection_failure");
                if (state.request.active || state.operation.active) throw std::runtime_error("incomplete_recording");
                finished = true;
            } else throw std::runtime_error("unknown_inference_event");
        }
        if (input.bad()) throw std::runtime_error("log_read_failure");
        if (!started || !finished) throw std::runtime_error("incomplete_recording");
        result.available = true;
        result.json_ = serialize_state(state);
        const std::unique_ptr<FILE, decltype(&std::fclose)> output(std::tmpfile(), &std::fclose);
        if (!output) throw std::runtime_error("summary_text_failure");
        print_state(output.get(), state);
        std::rewind(output.get());
        char buffer[4096];
        while (const auto length = std::fread(buffer, 1, sizeof(buffer), output.get())) result.text_.append(buffer, length);
        if (std::ferror(output.get())) throw std::runtime_error("summary_text_failure");
    } catch (const Json::exception &) {
        result.available = false;
        result.reason = "malformed_record";
    } catch (const std::exception & error) {
        result.available = false;
        result.reason = error.what();
    }
    return result;
}

Summary read_summary(const std::filesystem::path & path) {
    if (path.empty()) {
        Summary result;
        result.reason = "file_output_unavailable";
        return result;
    }
    std::ifstream input(path);
    return read_summary(input);
}

std::string Summary::serialize() const {
    if (available) return json_;
    return "{\"schema\":\"gemmini.cycle\",\"version\":2,\"record_type\":\"FINAL_INFERENCE_SUMMARY\","
        "\"available\":false,\"reason\":" + quote(reason.empty() ? "summary_unavailable" : reason) +
        ",\"phases\":[]}";
}

void Summary::print(FILE * output) const {
    if (!output) return;
    if (available) std::fwrite(text_.data(), 1, text_.size(), output);
    else std::fprintf(output, "\nInference performance: n/a (%s)\n",
                      reason.empty() ? "summary_unavailable" : reason.c_str());
}

std::string serialize() {
    if (!log::cycle.flush() || !log::cycle.healthy()) {
        Summary result;
        result.reason = "log_collection_failure";
        return result.serialize();
    }
    return read_summary(log::cycle.output_path()).serialize();
}

void print(FILE * output) {
    if (!log::cycle.flush() || !log::cycle.healthy()) {
        Summary result;
        result.reason = "log_collection_failure";
        result.print(output);
        return;
    }
    read_summary(log::cycle.output_path()).print(output);
}

} // namespace ggml::gemmini::performance
