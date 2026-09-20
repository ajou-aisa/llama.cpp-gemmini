#include <gemmini/host-timing.hpp>
#include <gemmini/log.hpp>
#include <gemmini/cycle_reader.hpp>
#include <gemmini/performance.hpp>

#include <cstdio>
#include <ctime>
#include <string>
#include <thread>
#include <limits>
#include <sstream>

namespace {
bool check(bool condition, const char * message) {
    if (!condition) std::fprintf(stderr, "FAIL: %s\n", message);
    return condition;
}

bool check_thread_cpu_timing() {
    using namespace ggml::gemmini::cycle;
    const HostSample start{100, 7, 20, true};
    const HostSample end{200, 7, 50, true};
    const HostSample cross_thread{200, 8, 50, true};
    const HostSample regression{200, 7, 19, true};
    const HostSample missing_tid{200, 0, 50, true};
    const HostSample unavailable_cpu{200, 7, 50, false};
    const std::string valid = serialize_thread_cpu_timing(start, end);
    const std::string missing_start = serialize_thread_cpu_timing({}, end);
    const std::string missing_end = serialize_thread_cpu_timing(start, {});
    const auto invalid_duration = [&](const HostSample &finish) {
        const std::string json = serialize_thread_cpu_timing(start, finish);
        return json.find("\"duration_ns\":null") != std::string::npos &&
            json.find("\"valid\":false") != std::string::npos;
    };
    const HostSample real_start = read_host_sample();
    const HostSample real_end = read_host_sample();
    bool ok =
        check(valid == "{\"clock\":\"thread_cpu\",\"unit\":\"nanosecond\",\"start_ns\":20,"
                       "\"end_ns\":50,\"duration_ns\":30,\"valid\":true}",
              "thread CPU timing uses its own counter, independent of wall time") &&
        check(invalid_duration(cross_thread) && invalid_duration(regression) &&
              invalid_duration(missing_tid) && invalid_duration(unavailable_cpu),
              "cross-thread, regressing, unidentified and unavailable CPU samples cannot produce a duration") &&
        check(missing_start.find("\"start_ns\":null") != std::string::npos &&
              missing_start.find("\"end_ns\":50") != std::string::npos &&
              missing_start.find("\"valid\":false") != std::string::npos &&
              missing_end.find("\"end_ns\":null") != std::string::npos &&
              missing_end.find("\"valid\":false") != std::string::npos,
              "unavailable CPU clock endpoints remain null") &&
        check(serialize_thread_cpu_timing(start, start).find("\"duration_ns\":0,\"valid\":true") !=
                  std::string::npos,
              "collected zero thread CPU duration is valid") &&
        check(real_start.tid == host_thread_id() && real_start.tid == real_end.tid &&
              real_end.ns >= real_start.ns,
              "shared host samples capture a stable thread identity and monotonic wall clock");
#if (defined(__linux__) || defined(__APPLE__)) && defined(CLOCK_THREAD_CPUTIME_ID)
    ok = check(real_start.thread_cpu_valid && real_end.thread_cpu_valid &&
               real_end.thread_cpu_ns >= real_start.thread_cpu_ns,
               "supported thread CPU clock produces valid nonregressing samples") && ok;
#else
    ok = check(!real_start.thread_cpu_valid && !real_end.thread_cpu_valid,
               "unsupported thread CPU clocks stay unavailable") && ok;
#endif
    return ok;
}

bool check_cycle_write_timing() {
    using namespace ggml::gemmini::log;
    FILE *output = std::tmpfile();
    if (!check(output != nullptr, "create real logger output")) return false;
    CycleLog logger(output);
    CycleWriteTiming outer, inner;
    const auto start = ggml::gemmini::cycle::read_host_sample();
    {
        ScopedCycleWriteTiming scope(outer);
        logger.write_json("{\"probe\":1}");
        {
            ScopedCycleWriteTiming nested(inner);
            logger.write_json("{\"probe\":2}");
        }
        std::thread other([&] { logger.write_json("{\"probe\":3}"); });
        other.join();
        logger.write_json("{\"probe\":4}");
    }
    const auto end = ggml::gemmini::cycle::read_host_sample();
    logger.write_json("{\"probe\":5}");
    std::rewind(output);
    char buffer[256]{};
    const std::string contents(buffer, std::fread(buffer, 1, sizeof(buffer), output));
    bool ok;
#if EXPECT_LOG_CYCLE
    ok = check(outer.valid && inner.valid && outer.calls == 2 && inner.calls == 1,
               "nested probes restore their parent and exclude other threads and unscoped writes") &&
         check(contents == "{\"probe\":1}\n{\"probe\":2}\n{\"probe\":3}\n"
                           "{\"probe\":4}\n{\"probe\":5}\n",
               "scoped measurement preserves the actual JSONL output") &&
         check(outer.io_ns + inner.io_ns > 0 &&
               outer.mutex_wait_ns + outer.io_ns + inner.mutex_wait_ns + inner.io_ns <= end.ns - start.ns,
               "mutex and I/O subintervals fit inside the enclosing wall interval");
#else
    (void) start;
    (void) end;
    ok = check(!outer.valid && !inner.valid && outer.calls == 0 && inner.calls == 0 && contents.empty(),
               "compiled-off logger produces no successful measurement or output");
#endif
    CycleWriteTiming empty;
    {
        ScopedCycleWriteTiming scope(empty);
        logger.write_json("");
    }
    ok = check(!empty.valid && empty.calls == 0, "empty output scope is unavailable") && ok;
    logger.set_output(nullptr);
    CycleWriteTiming no_output;
    {
        ScopedCycleWriteTiming scope(no_output);
        logger.write_json("{}");
    }
    ok = check(!no_output.valid && no_output.io_ns == 0, "null output cannot look like a measured write") && ok;
#if EXPECT_LOG_CYCLE
    for (const auto fault : {testing::LogFault::write, testing::LogFault::flush}) {
        logger.set_output(output);
        CycleWriteTiming failed;
        {
            ScopedCycleWriteTiming scope(failed);
            logger.write_json("{}");
            testing::set_log_fault(fault);
            logger.write_json("{}");
            logger.write_json("{}");
        }
        testing::clear_log_fault();
        ok = check(!failed.valid && failed.calls == 3,
                   "write/flush failure invalidates prior successful output and disabled writes") && ok;
    }
    logger.set_output(output);
    CycleWriteTiming failed_lock;
    bool caught = false;
    {
        ScopedCycleWriteTiming scope(failed_lock);
        testing::set_log_fault(testing::LogFault::mutex);
        try { logger.write_json("{}"); } catch (...) { caught = true; }
    }
    testing::clear_log_fault();
    ok = check(caught && !failed_lock.valid && failed_lock.calls == 1 && failed_lock.io_ns == 0,
               "lock exceptions leave an invalid probe even when caught inside its scope") && ok;
    CycleWriteTiming failed_serialization;
    {
        ScopedCycleWriteTiming scope(failed_serialization);
        logger.write_json("{}");
        logger.report_failure("injected serialization");
        logger.write_json("{}");
    }
    ok = check(!failed_serialization.valid && failed_serialization.calls == 2,
               "reported serialization failure keeps the probe invalid across later writes") && ok;
#endif
    logger.set_output(nullptr);
    std::fclose(output);
    return ok;
}

bool check_worker_cpu_totals() {
    using namespace ggml::gemmini::cycle;
    gemmini_cpu_sample start{}, end{};
    start.tid = end.tid = 7;
    start.thread_cpu_valid = end.thread_cpu_valid = 1;
    start.thread_cpu_ns = 20;
    end.thread_cpu_ns = 50;
    start.counter = 100; end.counter = 200;
    start.native_valid = end.native_valid = 1;
    start.native_source = end.native_source = GEMMINI_CPU_COUNTER_THREAD_PERF;
    start.owner_token = end.owner_token = 7;
    start.generation = end.generation = 1;
    gemmini_cpu_totals one{}, combined{};
    gemmini_cpu_timing_add(&one, &start, &end);
    gemmini_cpu_timing_merge(&combined, &one);
    gemmini_cpu_timing_merge(&combined, &one);
    if (!check(combined.interval_count == 2 && combined.thread_cpu_valid_count == 2 &&
               combined.thread_cpu_ns == 60 && combined.thread_cpu_reason == nullptr,
               "worker CPU intervals sum independently of cycle detail output")) return false;
    end.tid = 8;
    gemmini_cpu_timing_add(&combined, &start, &end);
    if (!check(combined.interval_count == 3 && combined.thread_cpu_valid_count == 2 &&
               serialize_cpu_totals(combined).find("\"thread_cpu_ns\":null") != std::string::npos,
               "one invalid worker keeps the aggregate incomplete rather than publishing a partial sum")) return false;
    combined = one;
    combined.thread_cpu_ns = std::numeric_limits<uint64_t>::max();
    gemmini_cpu_timing_merge(&combined, &one);
    if (!check(std::string(combined.thread_cpu_reason) == "aggregate_overflow" &&
               serialize_cpu_totals(combined).find("\"thread_cpu_ns\":null") != std::string::npos,
               "overflow never wraps into a valid CPU duration")) return false;
#if defined(__linux__) && defined(__aarch64__)
    if (!check(one.cycles == 100 && one.cycles_valid_count == 1 && !one.cycles_reason,
               "native worker CPU cycles retain a valid same-owner delta")) return false;
    end.tid = start.tid;
    end.generation = 2;
    gemmini_cpu_totals changed_owner{};
    gemmini_cpu_timing_add(&changed_owner, &start, &end);
    if (!check(changed_owner.cycles_valid_count == 0 &&
               std::string(changed_owner.cycles_reason) == "event_generation_mismatch",
               "counter reinitialization cannot manufacture a worker cycle total")) return false;
#else
    if (!check(one.cycles_valid_count == 0 &&
               serialize_cpu_totals(one).find("\"cycles\":null") != std::string::npos,
               "host timer ticks are never reported as CPU cycles")) return false;
#endif
    const auto reads = read_count_for_test();
    const auto actual = gemmini_cpu_timing_read();
#if EXPECT_LOG_CYCLE
    if (!check(actual.tid == host_thread_id() && actual.ns != 0,
               "worker sampler records thread identity and host time in compact and detail modes")) return false;
#if (defined(__linux__) || defined(__APPLE__)) && defined(CLOCK_THREAD_CPUTIME_ID)
    if (!check(actual.thread_cpu_valid,
               "worker sampler retains thread CPU time in compact and detail modes")) return false;
#else
    if (!check(!actual.thread_cpu_valid,
               "unsupported thread CPU clocks remain unavailable")) return false;
#endif
#else
    if (!check(actual.ns == 0 && actual.tid == 0 && read_count_for_test() == reads,
               "compiled-off worker timing reads no clocks")) return false;
#endif
    (void) reads;
    return true;
}

void record_test_cpu(uint64_t start_ns, uint64_t end_ns, uint64_t tid,
                     uint64_t cpu_ns, uint64_t cycles, const char * reason = nullptr,
                     uint64_t generation = 1, uint64_t owner = 0) {
    gemmini_cpu_sample start{}, end{};
    start.ns = start_ns; end.ns = end_ns;
    start.tid = end.tid = tid;
    start.thread_cpu_valid = end.thread_cpu_valid = 1;
    start.thread_cpu_ns = 0; end.thread_cpu_ns = cpu_ns;
    start.counter = 0; end.counter = cycles;
    start.native_valid = end.native_valid = 1;
    start.native_source = end.native_source = GEMMINI_CPU_COUNTER_THREAD_PERF;
    start.owner_token = end.owner_token = owner ? owner : tid;
    start.generation = end.generation = generation;
    gemmini_cpu_totals total{};
    total.interval_count = 1;
    total.cycles = cycles;
    total.thread_cpu_ns = cpu_ns;
    total.cycles_valid_count = total.thread_cpu_valid_count = reason ? 0 : 1;
    total.cycles_reason = total.thread_cpu_reason = reason;
    ggml::gemmini::performance::record_cpu(start, end, total);
}

bool contains(const std::string & value, const char * expected) {
    return value.find(expected) != std::string::npos;
}

std::string read_output(FILE * output) {
    std::fflush(output);
    std::rewind(output);
    std::string contents;
    char buffer[4096];
    while (const size_t length = std::fread(buffer, 1, sizeof(buffer), output)) {
        contents.append(buffer, length);
    }
    std::fseek(output, 0, SEEK_END);
    return contents;
}

ggml::gemmini::performance::Summary replay(const std::string & contents) {
    std::istringstream input(contents);
    return ggml::gemmini::performance::read_summary(input);
}

struct Recording {
    FILE * output = std::tmpfile();

    Recording() {
        if (output) {
            ggml::gemmini::log::cycle.set_output(output);
            ggml::gemmini::performance::reset();
        }
    }
    ~Recording() {
        ggml::gemmini::log::cycle.set_output(stderr);
        if (output) std::fclose(output);
    }
    ggml::gemmini::performance::Summary summary() { return replay(read_output(output)); }
};

[[maybe_unused]] std::string phase_json(const std::string & json, const char * phase) {
    const auto begin = json.find(std::string("{\"phase\":\"") + phase + '"');
    if (begin == std::string::npos) return {};
    const auto end = json.find("{\"phase\":", begin + 1);
    return json.substr(begin, end == std::string::npos ? end : end - begin);
}

bool check_cpu_interval_record() {
    using namespace ggml::gemmini;
    gemmini_cpu_sample start{}, end{};
    start.ns = 100; end.ns = 200;
    start.tid = end.tid = 7;
    start.thread_cpu_ns = 20; end.thread_cpu_ns = 50;
    start.thread_cpu_valid = end.thread_cpu_valid = 1;
    start.counter = 400; end.counter = 470;
    start.native_source = end.native_source = GEMMINI_CPU_COUNTER_THREAD_PERF;
    start.native_valid = end.native_valid = 1;
    start.owner_token = end.owner_token = 17;
    start.generation = end.generation = 3;
    gemmini_cycle_record_v2 identity{};
    identity.interval.layer = "layer\"\\\n";
    identity.interval.op = "packing\tstage";
    identity.identity_mask = GEMMINI_CYCLE_HAS_RUN_ID | GEMMINI_CYCLE_HAS_STRIPE_ID |
        GEMMINI_CYCLE_HAS_SLOT | GEMMINI_CYCLE_HAS_NODE_ID | GEMMINI_CYCLE_HAS_WORKER_ID;
    identity.run_id = 11; identity.stripe_id = 12; identity.slot = 13;
    identity.node_id = 14; identity.worker_id = 0;
    static_assert(noexcept(gemmini_cpu_timing_record(&identity, &start, &end)),
                  "raw interval recording must not throw across a C boundary");
    FILE *output = std::tmpfile();
    if (!check(output != nullptr, "create raw CPU JSONL output")) return false;
    log::cycle.set_output(output);
    performance::reset();
    performance::start_request(90);
    performance::begin_operation(performance::Phase::prefill, 90);
    gemmini_cpu_timing_record(&identity, &start, &end);
    identity.identity_mask = 0;
    end.tid = 8;
    gemmini_cpu_timing_record(&identity, &start, &end);
    end.tid = start.tid;
    start.native_source = end.native_source = GEMMINI_CPU_COUNTER_UNAVAILABLE;
    gemmini_cpu_timing_record(&identity, &start, &end);
    performance::end_operation(210, true);
    performance::finish_request(220);
    performance::finish_recording();
    const std::string json = read_output(output);
    const auto summary = replay(json);
    log::cycle.set_output(stderr);
    std::fclose(output);
    bool ok = true;
#if EXPECT_LOG_CYCLE
    ok = check(summary.available && contains(summary.serialize(), "\"thread_cpu_ns\":null"),
               "writing diagnostic CPU intervals alone does not add summary resource totals");
    ok = check(contains(json, "\"cpu_interval_sequence\":3") &&
               contains(json, "\"cpu_interval_samples\":3") &&
               contains(summary.serialize(), "\"cpu_interval_coverage\":\"verified\""),
               "operation completion accounts for every producer-emitted diagnostic CPU interval") && ok;
    std::string dropped = json;
#if EXPECT_CYCLE_DETAIL
    const char *cpu_marker = "\"record_type\":\"CPU_INTERVAL\"";
#else
    const char *cpu_marker = "\"kind\":\"cpu\"";
#endif
    const auto raw_marker = dropped.find(cpu_marker);
    const auto raw_begin = dropped.rfind('\n', raw_marker) + 1;
    const auto raw_end = dropped.find('\n', raw_marker) + 1;
    dropped.erase(raw_begin, raw_end - raw_begin);
    ok = check(!replay(dropped).available && replay(dropped).reason == "missing_cpu_interval_samples",
               "a dropped diagnostic CPU interval invalidates new recording completeness") && ok;
    std::string duplicated = json;
    const auto second = duplicated.find("\"cpu_interval_sequence\":2");
    duplicated.replace(second, std::string("\"cpu_interval_sequence\":2").size(), "\"cpu_interval_sequence\":1");
    ok = check(!replay(duplicated).available && replay(duplicated).reason == "missing_or_duplicate_cpu_interval",
               "duplicate raw sequence numbers cannot replace a missing CPU interval") && ok;
    std::string legacy = json;
    const auto count = legacy.find("\"cpu_interval_samples\":3,");
    legacy.erase(count, std::string("\"cpu_interval_samples\":3,").size());
    ok = check(replay(legacy).available &&
               contains(replay(legacy).serialize(), "\"cpu_interval_coverage\":\"unverified_legacy_log\""),
               "old logs retain canonical totals without claiming raw stage completeness") && ok;
    ok = check(replay(json + "{\"record_type\":\"EXSIA_TIMELINE\",\"inference_context\":null}\n").available &&
               !replay(json + "{\"record_type\":\"EXSIA_TIMELINE\",\"inference_context\":[]}\n").available,
               "explicitly excluded diagnostics do not acquire a request but malformed contexts fail") && ok;
    std::string mismatched = json;
    const auto diagnostic = mismatched.find(cpu_marker);
    const auto request = mismatched.find("\"request_id\":1", diagnostic);
    if (check(request != std::string::npos && request < mismatched.find('\n', diagnostic),
              "CPU diagnostic records carry their inference context")) {
        mismatched.replace(request, std::string("\"request_id\":1").size(), "\"request_id\":2");
        ok = check(!replay(mismatched).available,
                   "diagnostic context mismatches invalidate replay without adding resources") && ok;
    } else ok = false;
#if EXPECT_CYCLE_DETAIL
    ok = check(contains(json, "\"record_type\":\"CPU_INTERVAL\"") &&
               contains(json, "\"layer\":\"layer\\\"\\\\\\n\"") &&
               contains(json, "\"op\":\"packing\\tstage\"") &&
               contains(json, "\"run_id\":11,\"stripe_id\":12,\"slot\":13,\"node_id\":14,\"worker_id\":0") &&
               contains(json, "\"run_id\":null,\"stripe_id\":null,\"slot\":null,\"node_id\":null,\"worker_id\":null"),
               "detail CPU records preserve full escaped identity including explicit null IDs") && ok;
    ok = check(contains(json, "\"start\":{\"value\":400,\"valid\":true") &&
               contains(json, "\"end\":{\"value\":470,\"valid\":true") &&
               contains(json, "\"owner_token\":17,\"generation\":3") &&
               contains(json, "\"start_ns\":100,\"end_ns\":200,\"start_tid\":7,\"end_tid\":7") &&
               contains(json, "\"thread_cpu_timing\":{\"clock\":\"thread_cpu\",\"unit\":\"nanosecond\",\"start_ns\":20,\"end_ns\":50,\"duration_ns\":30,\"valid\":true}"),
               "detail records retain PMU provenance, wall endpoints and worker CPU samples") && ok;
    ok = check(contains(json, "\"value\":null,\"valid\":false,\"source\":null,\"reason\":\"not_thread_cpu_counter\"") &&
               contains(json, "\"duration_ns\":null,\"valid\":false"),
               "detail records retain unavailable native and cross-thread timing status") && ok;
#else
    const auto compact_begin = json.rfind('\n', json.find(cpu_marker)) + 1;
    const auto compact_end = json.find('\n', compact_begin);
    const std::string compact = json.substr(compact_begin, compact_end - compact_begin);
    ok = check(compact.rfind("{\"op\":", 0) == 0 &&
               compact.find("\"schema\"") == std::string::npos &&
               compact.find("\"version\"") == std::string::npos &&
               compact.find("\"record_type\"") == std::string::npos,
               "compact CPU records start at op and omit repeated schema headers") && ok;
    ok = check(contains(json, "\"op\":\"packing\\tstage\",\"kind\":\"cpu\",\"layer\":\"layer\\\"\\\\\\n\"") &&
               contains(json, "\"run_id\":11,\"stripe_id\":12,\"slot\":13,\"node_id\":14,\"worker_id\":0") &&
               !contains(compact, "\"host_timing\"") && !contains(compact, "\"thread_cpu_timing\"") &&
               !contains(compact, "\"native_cycles\""),
               "compact CPU records retain identity and one canonical cycle representation only") && ok;
    ok = check(contains(compact, "\"start\":400,\"end\":470") &&
               contains(compact, "\"ns_start\":100,\"ns_end\":200,\"tid\":7") &&
               !contains(compact, "\"owner_token\"") && !contains(compact, "\"generation\""),
               "compact CPU records keep cycle and shared timeline endpoints without PMU provenance") && ok;
#endif
#if defined(__linux__) && defined(__aarch64__)
    ok = check(contains(json, "\"delta\":70") && contains(json, "\"valid\":true") &&
               contains(json, "\"delta\":null") && contains(json, "\"valid\":false") &&
               contains(json, "\"reason\":\"thread_mismatch\""),
               "raw delta uses the same native interval validation as CPU totals") && ok;
#else
    ok = check(contains(json, "\"delta\":null") && contains(json, "\"valid\":false") &&
               contains(json, "\"reason\":\"not_thread_cpu_counter\""),
               "non-PMU hosts cannot publish valid native CPU cycles") && ok;
#endif
#else
    ok = check(json.empty() && !summary.available,
               "compiled-off raw CPU logging writes no records or available summary") && ok;
#endif
    start.native_source = end.native_source = GEMMINI_CPU_COUNTER_THREAD_PERF;
    end.native_valid = 0;
#if defined(__linux__) && defined(__aarch64__)
    end.native_reason = static_cast<uint8_t>(cycle::NativeCycleReason::counter_regression);
    const char *sample_reason = "\"reason\":\"counter_regression\"";
#else
    const char *sample_reason = "\"reason\":\"unavailable_sample\"";
#endif
    const auto invalid = cycle::serialize_cpu_native(start, end);
    ok = check(contains(invalid, "\"end\":{\"value\":470,\"valid\":false") &&
               contains(invalid, sample_reason) &&
               contains(invalid, "\"owner_token\":17,\"generation\":3"),
               "sample failures retain the original sampled value and provenance with invalid status") && ok;
    return ok;
}

bool check_cpu_resource_summary_from_timing_add() {
    using namespace ggml::gemmini::performance;
    Recording recording;
    if (!check(recording.output != nullptr, "create CPU resource recording")) return false;

    start_request(100);
    begin_operation(Phase::prefill, 100);
    gemmini_cpu_sample start{}, end{};
    start.ns = 110; end.ns = 150;
    start.tid = end.tid = 7;
    start.thread_cpu_ns = 20; end.thread_cpu_ns = 50;
    start.thread_cpu_valid = end.thread_cpu_valid = 1;
    start.counter = 100; end.counter = 180;
    start.native_valid = end.native_valid = 1;
    start.native_source = end.native_source = GEMMINI_CPU_COUNTER_THREAD_PERF;
    start.owner_token = end.owner_token = 9;
    start.generation = end.generation = 3;
    start.trace = end.trace = gemmini_trace_capture();
    gemmini_cpu_totals totals{};
    gemmini_cpu_timing_add(&totals, &start, &end);
    end_operation(160, true);
    token_ready(170);
    finish_request(180);
    finish_recording();

    const std::string output = read_output(recording.output);
    const auto summary = replay(output);
#if EXPECT_LOG_CYCLE
    const auto prefill = phase_json(summary.serialize(), "prefill");
    bool ok = check(summary.available &&
                    contains(output, "\"record_type\":\"RESOURCE_SAMPLE\"") &&
                    contains(output, "\"kind\":\"cpu\"") &&
                    contains(prefill, "\"thread_cpu_ns\":30,"),
                    "CPU timing add publishes the summary resource in compact and detail modes");
#if defined(__linux__) && defined(__aarch64__)
    ok = check(contains(prefill, "\"cpu_cycles\":80,"),
               "Linux AArch64 compact summary retains PMU CPU cycles") && ok;
#else
    ok = check(contains(prefill, "\"cpu_cycles\":null,\"cpu_cycles_reason\":\"not_thread_cpu_counter\""),
               "non-PMU hosts keep CPU cycles unavailable without losing thread CPU time") && ok;
#endif
    return ok;
#else
    return check(output.empty() && !summary.available,
                 "compiled-off logging publishes no CPU summary resource");
#endif
}

bool check_inference_summary() {
    using namespace ggml::gemmini::performance;
    Recording recording;
    if (!check(recording.output != nullptr, "create inference recording")) return false;
    bool ok = check(log_context().empty(), "logs outside a request have no inference context");
    begin_operation(Phase::prefill, 1);
    record_test_cpu(1, 90, 7, 999, 999);
    record_cpu_wall(1, 90);
    record_npu("warmup", "ignored", "work", 999, true, nullptr);
    end_operation(90, true);
    token_ready(90);
    start_request(100);
#if EXPECT_LOG_CYCLE
    const auto between_operations = log_context();
    ok = check(between_operations == "{\"request_id\":1,\"operation_id\":null,\"phase\":null,\"included\":true}",
               "request context leaves operation and phase unset between evaluations") && ok;
#endif
    begin_operation(Phase::prefill, 100);
#if EXPECT_LOG_CYCLE
    ok = check(log_context() == "{\"request_id\":1,\"operation_id\":1,\"phase\":\"prefill\",\"included\":true}",
               "prefill logs identify the first active operation") && ok;
#endif
    record_test_cpu(120, 150, 7, 20, 200);
    record_test_cpu(100, 200, 7, 60, 600);
    record_test_cpu(200, 220, 7, 10, 100);
    record_test_cpu(110, 190, 8, 30, 300);
    record_cpu_wall(100, 160);
    record_cpu_wall(140, 180);
    record_cpu_wall(190, 210);
    end_operation(220, true);
#if EXPECT_LOG_CYCLE
    ok = check(log_context() == between_operations,
               "ending an operation clears its log context without ending the request") && ok;
    const auto completed_prefill = recording.summary();
    ok = check(!completed_prefill.available && completed_prefill.reason == "incomplete_recording",
               "completed operations cannot make an unfinished recording available") && ok;
#endif
    token_ready(230);
    begin_operation(Phase::decode, 250);
#if EXPECT_LOG_CYCLE
    ok = check(log_context() == "{\"request_id\":1,\"operation_id\":2,\"phase\":\"decode\",\"included\":true}",
               "decode uses a distinct operation ID and phase") && ok;
#endif
    record_test_cpu(250, 300, 7, 30, 300);
    record_cpu_wall(250, 300);
    record_npu("im2p", "dense", "work", 1000, true, nullptr);
    record_npu("im2p", "rmd", "work", 400, true, nullptr);
#if EXPECT_LOG_CYCLE
    const auto active_decode = recording.summary();
    ok = check(!active_decode.available && active_decode.reason == "incomplete_recording",
               "active-operation replay cannot publish a partial summary") && ok;
#endif
    end_operation(300, false);
    token_ready(310);
    begin_operation(Phase::decode, 320);
    record_test_cpu(320, 370, 7, 20, 200);
    record_cpu_wall(330, 370);
    end_operation(370, true);
    token_ready(390);
    finish_request(400);
    ok = check(log_context().empty(), "finished requests do not label subsequent logs") && ok;
    set_npu_frequency(1000000000);
    finish_recording();
    const std::string recorded = read_output(recording.output);
    const auto summary = replay(recorded);
#if EXPECT_LOG_CYCLE
    std::string unconfigured = recorded;
    const std::string configured_frequency = "\"npu_frequency_hz\":1000000000";
    size_t frequency = 0;
    while ((frequency = unconfigured.find(configured_frequency, frequency)) != std::string::npos) {
        unconfigured.replace(frequency, configured_frequency.size(), "\"npu_frequency_hz\":0");
        ++frequency;
    }
    const auto without_frequency = replay(unconfigured);
    const std::string json = without_frequency.serialize();
    ok = check(without_frequency.available && summary.available,
               "completed recordings replay with and without configured NPU frequency") && ok;
    ok = check(contains(recorded, "\"token_step\":0") && contains(recorded, "\"token_step\":1") &&
               contains(recorded, "\"token_step\":2"),
               "operation events explicitly identify the generated ordinal targeted by prefill and decode") && ok;
    const std::string prefill = phase_json(json, "prefill");
    const std::string decode = phase_json(json, "decode");
    ok = check(contains(prefill, "\"cpu_cycles\":1000,") &&
                    contains(prefill, "\"thread_cpu_ns\":100,") &&
                    contains(prefill, "\"cpu_work_wall_ns\":100,"),
                    "summary deduplicates nested CPU samples per thread and unions overlapping CPU stages") &&
        check(contains(decode, "\"operations\":2,\"failed_operations\":1") &&
              contains(decode, "\"elapsed_ns\":100,") && contains(decode, "\"thread_cpu_ns\":50,") &&
              contains(decode, "\"cpu_cycles\":500,") && contains(decode, "\"cpu_work_wall_ns\":90,"),
              "prefill and decode retain separate operation totals including failed-operation resource cost") &&
        check(contains(json, "\"ttft_ns\":130,") && contains(json, "\"tpot_ns\":80,") &&
              contains(json, "\"tokens\":3,") && contains(json, "\"tpot_gaps\":2,"),
              "TTFT measures first readiness and TPOT averages the two actual token gaps") &&
        check(!contains(json, "warmup") && !contains(json, ":999,"),
              "timing before a request is ignored") &&
        check(contains(decode, "\"domain\":\"dense\"") && contains(decode, "\"domain\":\"rmd\"") &&
              contains(decode, "\"time_ns\":null,\"time_reason\":\"npu_frequency_unavailable\""),
              "device clock domains stay separate and absent frequency cannot manufacture NPU time") && ok;
    const auto timed = summary.serialize();
    ok = check(contains(timed, "\"time_ns\":1000,\"time_reason\":null") &&
               contains(timed, "\"time_ns\":400,\"time_reason\":null") &&
               contains(timed, "\"time_source\":\"cycles/configured_npu_frequency\""),
               "configured frequency converts each counter domain independently") && ok;
    ok = check(contains(recorded, "\"token_id\":null") && contains(recorded, "\"token_index\":0") &&
               contains(recorded, "\"token_index\":2"),
               "unknown token IDs remain null while emitted ordinals identify all three readiness events") && ok;
    FILE * output = std::tmpfile();
    if (!check(output != nullptr, "create final summary text output")) return false;
    summary.print(output);
    summary.print(output);
    ok = check(summary.serialize() == timed && recording.summary().serialize() == timed,
               "repeated file replay and text summaries never recount completed operations") && ok;
    const std::string text = read_output(output);
    std::fclose(output);
    ok = check(contains(text, "prefill:") && contains(text, "decode:") &&
               contains(text, "CPU cycles=1000") && contains(text, "worker CPU=") &&
               contains(text, "CPU work wall=") && contains(text, "NPU im2p/dense/work:") &&
               contains(text, "TTFT=") && contains(text, "TPOT="),
               "the real FILE output contains both phases, CPU resources, NPU domains and token latencies") && ok;
#else
    ok = check(recorded.empty() && !summary.available,
               "compiled-off inference logging leaves no raw records or replayable summary") && ok;
#endif
    return ok;
}

bool check_nested_cpu_integrity() {
    using namespace ggml::gemmini::performance;
    bool ok = true;
    for (int scenario = 0; scenario < 4; ++scenario) {
        Recording recording;
        if (!check(recording.output != nullptr, "create nested CPU integrity recording")) return false;
        start_request(100);
        begin_operation(Phase::prefill, 100);
        if (scenario == 3) record_test_cpu(100, 100, 7, 0, 0);
        else {
            record_test_cpu(100, 200, 7, 60, 600);
            record_test_cpu(120, 150, 7, 20, 200,
                scenario == 0 ? "nested_counter_failure" : nullptr,
                scenario == 1 ? 2 : 1, scenario == 2 ? 17 : 7);
        }
        end_operation(200, true);
        finish_request(220);
        finish_recording();
        const auto summary = recording.summary();
#if EXPECT_LOG_CYCLE
        const auto json = phase_json(summary.serialize(), "prefill");
        ok = check(summary.available, "nested sample validation keeps a complete recording readable") && ok;
        if (scenario == 3) {
            ok = check(contains(json, "\"cpu_cycles\":0,") && contains(json, "\"thread_cpu_ns\":0,"),
                       "a valid zero CPU resource sample stays zero rather than unavailable") && ok;
        } else if (scenario == 0) {
            ok = check(contains(json, "\"cpu_cycles\":null,\"cpu_cycles_reason\":\"nested_counter_failure\"") &&
                       contains(json, "\"thread_cpu_ns\":null,\"thread_cpu_ns_reason\":\"nested_counter_failure\""),
                       "a valid enclosing sample cannot erase an invalid nested sample") && ok;
        } else {
            ok = check(contains(json, "\"cpu_cycles\":null,\"cpu_cycles_reason\":\"nested_cpu_counter_domain_mismatch\"") &&
                       contains(json, "\"thread_cpu_ns\":60,"),
                       "native counters cannot deduplicate across owners or generations of one thread") && ok;
        }
#else
        ok = check(!summary.available && read_output(recording.output).empty(),
                   "compiled-off nested CPU fixtures emit no measurements") && ok;
#endif
    }
    return ok;
}

bool check_inference_summary_incomplete() {
    using namespace ggml::gemmini::performance;
    Recording recording;
    if (!check(recording.output != nullptr, "create incomplete-metric recording")) return false;
    start_request(100);
    begin_operation(Phase::prefill, 100);
    record_test_cpu(100, 180, 7, 40, 400);
    record_test_cpu(150, 200, 7, 30, 300);
    record_cpu_wall(100, 160);
    incomplete_cpu_wall("submission_wait_not_located");
    record_npu("fpga", "dense", "work", 10, true, nullptr);
    record_npu("fpga", "dense", "work", 0, false, "counter_read_failed");
    end_operation(200, true);
    token_ready(210);
    finish_request(220);
    start_request(1000);
    begin_operation(Phase::decode, 1000);
#if EXPECT_LOG_CYCLE
    bool ok = check(log_context() == "{\"request_id\":2,\"operation_id\":2,\"phase\":\"decode\",\"included\":true}",
                    "operation IDs remain monotonic across requests");
#else
    bool ok = true;
#endif
    end_operation(1010, true);
    token_ready(1050);
    token_ready(1090);
    finish_request(1100);
    finish_recording();
    const auto summary = recording.summary();
#if EXPECT_LOG_CYCLE
    const std::string json = summary.serialize();
    ok = check(summary.available, "incomplete metrics do not invalidate a complete recording") &&
        check(contains(json, "\"cpu_cycles\":null,\"cpu_cycles_reason\":\"partially_overlapping_cpu_spans\"") &&
              contains(json, "\"thread_cpu_ns\":null"),
              "partially overlapping same-thread spans do not produce a double-counted total") &&
        check(contains(json, "\"cpu_work_wall_ns\":null,\"cpu_work_wall_ns_reason\":\"submission_wait_not_located\"") &&
              contains(json, "\"cpu_work_wall_measured_ns\":60,"),
              "incomplete CPU work coverage retains measured stages without claiming complete wall time") &&
        check(contains(json, "\"cycles\":null,\"cycles_reason\":\"counter_read_failed\""),
              "invalid NPU counters stay unavailable in a replayed recording") && ok;
    const auto repeated = summary.serialize();
    ok = check(contains(repeated, "\"requests\":2,\"completed_requests\":2,\"tokens\":3,") &&
               contains(repeated, "\"ttft_ns\":80,") && contains(repeated, "\"tpot_ns\":40,"),
               "interactive requests average TTFT and exclude idle inter-request gaps from TPOT") && ok;
#else
    ok = check(read_output(recording.output).empty() && !summary.available,
               "compiled-off logging cannot replay incomplete metrics") && ok;
#endif
    return ok;
}

// Feed counters through the public sink without loading an NPU backend or
// simulator. These are test inputs, not measured device performance.
bool check_supplied_npu_final_output() {
    using namespace ggml::gemmini::performance;
    Recording recording;
    if (!check(recording.output != nullptr, "create supplied-counter recording")) return false;
    constexpr uint64_t supplied_cycles = UINT64_C(9007199254740993);
    start_request(100);
    begin_operation(Phase::prefill, 100);
    record_test_cpu(100, 180, 7, 40, 400);
    record_cpu_wall(100, 180);
    record_npu("supplied_test_counter", "dense", "work_total_cycles", supplied_cycles, true, nullptr);
    record_npu("supplied_test_counter", "residual", "work_total_cycles", 0, true, nullptr);
    record_npu("supplied_test_counter", "unavailable", "work_total_cycles", 0, false,
               "provider_counter_unavailable");
    end_operation(180, true);
    token_ready(190);
    finish_request(200);
    finish_recording();
    const auto summary = recording.summary();
#if EXPECT_LOG_CYCLE
    const auto json = summary.serialize();
    const auto entry = [&](const char *domain) {
        const std::string needle = std::string("{\"backend\":\"supplied_test_counter\",\"domain\":\"") + domain + "\"";
        const auto begin = json.find(needle);
        if (begin == std::string::npos) return std::string{};
        const auto end = json.find('}', begin);
        return json.substr(begin, end - begin + 1);
    };
    bool ok = check(summary.available && contains(phase_json(json, "prefill"), "\"cpu_cycles\":400,"),
                    "received NPU counters do not alter CPU cycle totals") &&
        check(contains(entry("dense"), "\"cycles\":9007199254740993,\"cycles_reason\":null"),
              "supplied uint64 cycle values retain precision above 2^53") &&
        check(contains(entry("residual"), "\"cycles\":0,\"cycles_reason\":null"),
              "a supplied valid zero remains a valid zero in final output") &&
        check(contains(entry("unavailable"), "\"cycles\":null,\"cycles_reason\":\"provider_counter_unavailable\""),
              "missing device counters retain their reason rather than becoming zero") &&
        check(contains(entry("dense"), "\"time_ns\":null,\"time_reason\":\"npu_frequency_unavailable\""),
              "the receiver never invents elapsed time from an unknown device frequency");
    FILE *output = std::tmpfile();
    if (!check(output != nullptr, "create supplied-counter final output")) return false;
    summary.print(output);
    const auto text = read_output(output);
    std::fclose(output);
    ok = check(contains(text, "CPU cycles=400") &&
               contains(text, "NPU supplied_test_counter/dense/work_total_cycles:") &&
               contains(text, "9007199254740993") &&
               contains(text, "NPU supplied_test_counter/residual/work_total_cycles:") &&
               contains(text, "provider_counter_unavailable"),
               "the final FILE writer includes supplied Dense/Residual counters and failure reasons") && ok;
    return ok;
#else
    return check(read_output(recording.output).empty() && !summary.available,
                 "disabled logging emits no supplied NPU counters");
#endif
}

bool check_inference_summary_overflow() {
    using namespace ggml::gemmini::performance;
    Recording recording;
    if (!check(recording.output != nullptr, "create overflow recording")) return false;
    start_request(10);
    begin_operation(Phase::decode, 10);
    record_test_cpu(10, 20, 7, 1, std::numeric_limits<uint64_t>::max());
    record_test_cpu(20, 30, 7, 1, 1);
    record_cpu_wall(10, 30);
    end_operation(30, true);
    token_ready(40);
    token_ready(39);
    finish_request(50);
    finish_recording();
    const auto summary = recording.summary();
#if EXPECT_LOG_CYCLE
    const auto invalid = summary.serialize();
    return check(summary.available &&
                 contains(invalid, "\"cpu_cycles\":null,\"cpu_cycles_reason\":\"aggregate_overflow\"") &&
               contains(invalid, "\"thread_cpu_ns\":2,") &&
               contains(invalid, "\"tpot_ns\":null,\"tpot_ns_reason\":\"clock_regression\""),
                 "counter overflow and token clock regression invalidate only their affected metrics");
#else
    return check(read_output(recording.output).empty() && !summary.available,
                 "compiled-off logging cannot replay overflow metrics");
#endif
}

bool check_long_recording() {
    using namespace ggml::gemmini::performance;
    Recording recording;
    if (!check(recording.output != nullptr, "create long recording")) return false;
    constexpr uint64_t large = (uint64_t{1} << 53) + 1;
    constexpr uint64_t operations = 2048;
    start_request(1);
    for (uint64_t i = 0; i < operations; ++i) {
        const uint64_t start = i * 10 + 1;
        const uint64_t cycles = i == 0 ? large : 1;
        begin_operation(Phase::decode, start);
        record_test_cpu(start, start + 5, 7, 3, cycles);
        record_cpu_wall(start, start + 5);
        record_npu("fpga", "dense", "work", cycles, true, nullptr);
        end_operation(start + 5, true);
        token_ready(start + 6);
    }
    finish_request(operations * 10 + 1);
    finish_recording();
    const auto summary = recording.summary();
#if EXPECT_LOG_CYCLE
    const std::string json = summary.serialize();
    const std::string expected = std::to_string(large + operations - 1);
    return check(summary.available && summary.peak_operation_samples == 3,
                 "long recordings retain only one operation's three resource samples") &&
        check(contains(json, ("\"cpu_cycles\":" + expected + ',').c_str()) &&
              contains(json, ("\"cycles\":" + expected + ',').c_str()) &&
              contains(json, "\"operations\":2048,") && contains(json, "\"tokens\":2048,"),
              "replay preserves exact CPU and NPU integers above 2^53 across thousands of operations");
#else
    return check(read_output(recording.output).empty() && !summary.available,
                 "compiled-off long recordings emit no records or available summary");
#endif
}

bool check_recording_integrity() {
    using namespace ggml::gemmini::performance;
    Recording recording;
    if (!check(recording.output != nullptr, "create recording integrity fixture")) return false;
    start_request(100);
    begin_operation(Phase::prefill, 100);
    record_test_cpu(100, 110, 7, 5, 50);
    record_cpu_wall(100, 110);
    record_npu("fpga", "dense", "work", 100, true, nullptr);
    end_operation(110, true);
    token_ready(120, 0);
    finish_request(130);
    finish_recording();
    const auto contents = read_output(recording.output);
    const auto summary = replay(contents);
#if EXPECT_LOG_CYCLE
    bool ok = check(summary.available && contains(summary.serialize(), "\"ttft_ns\":20,") &&
                    contains(summary.serialize(), "\"tpot_ns\":null,"),
                    "one-token recording supplies TTFT without manufacturing TPOT");
    ok = check(contains(contents, "\"token_id\":0") && contains(contents, "\"token_index\":0") &&
               contains(contents, "\"boundary\":\"tokenized_prompt_before_inference\"") &&
               contains(contents, "\"boundary\":\"sampling_and_accept_complete\"") &&
               contains(summary.serialize(), "\"latency_boundary_status\":\"verified\"") &&
               contains(summary.serialize(), "\"ttft_aggregation\":\"request_mean\"") &&
               contains(summary.serialize(), "\"tpot_aggregation\":\"token_gap_weighted_mean\""),
               "request events preserve token zero and declare current measured latency boundaries") && ok;
    std::string legacy_boundary = contents;
    const std::string start_boundary = "\"boundary\":\"tokenized_prompt_before_inference\",";
    legacy_boundary.erase(legacy_boundary.find(start_boundary), start_boundary.size());
    const auto legacy_latency = replay(legacy_boundary);
    ok = check(legacy_latency.available && contains(legacy_latency.serialize(), "\"ttft_ns\":20,") &&
               contains(legacy_latency.serialize(), "\"latency_boundary_status\":\"unverified_legacy_log\"") &&
               contains(legacy_latency.serialize(), "\"request_start_boundary\":null"),
               "legacy timestamps keep their calculation without inventing a verified request boundary") && ok;
    const auto reject = [&](const std::string & text, const char * message) {
        const auto invalid = replay(text);
        return check(!invalid.available && !invalid.reason.empty(), message);
    };
    const auto replace = [&](const std::string & from, const std::string & to, const char * message) {
        std::string changed = contents;
        const auto offset = changed.find(from);
        if (!check(offset != std::string::npos, "integrity mutation locates its original field")) return false;
        changed.replace(offset, from.size(), to);
        return reject(changed, message);
    };
    const auto remove_record = [&](const char * marker, const char * message) {
        std::string changed = contents;
        const auto offset = changed.find(marker);
        if (!check(offset != std::string::npos, "integrity mutation locates its original record")) return false;
        const auto preceding = changed.rfind('\n', offset);
        const auto begin = preceding == std::string::npos ? 0 : preceding + 1;
        const auto end = changed.find('\n', offset);
        changed.erase(begin, end == std::string::npos ? end : end - begin + 1);
        return reject(changed, message);
    };
    ok = reject("", "empty input cannot supply a final summary") && ok;
    ok = reject(contents + "{broken}\n", "malformed JSON cannot be ignored after session completion") && ok;
    ok = reject(contents + "{\"record_type\":\"LOG_ERROR\"}\n",
                "explicit logging errors invalidate a completed recording") && ok;
    ok = reject(contents.substr(0, contents.size() - 2),
                "truncated final JSON record is rejected") && ok;
    ok = replace("\"version\":2", "\"version\":2,\"version\":2",
                 "duplicate JSON object keys are rejected") && ok;
    ok = replace("\"version\":2", "\"version\":2.0",
                 "canonical integer fields reject floating-point JSON values") && ok;
    ok = replace("\"request_id\":1", "\"request_id\":2",
                 "mixed request identities are rejected") && ok;
    ok = replace("\"operation_id\":1", "\"operation_id\":2",
                 "mixed operation identities are rejected") && ok;
    ok = replace("\"token_step\":0", "\"token_step\":1",
                 "operation token step must match the request target ordinal") && ok;
    ok = replace("\"token_index\":0", "\"token_index\":1",
                 "the first generated token cannot acquire a missing prior ordinal") && ok;
    ok = replace("\"token_id\":0", "\"token_id\":0.5",
                 "token identity rejects fractional JSON values") && ok;
    ok = replace("\"boundary\":\"sampling_and_accept_complete\"", "\"boundary\":\"stdout_chunk\"",
                 "a different token boundary is not silently reported as the current metric") && ok;
    ok = replace("\"clock\":\"steady_clock\"", "\"clock\":\"system_clock\"",
                 "request latencies cannot combine another clock domain") && ok;
    ok = replace("\"execution_id\":\"", "\"execution_id\":\"other-",
                 "mixed execution identities are rejected") && ok;
    ok = replace("\"resource_samples\":3", "\"resource_samples\":4",
                 "operation completion must account for every canonical resource sample") && ok;
    ok = replace("\"sequence\":1", "\"sequence\":2",
                 "duplicate canonical resource sequence numbers are rejected") && ok;
    ok = replace("\"log_healthy\":true", "\"log_healthy\":false",
                 "recording loss cannot produce an available summary") && ok;
    ok = replace("\"cycles\":50", "\"cycles\":50.5",
                 "cycle counts reject floating-point values") && ok;
    ok = replace("\"cycles\":50", "\"cycles\":-1",
                 "cycle counts reject negative values") && ok;
    ok = replace("\"cycles\":50", "\"cycles\":51",
                 "valid cycle totals must agree with original sampled endpoints") && ok;
    ok = remove_record("\"record_type\":\"RESOURCE_SAMPLE\"",
                       "missing canonical resource records are rejected") && ok;
    ok = remove_record("\"event\":\"session_start\"", "missing recording start is rejected") && ok;
    ok = remove_record("\"event\":\"session_end\"", "missing recording completion is rejected") && ok;
    ok = remove_record("\"event\":\"request_start\"", "missing request start is rejected") && ok;
    ok = remove_record("\"event\":\"request_end\"", "missing request completion is rejected") && ok;
    ok = remove_record("\"event\":\"operation_start\"", "missing operation start is rejected") && ok;
    ok = remove_record("\"event\":\"operation_end\"", "missing operation completion is rejected") && ok;
    ok = remove_record("\"event\":\"token_ready\"", "missing token lifecycle records are rejected") && ok;
    ok = reject(contents + contents, "multiple recording sessions cannot be silently combined") && ok;
    return ok;
#else
    return check(contents.empty() && !summary.available,
                 "compiled-off integrity fixture has no replayable records");
#endif
}

bool check_operation_clock_order() {
    using namespace ggml::gemmini::performance;
    bool ok = true;
    for (int scenario = 0; scenario < 3; ++scenario) {
        Recording recording;
        if (!check(recording.output != nullptr, "create operation clock-order recording")) return false;
        start_request(100);
        begin_operation(Phase::prefill, 100);
        end_operation(110, true);
        if (scenario == 0) token_ready(105);
        if (scenario == 2) {
            begin_operation(Phase::decode, 105);
            end_operation(120, true);
        }
        finish_request(scenario == 1 ? 105 : 130);
        finish_recording();
        const auto summary = recording.summary();
#if EXPECT_LOG_CYCLE
        if (scenario == 2) {
            ok = check(!summary.available && summary.reason == "invalid_operation_order",
                       "an operation cannot begin before the previous operation completed") && ok;
        } else {
            const auto json = summary.serialize();
            ok = check(summary.available &&
                       contains(json, "\"ttft_ns\":null,\"ttft_ns_reason\":\"clock_regression\"") &&
                       contains(json, "\"tpot_ns\":null,\"tpot_ns_reason\":\"clock_regression\"") &&
                       contains(json, scenario == 0 ? "\"request_elapsed_ns\":30," :
                           "\"request_elapsed_ns\":null,\"request_elapsed_ns_reason\":\"clock_regression\""),
                       "token and request endpoints cannot precede the last completed operation") && ok;
        }
#else
        ok = check(read_output(recording.output).empty() && !summary.available,
                   "compiled-off clock-order recording has no replayable records") && ok;
#endif
    }
    return ok;
}
}

int main() {
    using namespace ggml::gemmini::cycle;
    const uint64_t caller_tid = host_thread_id();
    uint64_t worker_tid = 0;
    std::string worker_json;
    std::thread worker([&] {
        worker_tid = host_thread_id();
        worker_json = serialize_host_timing(100, 160, worker_tid, worker_tid);
    });
    worker.join();

    const std::string delayed_json = serialize_host_timing(100, 160, worker_tid, worker_tid);
    const std::string cross_task = serialize_host_timing(120, 180, worker_tid, caller_tid);
    const std::string regression = serialize_host_timing(180, 120, worker_tid, worker_tid);
    const std::string missing = serialize_host_timing(0, 0, 0, 0);
    const std::string zero = serialize_host_timing(0, 0, worker_tid, worker_tid);
    bool ok =
        check(caller_tid != 0 && worker_tid != 0 && caller_tid != worker_tid,
              "concurrent threads have distinct identities") &&
        check(worker_json == delayed_json,
              "delayed serialization preserves the executing worker and execution identity") &&
        check(contains(delayed_json, ("\"execution_id\":\"" + host_execution_id() + '"').c_str()),
              "standalone execution identity matches serialized timing on every worker") &&
        check(delayed_json.find("\"start_tid\":" + std::to_string(worker_tid)) != std::string::npos &&
              delayed_json.find("\"duration_ns\":60") != std::string::npos,
              "host interval preserves captured identity and elapsed nanoseconds") &&
        check(cross_task.find("\"valid\":true") != std::string::npos,
              "cross-task host timing remains valid independently of CPU counter ownership") &&
        check(regression.find("\"valid\":false") != std::string::npos &&
              regression.find("\"duration_ns\":null") != std::string::npos,
              "regressing host interval never underflows") &&
        check(missing.find("\"valid\":false") != std::string::npos &&
              missing.find("\"start_ns\":null") != std::string::npos,
              "uncollected timing remains unavailable") &&
        check(zero.find("\"valid\":true") != std::string::npos &&
              zero.find("\"duration_ns\":0") != std::string::npos,
              "collected zero duration is valid");
    ok = check_thread_cpu_timing() && ok;
    ok = check_worker_cpu_totals() && ok;
    ok = check_cpu_interval_record() && ok;
    ok = check_cycle_write_timing() && ok;
    ok = check_cpu_resource_summary_from_timing_add() && ok;
    ok = check_inference_summary() && ok;
    ok = check_nested_cpu_integrity() && ok;
    ok = check_inference_summary_incomplete() && ok;
    ok = check_supplied_npu_final_output() && ok;
    ok = check_inference_summary_overflow() && ok;
    ok = check_long_recording() && ok;
    ok = check_recording_integrity() && ok;
    ok = check_operation_clock_order() && ok;
    if (ok) std::printf("%s\n%s\n", delayed_json.c_str(), cross_task.c_str());
    return ok ? 0 : 1;
}
