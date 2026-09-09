#include <gemmini/host-timing.hpp>
#include <gemmini/log.hpp>

#include <cstdio>
#include <ctime>
#include <string>
#include <thread>

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
    ok = check_cycle_write_timing() && ok;
    if (ok) std::printf("%s\n%s\n", delayed_json.c_str(), cross_task.c_str());
    return ok ? 0 : 1;
}
