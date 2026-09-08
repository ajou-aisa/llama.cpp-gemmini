#include <gemmini/host-timing.hpp>

#include <cstdio>
#include <string>
#include <thread>

namespace {
bool check(bool condition, const char * message) {
    if (!condition) std::fprintf(stderr, "FAIL: %s\n", message);
    return condition;
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
    const bool ok =
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
    if (ok) std::printf("%s\n%s\n", delayed_json.c_str(), cross_task.c_str());
    return ok ? 0 : 1;
}
