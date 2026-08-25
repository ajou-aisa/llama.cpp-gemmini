#include "gemmini/layer.hpp"
#include "gemmini/log.hpp"
#include "../include/llama.h"
#include "llama-impl.h"

#include <atomic>
#include <condition_variable>
#include <chrono>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <mutex>
#include <optional>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

static_assert(noexcept(gemmini_log_file(nullptr)));
static_assert(noexcept(gemmini_log_truncate_file(nullptr)));
static_assert(noexcept(gemmini_log_debug_set_output_path(nullptr)));
static_assert(noexcept(gemmini_log_cycle_set_output_path(nullptr)));
static_assert(noexcept(gemmini_log_debug_set_output(nullptr)));
static_assert(noexcept(gemmini_log_cycle_set_output(nullptr)));
static_assert(noexcept(gemmini_hardware_counter_lease_acquire()));
static_assert(noexcept(gemmini_hardware_counter_lease_release()));
static_assert(noexcept(gemmini_log_debug("%d", 1)));
static_assert(noexcept(gemmini_log_debug_layer(nullptr, "%d", 1)));
static_assert(noexcept(gemmini_log_debug_loc(nullptr, 0, nullptr, "%d", 1)));
static_assert(noexcept(gemmini_log_debug_to({nullptr}, "%d", 1)));
static_assert(noexcept(gemmini_log_debug_to_layer({nullptr}, nullptr, "%d", 1)));
static_assert(noexcept(gemmini_log_debug_to_loc({nullptr}, nullptr, 0, nullptr, "%d", 1)));
static_assert(noexcept(gemmini_log_ws_cycle(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)));
static_assert(noexcept(gemmini_log_cycle_record(nullptr)));
static_assert(noexcept(gemmini_log_cycle_record_v2(nullptr)));
static_assert(noexcept(gemmini_log_cycle(nullptr, nullptr, 0, 0)));
static_assert(noexcept(gemmini_log_cycle_loc(nullptr, 0, nullptr, nullptr, nullptr, 0, 0)));
static_assert(noexcept(gemmini_log_cycle_to({nullptr}, nullptr, nullptr, 0, 0)));
static_assert(noexcept(gemmini_log_cycle_to_loc({nullptr}, nullptr, 0, nullptr, nullptr, nullptr, 0, 0)));

static std::string read_file(const std::filesystem::path & path) {
    std::ifstream input(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

static void write_file(const std::filesystem::path & path, const char * content) {
    std::ofstream(path, std::ios::binary) << content;
}

static bool parse_json_string(const std::string & value, size_t & offset) {
    if (offset == value.size() || value[offset++] != '"') return false;
    while (offset != value.size()) {
        const unsigned char c = static_cast<unsigned char>(value[offset++]);
        if (c == '"') return true;
        if (c < 0x20) return false;
        if (c != '\\') continue;
        if (offset == value.size()) return false;
        const char escape = value[offset++];
        if (std::string("\"\\/bfnrt").find(escape) != std::string::npos) continue;
        if (escape != 'u' || value.size() - offset < 4) return false;
        for (int i = 0; i != 4; ++i) {
            if (!std::isxdigit(static_cast<unsigned char>(value[offset++]))) return false;
        }
    }
    return false;
}

static bool parse_json_object_line(const std::string & value) {
    size_t offset = 0;
    if (value.empty() || value[offset++] != '{') return false;
    if (offset != value.size() && value[offset] == '}') return ++offset == value.size();
    for (;;) {
        if (!parse_json_string(value, offset) || offset == value.size() || value[offset++] != ':') return false;
        if (offset == value.size()) return false;
        if (value[offset] == '"') {
            if (!parse_json_string(value, offset)) return false;
        } else if (value.compare(offset, 4, "true") == 0 || value.compare(offset, 4, "null") == 0) {
            offset += 4;
        } else if (value.compare(offset, 5, "false") == 0) {
            offset += 5;
        } else {
            const size_t start = offset;
            if (value[offset] == '-') ++offset;
            while (offset != value.size() && std::isdigit(static_cast<unsigned char>(value[offset]))) ++offset;
            if (offset == start || (value[start] == '-' && offset == start + 1)) return false;
        }
        if (offset == value.size()) return false;
        if (value[offset] == '}') return ++offset == value.size();
        if (value[offset++] != ',') return false;
    }
}

static bool every_line_is_json(const std::filesystem::path & path, size_t * count = nullptr) {
    std::ifstream input(path);
    std::string line;
    size_t lines = 0;
    while (std::getline(input, line)) {
        ++lines;
        if (!parse_json_object_line(line)) {
            std::cerr << "invalid JSON line in " << path << ": " << line << '\n';
            return false;
        }
    }
    if (count) *count = lines;
    return input.eof();
}

class StartGate {
public:
    explicit StartGate(size_t participants) : participants_(participants) {}
    void arrive_and_wait() {
        std::unique_lock<std::mutex> lock(mutex_);
        if (++arrived_ == participants_) {
            released_ = true;
            ready_.notify_all();
            return;
        }
        ready_.wait(lock, [&] { return released_; });
    }
private:
    const size_t participants_;
    size_t arrived_ = 0;
    bool released_ = false;
    std::mutex mutex_;
    std::condition_variable ready_;
};

struct CallbackContext {
    const uint64_t magic;
    std::atomic<size_t> calls{0};
    std::atomic<size_t> mismatches{0};
};

static CallbackContext callback_a_context{0xa11ca11bu};
static CallbackContext callback_b_context{0xb22cb22cu};

static void callback_a(ggml_log_level, const char *, void * user_data) {
    auto * context = static_cast<CallbackContext *>(user_data);
    if (context != &callback_a_context || context->magic != 0xa11ca11bu) {
        callback_a_context.mismatches.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    context->calls.fetch_add(1, std::memory_order_relaxed);
}

static void callback_b(ggml_log_level, const char *, void * user_data) {
    auto * context = static_cast<CallbackContext *>(user_data);
    if (context != &callback_b_context || context->magic != 0xb22cb22cu) {
        callback_b_context.mismatches.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    context->calls.fetch_add(1, std::memory_order_relaxed);
}

static std::atomic<size_t> reentrant_calls{0};

static void reentrant_callback(ggml_log_level, const char *, void *) {
    reentrant_calls.fetch_add(1, std::memory_order_relaxed);
    llama_log_set(callback_b, &callback_b_context);
    llama_log_internal(GGML_LOG_LEVEL_INFO, "nested callback emission");
}

static bool test_llama_callback_replacement() {
    callback_a_context.calls = 0;
    callback_a_context.mismatches = 0;
    callback_b_context.calls = 0;
    callback_b_context.mismatches = 0;

    constexpr size_t iterations = 200000;
    llama_log_set(callback_a, &callback_a_context);
    StartGate gate(3);
    std::thread setter([&] {
        gate.arrive_and_wait();
        for (size_t i = 0; i != iterations; ++i) {
            if ((i & 1) == 0) {
                llama_log_set(callback_a, &callback_a_context);
            } else {
                llama_log_set(callback_b, &callback_b_context);
            }
        }
    });
    auto emit = [&] {
        gate.arrive_and_wait();
        for (size_t i = 0; i != iterations; ++i) {
            llama_log_internal(GGML_LOG_LEVEL_INFO, "callback replacement %zu", i);
        }
    };
    std::thread first_emitter(emit);
    std::thread second_emitter(emit);
    setter.join();
    first_emitter.join();
    second_emitter.join();

    const size_t calls = callback_a_context.calls.load() + callback_b_context.calls.load();
    const size_t mismatches = callback_a_context.mismatches.load() + callback_b_context.mismatches.load();
    if (calls + mismatches != 2 * iterations || mismatches != 0) {
        std::cerr << "llama callback pointer/user-data replacement was not atomic: "
                  << calls << " calls, " << mismatches << " mismatches\n";
        return false;
    }

    reentrant_calls = 0;
    const size_t nested_before = callback_b_context.calls.load();
    llama_log_set(reentrant_callback, nullptr);
    llama_log_internal(GGML_LOG_LEVEL_INFO, "outer callback emission");
    llama_log_set(nullptr, nullptr);
    if (reentrant_calls.load() != 1 || callback_b_context.calls.load() != nested_before + 1) {
        std::cerr << "reentrant llama callback replacement/emission failed\n";
        return false;
    }
    return true;
}

class ScopedCurrentPath {
public:
    ScopedCurrentPath() : path_(std::filesystem::current_path()) {}
    ~ScopedCurrentPath() { std::filesystem::current_path(path_); }

private:
    std::filesystem::path path_;
};

class ScopedEnv {
public:
    explicit ScopedEnv(const char * name) : name_(name) {
        if (const char * value = std::getenv(name)) {
            value_ = value;
        }
    }

    ~ScopedEnv() {
        set(value_ ? value_->c_str() : nullptr);
    }

    void set(const char * value) const {
#if defined(_WIN32)
        _putenv_s(name_, value ? value : "");
#else
        if (value) {
            setenv(name_, value, 1);
        } else {
            unsetenv(name_);
        }
#endif
    }

private:
    const char * name_;
    std::optional<std::string> value_;
};

static bool expect_path(const char * name, const std::filesystem::path & actual,
                        const std::filesystem::path & expected) {
    if (actual == expected) {
        return true;
    }
    std::cerr << name << ": expected " << expected << ", got " << actual << '\n';
    return false;
}

static bool environment_is_restored(const std::filesystem::path & expected_cwd,
                                    const std::optional<std::string> & expected_env) {
    const char * actual_env = std::getenv("GEMMINI_LOG_DIR");
    return std::filesystem::current_path() == expected_cwd &&
           ((actual_env == nullptr && !expected_env) ||
            (actual_env != nullptr && expected_env && actual_env == *expected_env));
}

static std::optional<std::filesystem::path> create_test_root() {
    std::error_code error;
    const std::filesystem::path base = std::filesystem::canonical(
        std::filesystem::temp_directory_path(), error) / "gemmini-log-regression";
    if (error) {
        std::cerr << "could not resolve temporary directory: " << error.message() << '\n';
        return std::nullopt;
    }
    for (unsigned int index = 0; index != 1024; ++index) {
        std::filesystem::path candidate = base;
        candidate += "-" + std::to_string(index);
        error.clear();
        if (std::filesystem::create_directory(candidate, error)) {
            return candidate;
        }
        if (error) {
            std::cerr << "could not create test directory: " << error.message() << '\n';
            return std::nullopt;
        }
    }
    std::cerr << "could not create a unique test directory\n";
    return std::nullopt;
}

static bool test_default_setup(const std::filesystem::path & test_root) {
    ScopedCurrentPath cwd;
    ScopedEnv env("GEMMINI_LOG_DIR");
    std::filesystem::current_path(test_root);
    env.set(nullptr);

    constexpr size_t thread_count = 16;
    std::vector<ggml::gemmini::log::DefaultOutputSetupResult> results(thread_count);
    std::vector<std::thread> threads;
    threads.reserve(thread_count);
    for (size_t i = 0; i < thread_count; ++i) {
        threads.emplace_back([&, i] { results[i] = ggml::gemmini::log::setup_default_outputs(); });
    }
    for (std::thread & thread : threads) {
        thread.join();
    }
    for (const auto & result : results) {
        if (!result.debug || !result.cycle) {
            std::cerr << "concurrent default output setup failed\n";
            return false;
        }
    }

    const std::filesystem::path debug_path = test_root / "output/log/debug-log.jsonl";
    const std::filesystem::path cycle_path = test_root / "output/log/cycle-log.jsonl";
#if EXPECT_LOG_DEBUG
    if (!std::filesystem::exists(debug_path)) {
        std::cerr << "enabled default debug output was not created\n";
        return false;
    }
    std::ofstream(debug_path, std::ios::app) << "{\"sentinel\":\"debug\"}\n";
#endif
#if EXPECT_LOG_CYCLE
    if (!std::filesystem::exists(cycle_path)) {
        std::cerr << "enabled default cycle output was not created\n";
        return false;
    }
    std::ofstream(cycle_path, std::ios::app) << "{\"sentinel\":\"cycle\"}\n";
#endif

    threads.clear();
    for (size_t i = 0; i < thread_count; ++i) {
        threads.emplace_back([] { (void)ggml::gemmini::log::setup_default_outputs(); });
    }
    for (std::thread & thread : threads) {
        thread.join();
    }

#if EXPECT_LOG_DEBUG
    if (read_file(debug_path).find("{\"sentinel\":\"debug\"}\n") == std::string::npos ||
        !every_line_is_json(debug_path)) {
        std::cerr << "repeated setup truncated default debug output\n";
        return false;
    }
#else
    (void)ggml::gemmini::log::debug.set_output_path(GEMMINI_LOG_DEFAULT_DEBUG_PATH, true);
    (void)gemmini_log_debug_set_output_path(GEMMINI_LOG_DEFAULT_DEBUG_PATH);
    gemmini_log_ws_cycle(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
    if (std::filesystem::exists(debug_path) ||
        std::filesystem::exists(test_root / "output/log/log-ws-loop.jsonl")) {
        std::cerr << "disabled debug setup created a default output\n";
        return false;
    }
#endif
#if EXPECT_LOG_CYCLE
    const std::string repeated_cycle_output = read_file(cycle_path);
    if (repeated_cycle_output.find("{\"sentinel\":\"cycle\"}\n") == std::string::npos ||
        !every_line_is_json(cycle_path)) {
        std::cerr << "repeated setup truncated default cycle output: bytes="
                  << repeated_cycle_output.size() << " content=" << repeated_cycle_output << '\n';
        return false;
    }
#else
    (void)ggml::gemmini::log::cycle.set_output_path(GEMMINI_LOG_DEFAULT_CYCLE_PATH, true);
    (void)gemmini_log_cycle_set_output_path(GEMMINI_LOG_DEFAULT_CYCLE_PATH);
    if (std::filesystem::exists(cycle_path)) {
        std::cerr << "disabled cycle setup created a default output\n";
        return false;
    }
#endif
    return true;
}

static bool test_explicit_debug_before_default_setup() {
    const std::optional<std::filesystem::path> created_root = create_test_root();
    if (!created_root) {
        return false;
    }

    const std::filesystem::path test_root = *created_root;
    const std::filesystem::path explicit_path = test_root / "explicit-debug.jsonl";
    const std::filesystem::path default_path = test_root / "output/log/debug-log.jsonl";
    bool failed = false;
    {
        ScopedCurrentPath cwd;
        ScopedEnv env("GEMMINI_LOG_DIR");
        std::filesystem::current_path(test_root);
        env.set(nullptr);

        if (!ggml::gemmini::log::debug.set_output_path(explicit_path.c_str())) {
            std::cerr << "could not configure explicit debug output\n";
            failed = true;
        } else {
            ggml::gemmini::log::debug("explicit-sentinel-before-default");
            for (int i = 0; i != 2; ++i) {
                const auto result = ggml::gemmini::log::setup_default_outputs();
                if (!result.debug || !result.cycle) {
                    std::cerr << "default output setup failed\n";
                    failed = true;
                }
            }
            ggml::gemmini::log::debug("explicit-sentinel-after-default");

            const std::string output = read_file(explicit_path);
            if (output.find("explicit-sentinel-before-default") == std::string::npos ||
                output.find("explicit-sentinel-after-default") == std::string::npos ||
                !every_line_is_json(explicit_path)) {
                std::cerr << "default setup replaced explicit debug output\n";
                failed = true;
            }
            if (std::filesystem::exists(default_path)) {
                std::cerr << "default debug output was created despite explicit output\n";
                failed = true;
            }
        }
    }

    ggml::gemmini::log::debug.set_output(stderr);
    ggml::gemmini::log::cycle.set_output(stderr);
    std::error_code cleanup_error;
    std::filesystem::remove_all(test_root, cleanup_error);
    if (cleanup_error) {
        std::cerr << "could not remove test directory: " << cleanup_error.message() << '\n';
        failed = true;
    }
    return !failed;
}

static bool test_hardware_counter_contract(const std::filesystem::path & root) {
#if !EXPECT_LOG_CYCLE
    (void)root;
    return true;
#else
    const std::filesystem::path output = root / "hardware-counter.jsonl";
    if (!ggml::gemmini::log::cycle.set_output_path(output.c_str(), true)) {
        std::cerr << "could not initialize hardware counter sink\n";
        return false;
    }

    gemmini_log_ws_cycle(UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX,
                         2, 3, 4, 1, 1, 1, 2, 3, 4, 0, 1);
    gemmini_log_ws_cycle(static_cast<uint64_t>(UINT32_MAX) + 1, 1, 2, 3, 4,
                         2, 3, 4, 1, 1, 1, 2, 3, 4, 0, 1);
    gemmini_log_ws_cycle(10, 11, 2, 3, 4,
                         2, 3, 4, 1, 1, 1, 2, 3, 4, 0, 1);
    ggml::gemmini::log::cycle.set_output(stderr);

    const std::string json = read_file(output);
    const auto count = [&json](const std::string & needle) {
        size_t found = 0;
        for (size_t at = 0; (at = json.find(needle, at)) != std::string::npos; at += needle.size()) ++found;
        return found;
    };
    if (!every_line_is_json(output) ||
        count("\"record_type\":\"WS_LOOP_TELEMETRY\"") != 3 ||
        count("\"containing_interval_counter_bits\":64") != 3 ||
        count("\"occupancy_counter_bits\":32") != 3 ||
        count("\"valid\":true") != 1 || count("\"valid\":false") != 2) {
        std::cerr << "hardware counter width/wrap contract failed: " << json << '\n';
        return false;
    }

    constexpr size_t thread_count = 12;
    constexpr size_t iterations = 2000;
    StartGate gate(thread_count);
    std::atomic<unsigned int> owners{0};
    std::atomic<size_t> overlaps{0};
    std::vector<std::thread> threads;
    threads.reserve(thread_count);
    for (size_t thread = 0; thread != thread_count; ++thread) {
        threads.emplace_back([&] {
            gate.arrive_and_wait();
            for (size_t i = 0; i != iterations; ++i) {
                ggml::gemmini::log::HardwareCounterLease lease;
                if (owners.fetch_add(1, std::memory_order_relaxed) != 0) {
                    overlaps.fetch_add(1, std::memory_order_relaxed);
                }
                if (owners.fetch_sub(1, std::memory_order_relaxed) != 1) {
                    overlaps.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });
    }
    for (std::thread & thread : threads) thread.join();
    if (overlaps.load(std::memory_order_relaxed) != 0 || owners.load(std::memory_order_relaxed) != 0) {
        std::cerr << "hardware counter leases overlapped ownership\n";
        return false;
    }
    return true;
#endif
}

static bool test_atomic_cycle_sink(const std::filesystem::path & root) {
#if !EXPECT_LOG_CYCLE
    (void)root;
    return true;
#else
    const std::filesystem::path first = root / "atomic-first.jsonl";
    const std::filesystem::path second = root / "atomic-second.jsonl";
    if (!ggml::gemmini::log::truncate_file(first.c_str()) ||
        !ggml::gemmini::log::truncate_file(second.c_str()) ||
        !ggml::gemmini::log::cycle.set_output_path(first.c_str())) {
        std::cerr << "could not initialize atomic cycle sink\n";
        return false;
    }

    gemmini_cycle_record c_record{};
    c_record.layer = "c-layer\n";
    c_record.op = "c-op\"";
    c_record.start = 1;
    c_record.end = 4;
    c_record.file = "c-file";
    c_record.line = 7;
    c_record.func = "c-func";
    gemmini_log_cycle_record(&c_record);
    ggml::gemmini::log::cycle.write({"cpp-layer", "cpp-op", 4, 9, "cpp-file", 8, "cpp-func"});
    ggml::gemmini::log::cycle.write_json(
        "{\"schema\":\"gemmini.cycle\",\"version\":2,\"record_type\":\"TEST_AGGREGATE\"}");
    gemmini_log_ws_cycle(100, 10, 20, 30, 40, 2, 3, 4, 1, 1, 1, 2, 3, 4, 0, 1);

    StartGate gate(2);
    std::atomic<bool> setup_ok{true};
    std::thread writer([&] {
        gate.arrive_and_wait();
        for (uint64_t i = 0; i != 200; ++i) {
            ggml::gemmini::log::cycle.write({"writer", "record", i, i + 1, nullptr, 0, nullptr});
        }
    });
    std::thread setter([&] {
        gate.arrive_and_wait();
        for (int i = 0; i != 80; ++i) {
            const std::filesystem::path & path = i % 2 == 0 ? second : first;
            if (!ggml::gemmini::log::cycle.set_output_path(path.c_str())) setup_ok = false;
        }
    });
    writer.join();
    setter.join();
    ggml::gemmini::log::cycle.set_output(stderr);

    size_t first_lines = 0;
    size_t second_lines = 0;
    if (!setup_ok || !every_line_is_json(first, &first_lines) ||
        !every_line_is_json(second, &second_lines) || first_lines + second_lines != 204) {
        std::cerr << "atomic writer/replacement lost or split a structured record: "
                  << first_lines << '+' << second_lines << " lines\n";
        return false;
    }
    const std::string combined = read_file(first) + read_file(second);
    if (combined.find("\"layer\":\"c-layer\\n\"") == std::string::npos ||
        combined.find("\"op\":\"c-op\\\"\"") == std::string::npos ||
        combined.find("\"run_id\":null") == std::string::npos ||
        combined.find("\"stripe_id\":null") == std::string::npos ||
        combined.find("\"delta\":3") == std::string::npos ||
        combined.find("\"layer\":\"cpp-layer\"") == std::string::npos ||
        combined.find("\"record_type\":\"TEST_AGGREGATE\"") == std::string::npos ||
        combined.find("\"record_type\":\"CYCLE_INTERVAL\"") == std::string::npos ||
        combined.find("\"record_type\":\"WS_LOOP_TELEMETRY\"") == std::string::npos) {
        std::cerr << "structured C/C++ cycle fields were not preserved\n";
        return false;
    }
    return true;
#endif
}

struct TargetLockProbe {
    std::mutex mutex;
    std::condition_variable changed;
    bool writer_entered = false;
    bool release_writer = false;
    bool replacement_attempting = false;
    bool replacement_completed = false;
};

static void target_lock_hook(ggml::gemmini::log::testing::TargetWriteKind, void * user_data) {
    auto & probe = *static_cast<TargetLockProbe *>(user_data);
    std::unique_lock<std::mutex> lock(probe.mutex);
    probe.writer_entered = true;
    probe.changed.notify_all();
    probe.changed.wait(lock, [&] { return probe.release_writer; });
}

static bool test_target_lock_blocks_replacement(const std::filesystem::path & root,
                                                ggml::gemmini::log::testing::TargetWriteKind kind) {
#if !EXPECT_LOG_DEBUG
    (void) root;
    (void) kind;
    return true;
#else
    using ggml::gemmini::log::testing::TargetWriteKind;
    const auto persistent = root / "lock-persistent.jsonl";
    const auto targeted = root / "lock-targeted.jsonl";
    const auto replacement = root / "lock-replacement.jsonl";
    if (!gemmini_log_debug_set_output_path(persistent.c_str())) return false;
    TargetLockProbe probe;
    ggml::gemmini::log::testing::set_target_lock_hook(target_lock_hook, &probe);
    std::thread writer([&] {
        if (kind == TargetWriteKind::plain) {
            gemmini_log_debug_to(gemmini_log_file(targeted.c_str()), "plain");
        } else if (kind == TargetWriteKind::layer) {
            gemmini_log_debug_to_layer(gemmini_log_file(targeted.c_str()), "layer", "layered");
        } else {
            gemmini_log_debug_to_loc(gemmini_log_file(targeted.c_str()), "file", 1, "func", "located");
        }
    });
    {
        std::unique_lock<std::mutex> lock(probe.mutex);
        probe.changed.wait(lock, [&] { return probe.writer_entered; });
    }
    std::thread replacer([&] {
        {
            std::lock_guard<std::mutex> lock(probe.mutex);
            probe.replacement_attempting = true;
            probe.changed.notify_all();
        }
        (void) gemmini_log_debug_set_output_path(replacement.c_str());
        {
            std::lock_guard<std::mutex> lock(probe.mutex);
            probe.replacement_completed = true;
            probe.changed.notify_all();
        }
    });
    bool replacement_bypassed_lock = false;
    {
        std::unique_lock<std::mutex> lock(probe.mutex);
        probe.changed.wait(lock, [&] { return probe.replacement_attempting; });
        replacement_bypassed_lock = probe.changed.wait_for(
            lock, std::chrono::seconds(1), [&] { return probe.replacement_completed; });
        probe.release_writer = true;
        probe.changed.notify_all();
    }
    writer.join();
    replacer.join();
    ggml::gemmini::log::testing::clear_target_lock_hook();
    gemmini_log_debug_set_output(stderr);
    return !replacement_bypassed_lock;
#endif
}

static bool test_atomic_debug_sink(const std::filesystem::path & root) {
#if !EXPECT_LOG_DEBUG
    (void) root;
    return true;
#else
    const std::filesystem::path first = root / "debug-atomic-first.jsonl";
    const std::filesystem::path second = root / "debug-atomic-second.jsonl";
    const std::filesystem::path targeted = root / "debug-targeted.jsonl";
    if (!ggml::gemmini::log::debug.set_output_path(first.c_str(), true)) {
        std::cerr << "could not initialize atomic debug sink\n";
        return false;
    }

    constexpr size_t writer_count = 4;
    constexpr size_t records_per_writer = 400;
    StartGate gate(writer_count + 1);
    std::atomic<bool> setup_ok{true};
    std::vector<std::thread> threads;
    for (size_t writer = 0; writer != writer_count; ++writer) {
        threads.emplace_back([&, writer] {
            gate.arrive_and_wait();
            for (size_t record = 0; record != records_per_writer; ++record) {
                gemmini_log_debug("writer=%zu record=%zu", writer, record);
                gemmini_log_debug_to(gemmini_log_file(targeted.c_str()),
                                     "target-writer=%zu record=%zu", writer, record);
            }
        });
    }
    std::thread setter([&] {
        gate.arrive_and_wait();
        for (size_t replacement = 0; replacement != 120; ++replacement) {
            const std::filesystem::path & path = replacement % 2 == 0 ? second : first;
            if (!ggml::gemmini::log::debug.set_output_path(path.c_str(), true)) setup_ok = false;
        }
    });
    for (std::thread & thread : threads) thread.join();
    setter.join();
    gemmini_log_debug("final-debug-record");
    ggml::gemmini::log::debug.set_output(stderr);

    size_t first_lines = 0;
    size_t second_lines = 0;
    size_t targeted_lines = 0;
    const bool valid = every_line_is_json(first, &first_lines) &&
        every_line_is_json(second, &second_lines) && every_line_is_json(targeted, &targeted_lines);
    if (!setup_ok || !valid || first_lines + second_lines == 0 ||
        targeted_lines != writer_count * records_per_writer) {
        std::cerr << "atomic debug replacement produced stale or interleaved output\n";
        return false;
    }
    return true;
#endif
}

static bool test_setup_failures(const std::filesystem::path & root) {
#if !EXPECT_LOG_CYCLE
    (void)root;
    return true;
#else
    const std::filesystem::path parent_file = root / "not-a-directory";
    write_file(parent_file, "file\n");
    const std::filesystem::path child = parent_file / "child.jsonl";
    bool ok = true;
    ggml::gemmini::log::cycle.set_output(stderr);
    if (ggml::gemmini::log::cycle.set_output_path(child.c_str())) {
        std::cerr << "invalid output parent setup succeeded\n";
        ok = false;
    }
    ggml::gemmini::log::cycle.set_output(stderr);
    if (ggml::gemmini::log::truncate_file(child.c_str())) {
        std::cerr << "invalid truncate parent setup succeeded\n";
        ok = false;
    }
    if (ggml::gemmini::log::cycle.set_output_path("../escaped-cycle.jsonl") ||
        std::filesystem::exists(root.parent_path() / "escaped-cycle.jsonl")) {
        std::cerr << "cycle setup accepted relative traversal\n";
        ok = false;
    }
    ggml::gemmini::log::cycle.set_output(stderr);
    return ok;
#endif
}

static bool run_fault_probe(const std::string & name) {
#if !EXPECT_LOG_CYCLE
    (void)name;
    return true;
#else
    const std::optional<std::filesystem::path> root = create_test_root();
    if (!root) return false;
    const std::filesystem::path first = *root / "fault-first.jsonl";
    const std::filesystem::path second = *root / "fault-second.jsonl";
    using ggml::gemmini::log::testing::LogFault;
    ggml::gemmini::log::cycle.set_output(stderr);
    bool ok = true;
    if (name == "open") {
        ggml::gemmini::log::testing::set_log_fault(LogFault::open);
        ok = !ggml::gemmini::log::cycle.set_output_path(first.c_str());
    } else {
        ok = ggml::gemmini::log::cycle.set_output_path(first.c_str(), true);
        if (name == "write") {
            ggml::gemmini::log::testing::set_log_fault(LogFault::write);
            gemmini_log_cycle("fault", "write", 1, 2);
            gemmini_log_cycle("caller", "continues", 2, 3);
        } else if (name == "flush") {
            ggml::gemmini::log::testing::set_log_fault(LogFault::flush);
            gemmini_log_cycle("fault", "flush", 1, 2);
            gemmini_log_cycle("caller", "continues", 2, 3);
        } else if (name == "replacement") {
            ggml::gemmini::log::testing::set_log_fault(LogFault::replacement);
            ok = ok && !ggml::gemmini::log::cycle.set_output_path(second.c_str());
            gemmini_log_cycle("old-sink", "preserved", 1, 2);
        } else if (name == "allocation") {
            ggml::gemmini::log::testing::set_log_fault(LogFault::allocation);
            {
                ggml::gemmini::log::HardwareCounterLease lease;
                gemmini_log_ws_cycle(100, 10, 20, 30, 40,
                                     2, 3, 4, 1, 1, 1, 2, 3, 4, 0, 1);
            }
            {
                ggml::gemmini::log::HardwareCounterLease lease;
                gemmini_log_ws_cycle(101, 10, 20, 30, 40,
                                     2, 3, 4, 1, 1, 1, 2, 3, 4, 0, 1);
            }
        } else {
            ok = false;
        }
    }
    ggml::gemmini::log::testing::clear_log_fault();
    ggml::gemmini::log::cycle.set_output(stderr);
    if (std::filesystem::exists(first) && !every_line_is_json(first)) ok = false;
    const std::string first_output = read_file(first);
    if ((name == "write" || name == "flush") && first_output.find("caller") != std::string::npos) {
        std::cerr << "disabled sink accepted a later caller record\n";
        ok = false;
    }
    if (name == "replacement" &&
        (first_output.find("old-sink") == std::string::npos || std::filesystem::exists(second))) {
        std::cerr << "failed replacement did not preserve the old sink\n";
        ok = false;
    }
    if (name == "allocation" &&
        (first_output.find("\"containing_interval_cycles\":100,") != std::string::npos ||
         first_output.find("\"containing_interval_cycles\":101,") == std::string::npos)) {
        std::cerr << "allocation failure escaped or poisoned subsequent lease/log use\n";
        ok = false;
    }
    if (name == "open" && std::filesystem::exists(first)) {
        std::cerr << "injected open failure created an output\n";
        ok = false;
    }
    std::error_code error;
    std::filesystem::remove_all(*root, error);
    return ok && !error;
#endif
}

static bool test_semantic_matmul_layer_resolver() {
    using ggml::gemmini::types::resolve_matmul_layer;
    struct Case {
        std::string_view model_arch;
        std::string_view weight_name;
        std::string_view input_name;
        std::string_view output_name;
        std::string_view expected;
    };
    const std::vector<Case> cases = {
        {"gpt2", "blk.0.attn_qkv.weight", "", "", "blk.0.attn.qkv_proj"},
        {"gpt2", "blk.11.attn_output.weight", "", "", "blk.11.attn.out_proj"},
        {"gpt2", "blk.42.ffn_up.weight", "", "", "blk.42.mlp.up_proj"},
        {"gpt2", "blk.123456.ffn_down.weight", "", "", "blk.123456.mlp.down_proj"},
        {"llama", "blk.15.attn_q.weight", "", "", "blk.15.attn.q_proj"},
        {"llama", "blk.0.attn_k.weight", "", "", "blk.0.attn.k_proj"},
        {"llama", "blk.11.attn_v.weight", "", "", "blk.11.attn.v_proj"},
        {"llama", "blk.12.attn_output.weight", "", "", "blk.12.attn.out_proj"},
        {"llama", "blk.13.ffn_up.weight", "", "", "blk.13.mlp.up_proj"},
        {"llama", "blk.14.ffn_gate.weight", "", "", "blk.14.mlp.gate_proj"},
        {"llama", "blk.15.ffn_down.weight", "", "", "blk.15.mlp.down_proj"},
        {"gpt2", "output.weight", "ignored/input", "not-a-consumer", "lm_head"},
        {"llama", "output.weight", "", "result_output-bad", "lm_head"},
        {"gpt2", "token_embd.weight", "", "result_output", "lm_head"},
        {"llama", "token_embd.weight", "", "result_output-0", "lm_head"},
        {"llama", "token_embd.weight", "", "result_output-123456789012345678901234567890", "lm_head"},

        {"llama", "token_embd.weight", "", "result_output--1", "unclassified.token_embd.weight"},
        {"llama", "token_embd.weight", "", "result_output-x", "unclassified.token_embd.weight"},
        {"llama", "token_embd.weight", "", "result_output-1-extra", "unclassified.token_embd.weight"},
        {"llama", "blk..attn_q.weight", "", "", "unclassified.blk..attn_q.weight"},
        {"llama", "blk.-1.attn_q.weight", "", "", "unclassified.blk.-1.attn_q.weight"},
        {"llama", "blk.+1.attn_q.weight", "", "", "unclassified.blk._1.attn_q.weight"},
        {"llama", "blk.one.attn_q.weight", "", "", "unclassified.blk.one.attn_q.weight"},
        {"llama", "blk.18446744073709551616.attn_q.weight", "", "", "unclassified.blk.18446744073709551616.attn_q.weight"},
        {"llama", "blk.1.attn_q.weight.extra", "", "", "unclassified.blk.1.attn_q.weight.extra"},
        {"gpt2", "blk.1.attn_q.weight", "", "", "unclassified.blk.1.attn_q.weight"},
        {"llama", "blk.1.attn_qkv.weight", "", "", "unclassified.blk.1.attn_qkv.weight"},
        {"unknown", "output.weight", "", "result_output", "unclassified.output.weight"},
        {"llama", "blk.1.attn_q.bias", "", "", "unclassified.blk.1.attn_q.bias"},
        {"llama", "blk.1.attn_q", "", "", "unclassified.blk.1.attn_q"},
        {"llama", "attn_q.weight", "", "", "unclassified.attn_q.weight"},
        {"llama", "blk.1.attn_q.weight.lora_a", "", "", "unclassified.blk.1.attn_q.weight.lora_a"},
        {"llama", "blk.1.attn_q.weight.lora_b", "", "", "unclassified.blk.1.attn_q.weight.lora_b"},
        {"llama", {}, {}, {}, "unclassified.unknown"},
        {"llama", "", "", "", "unclassified.unknown"},
        {"llama", "", "input-priority-sentinel", "output-priority/sentinel", "unclassified.output-priority_sentinel"},
        {"llama", "", "", "consumer/bad name", "unclassified.consumer_bad_name"},
        {"llama", "", "input\nname", "", "unclassified.input_name"},
        {"llama", "bad name/\x01", "ignored", "ignored", "unclassified.bad_name__"},
    };

    bool ok = true;
    for (const Case & test : cases) {
        std::string actual;
        try {
            actual = resolve_matmul_layer(
                test.model_arch, test.weight_name, test.input_name, test.output_name);
        } catch (...) {
            std::cerr << "semantic matmul resolver threw for weight " << test.weight_name << '\n';
            ok = false;
            continue;
        }
        if (actual != test.expected) {
            std::cerr << "semantic matmul resolver: expected " << test.expected
                      << ", got " << actual << '\n';
            ok = false;
        }
    }

    struct RoleCase {
        std::string_view weight_suffix;
        std::string_view expected_suffix;
    };
    const RoleCase gpt2_roles[] = {
        {".attn_qkv.weight", ".attn.qkv_proj"},
        {".attn_output.weight", ".attn.out_proj"},
        {".ffn_up.weight", ".mlp.up_proj"},
        {".ffn_down.weight", ".mlp.down_proj"},
    };
    const RoleCase llama_roles[] = {
        {".attn_q.weight", ".attn.q_proj"},
        {".attn_k.weight", ".attn.k_proj"},
        {".attn_v.weight", ".attn.v_proj"},
        {".attn_output.weight", ".attn.out_proj"},
        {".ffn_up.weight", ".mlp.up_proj"},
        {".ffn_gate.weight", ".mlp.gate_proj"},
        {".ffn_down.weight", ".mlp.down_proj"},
    };
    const auto check_blocks =
        [&](std::string_view arch, unsigned block_count, const auto & roles) {
            for (unsigned block = 0; block < block_count; ++block) {
                const std::string prefix = "blk." + std::to_string(block);
                for (const RoleCase & role : roles) {
                    const std::string actual = resolve_matmul_layer(
                        arch, prefix + std::string(role.weight_suffix), "", "");
                    const std::string expected =
                        prefix + std::string(role.expected_suffix);
                    if (actual != expected) {
                        std::cerr << "semantic matmul resolver block coverage mismatch: "
                                  << arch << ' ' << block << ' '
                                  << role.weight_suffix << " -> " << actual
                                  << " expected " << expected << '\n';
                        ok = false;
                    }
                }
            }
        };
    check_blocks("gpt2", 12, gpt2_roles);
    check_blocks("llama", 16, llama_roles);

    const std::string long_weight(120, 'x');
    const std::string long_expected = "unclassified." + std::string(96, 'x');
    const std::string long_actual = resolve_matmul_layer("llama", long_weight, "", "");
    if (long_actual != long_expected || long_actual.size() != 109) {
        std::cerr << "semantic matmul fallback was not truncated to 96 payload bytes\n";
        ok = false;
    }
    return ok;
}

static bool test_legacy_layer_parser_contract() {
    using namespace ggml::gemmini::types;
    const bool ok = layer_name_view(nullptr).empty() &&
        layer_name_view("---").empty() &&
        layer_name_view("--attn_norm-tail") == "attn_norm" &&
        parse_layer("--attn_norm-tail") == LayerType::attn_norm &&
        parse_layer(std::string_view("ffn_gate_par")) == LayerType::ffn_gate_par &&
        parse_layer(std::string_view("not-a-layer")) == LayerType::unknown &&
        std::string_view(to_string(LayerType::result_norm)) == "result_norm";
    if (!ok) {
        std::cerr << "legacy layer parser contract changed\n";
    }
    return ok;
}

int main(int argc, char ** argv) {
    if (argc == 2 && std::string(argv[1]) == "--llama-callback") {
        return test_llama_callback_replacement() ? 0 : 2;
    }
    if (argc == 2 && std::string(argv[1]) == "--happy") {
        const std::optional<std::filesystem::path> root = create_test_root();
        if (!root) return 2;
        const bool ok = test_atomic_cycle_sink(*root);
        ggml::gemmini::log::cycle.set_output(stderr);
        std::error_code error;
        std::filesystem::remove_all(*root, error);
        return ok && !error ? 0 : 2;
    }
    if (argc == 3 && std::string(argv[1]) == "--fault") {
        return run_fault_probe(argv[2]) ? 0 : 2;
    }
    if (argc == 2 && std::string(argv[1]) == "--explicit-debug-before-default") {
        return test_explicit_debug_before_default_setup() ? 0 : 2;
    }

    const bool preserve_prefix_sentinel = argc == 3 &&
        std::string(argv[2]) == "--preserve-prefix-sentinel";
    if (argc != 2 && !preserve_prefix_sentinel) {
        std::cerr << "usage: test-gemmini-log PATH_PREFIX [--preserve-prefix-sentinel]\n";
        return 1;
    }

    const std::filesystem::path caller_root = std::filesystem::absolute(std::string(argv[1]) + ".paths");
    const std::filesystem::path caller_sentinel = caller_root / "unrelated-sentinel";
    if (preserve_prefix_sentinel && !std::filesystem::is_regular_file(caller_sentinel)) {
        std::cerr << "adversarial harness requires an unrelated caller sentinel\n";
        return 1;
    }
    const std::optional<std::filesystem::path> created_root = create_test_root();
    if (!created_root) {
        return 1;
    }
    const std::filesystem::path test_root = *created_root;
    const std::filesystem::path cycle_path = test_root / "cycle.jsonl";
    const std::filesystem::path debug_path = test_root / "debug.jsonl";
    const std::filesystem::path original_cwd = std::filesystem::current_path();
    const std::optional<std::string> original_env = []() -> std::optional<std::string> {
        if (const char * value = std::getenv("GEMMINI_LOG_DIR")) {
            return value;
        }
        return std::nullopt;
    }();
    bool failed = !test_legacy_layer_parser_contract() ||
        !test_semantic_matmul_layer_resolver();

    if (!test_default_setup(test_root)) {
        failed = true;
    }
    {
        ScopedCurrentPath cwd;
        ScopedEnv env("GEMMINI_LOG_DIR");
        std::filesystem::current_path(test_root);

        env.set(nullptr);
        failed |= !expect_path("null path", ggml::gemmini::log::resolve_output_path(nullptr), {});
        failed |= !expect_path("empty path", ggml::gemmini::log::resolve_output_path(""), {});
        const std::filesystem::path absolute_path = test_root / "absolute.jsonl";
        failed |= !expect_path("absolute path",
                               ggml::gemmini::log::resolve_output_path(absolute_path.string().c_str()),
                               absolute_path);
        failed |= !expect_path("ordinary relative path",
                               ggml::gemmini::log::resolve_output_path("ordinary/relative.jsonl"),
                               test_root / "output/log/ordinary/relative.jsonl");
        failed |= !expect_path("short ordinary relative path",
                               ggml::gemmini::log::resolve_output_path("l"), test_root / "output/log/l");
        failed |= !expect_path("logical default path",
                               ggml::gemmini::log::resolve_output_path("log/nested/default.jsonl"),
                               test_root / "output/log/nested/default.jsonl");
    }
    if (!environment_is_restored(original_cwd, original_env)) {
        std::cerr << "unset case did not restore CWD/environment\n";
        failed = true;
    }

    {
        ScopedCurrentPath cwd;
        ScopedEnv env("GEMMINI_LOG_DIR");
        std::filesystem::current_path(test_root);
        env.set("");
        failed |= !expect_path("empty override path",
                               ggml::gemmini::log::resolve_output_path("log/empty.jsonl"),
                               test_root / "output/log/empty.jsonl");
    }
    if (!environment_is_restored(original_cwd, original_env)) {
        std::cerr << "empty override case did not restore CWD/environment\n";
        failed = true;
    }

    {
        ScopedCurrentPath cwd;
        ScopedEnv env("GEMMINI_LOG_DIR");
        std::filesystem::current_path(test_root);
        const std::filesystem::path override_root = test_root / "override";
        env.set(override_root.string().c_str());
        failed |= !expect_path("override ordinary path",
                               ggml::gemmini::log::resolve_output_path("ordinary/relative.jsonl"),
                               override_root / "ordinary/relative.jsonl");
        failed |= !expect_path("override prefix stripping",
                               ggml::gemmini::log::resolve_output_path("log/nested/override.jsonl"),
                               override_root / "nested/override.jsonl");
        failed |= !expect_path("override traversal rejection",
                               ggml::gemmini::log::resolve_output_path("../escaped.jsonl"), {});
        if (ggml::gemmini::log::truncate_file("../escaped.jsonl") ||
            std::filesystem::exists(test_root / "escaped.jsonl")) {
            std::cerr << "relative traversal escaped GEMMINI_LOG_DIR\n";
            failed = true;
        }
        std::error_code symlink_error;
        std::filesystem::create_directories(override_root, symlink_error);
        const std::filesystem::path outside = test_root / "outside";
        std::filesystem::create_directories(outside, symlink_error);
        std::filesystem::create_directory_symlink(outside, override_root / "link", symlink_error);
        if (!symlink_error && !ggml::gemmini::log::resolve_output_path("link/escaped.jsonl").empty()) {
            std::cerr << "relative symlink escaped GEMMINI_LOG_DIR\n";
            failed = true;
        }
    }
    if (!environment_is_restored(original_cwd, original_env)) {
        std::cerr << "non-empty override case did not restore CWD/environment\n";
        failed = true;
    }

    {
        ScopedCurrentPath cwd;
        ScopedEnv env("GEMMINI_LOG_DIR");
        std::filesystem::current_path(test_root);
        env.set(nullptr);
        const std::filesystem::path created = test_root / "output/log/created/parent/output.jsonl";
        if (!ggml::gemmini::log::truncate_file("log/created/parent/output.jsonl") ||
            !std::filesystem::exists(created) ||
            std::filesystem::exists(test_root / "log")) {
            std::cerr << "logical output parent creation failed\n";
            failed = true;
        }
    }
    if (!environment_is_restored(original_cwd, original_env)) {
        std::cerr << "parent creation case did not restore CWD/environment\n";
        failed = true;
    }
    const std::string json_sentinel = "{\"sentinel\":1}\n";
    write_file(cycle_path, json_sentinel.c_str());
    ggml::gemmini::log::cycle.set_output(stderr);
    ggml::gemmini::log::cycle(ggml::gemmini::log::file(cycle_path.c_str()), "test", "append", 1, 2);
    if (read_file(cycle_path).rfind(json_sentinel, 0) != 0 || !every_line_is_json(cycle_path)) {
        std::cerr << "cycle output truncated existing records\n";
        failed = true;
    }

#if !EXPECT_LOG_DEBUG
    std::filesystem::remove(debug_path);
    if (!ggml::gemmini::log::debug.set_output_path(debug_path.c_str())) {
        std::cerr << "disabled debug output setup failed\n";
        failed = true;
    }
    ggml::gemmini::log::debug.set_output(stderr);
    if (std::filesystem::exists(debug_path)) {
        std::cerr << "disabled debug output created a file\n";
        failed = true;
    }
#endif

    using ggml::gemmini::log::testing::TargetWriteKind;
    if (!test_hardware_counter_contract(test_root) ||
        !test_atomic_cycle_sink(test_root) || !test_atomic_debug_sink(test_root) ||
        !test_target_lock_blocks_replacement(test_root, TargetWriteKind::plain) ||
        !test_target_lock_blocks_replacement(test_root, TargetWriteKind::layer) ||
        !test_target_lock_blocks_replacement(test_root, TargetWriteKind::location) ||
        !test_setup_failures(test_root)) {
        failed = true;
    }

    ggml::gemmini::log::debug.set_output(stderr);
    ggml::gemmini::log::cycle.set_output(stderr);
    std::filesystem::remove(cycle_path);
    std::filesystem::remove(debug_path);
    if (preserve_prefix_sentinel && read_file(caller_sentinel) != "caller-owned\n") {
        std::cerr << "caller-owned sentinel was not preserved\n";
        failed = true;
    }
    std::error_code cleanup_error;
    std::filesystem::remove_all(test_root, cleanup_error);
    if (cleanup_error) {
        std::cerr << "could not remove test directory: " << cleanup_error.message() << '\n';
        failed = true;
    }
    return failed ? 2 : 0;
}
