#include "gemmini/log.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <optional>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

static std::string read_file(const std::filesystem::path & path) {
    std::ifstream input(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

static void write_file(const std::filesystem::path & path, const char * content) {
    std::ofstream(path, std::ios::binary) << content;
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
    std::ofstream(debug_path, std::ios::app) << "debug-sentinel\n";
#endif
#if EXPECT_LOG_CYCLE
    if (!std::filesystem::exists(cycle_path)) {
        std::cerr << "enabled default cycle output was not created\n";
        return false;
    }
    std::ofstream(cycle_path, std::ios::app) << "cycle-sentinel\n";
#endif

    threads.clear();
    for (size_t i = 0; i < thread_count; ++i) {
        threads.emplace_back([] { (void)ggml::gemmini::log::setup_default_outputs(); });
    }
    for (std::thread & thread : threads) {
        thread.join();
    }

#if EXPECT_LOG_DEBUG
    if (read_file(debug_path).find("debug-sentinel\n") == std::string::npos) {
        std::cerr << "repeated setup truncated default debug output\n";
        return false;
    }
#else
    (void)ggml::gemmini::log::debug.set_output_path(GEMMINI_LOG_DEFAULT_DEBUG_PATH, true);
    (void)gemmini_log_debug_set_output_path(GEMMINI_LOG_DEFAULT_DEBUG_PATH);
    gemmini_log_ws_loop(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
    if (std::filesystem::exists(debug_path) ||
        std::filesystem::exists(test_root / "output/log/log-ws-loop.jsonl")) {
        std::cerr << "disabled debug setup created a default output\n";
        return false;
    }
#endif
#if EXPECT_LOG_CYCLE
    if (read_file(cycle_path).find("cycle-sentinel\n") == std::string::npos) {
        std::cerr << "repeated setup truncated default cycle output\n";
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
                output.find("explicit-sentinel-after-default") == std::string::npos) {
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

int main(int argc, char ** argv) {
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
    const std::string sentinel = "sentinel\n";
    bool failed = false;

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
                               "ordinary/relative.jsonl");
        failed |= !expect_path("short ordinary relative path",
                               ggml::gemmini::log::resolve_output_path("l"), "l");
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
    write_file(cycle_path, sentinel.c_str());
    ggml::gemmini::log::cycle(ggml::gemmini::log::file(cycle_path.c_str()), "test", "append", 1, 2);
    if (read_file(cycle_path).rfind(sentinel, 0) != 0) {
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
