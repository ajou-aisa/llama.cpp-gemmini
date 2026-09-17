#pragma once
#include "trace-context.h"
#include "cpu-timing.h"
#include "performance.hpp"
#include <string>
#include <string_view>
#include <exception>

namespace ggml::gemmini::trace {
performance::Context inference_context(const gemmini_trace_context &context) noexcept;
std::string append_metadata(std::string json, const gemmini_trace_context &context,
                            uint64_t segment_id);

std::string annotate_origin(std::string json, const gemmini_trace_context &context);

class ScopedContext {
public:
    explicit ScopedContext(gemmini_trace_context context, bool fork_task = false) noexcept
        : previous_(gemmini_trace_bind(fork_task ? gemmini_trace_fork(context) : context)) {}
    ~ScopedContext() noexcept { gemmini_trace_restore(previous_); }
    ScopedContext(const ScopedContext &) = delete;
    ScopedContext &operator=(const ScopedContext &) = delete;
private:
    gemmini_trace_context previous_{};
};

class ScopedRole {
public:
    explicit ScopedRole(uint8_t role) noexcept : previous_(gemmini_trace_capture()) {
        auto next = previous_; next.role = role; saved_ = gemmini_trace_bind(next);
    }
    ~ScopedRole() noexcept { gemmini_trace_restore(saved_); }
private:
    gemmini_trace_context previous_{}, saved_{};
};

/* A measured stage belongs to one executing task. Thread handoff carries only
 * ScopedContext's value snapshot; a CPU sample is never continued on a peer. */
class CpuStage {
public:
    CpuStage(const char *layer, const char *stage) noexcept
        : layer_(layer), stage_(stage), exceptions_(std::uncaught_exceptions()),
          start_(gemmini_cpu_timing_read()) {}
    ~CpuStage() noexcept { finish(std::uncaught_exceptions() == exceptions_); }
    void finish(bool success = true) noexcept;
    CpuStage(const CpuStage &) = delete;
    CpuStage &operator=(const CpuStage &) = delete;
private:
    const char *layer_, *stage_;
    int exceptions_;
    gemmini_cpu_sample start_{};
    bool finished_ = false;
};
}
