#include "weight_cache.hpp"

#include "ggml-potal-buffer.hpp"
#include "../log.hpp"

#include <chrono>
#include <cstdlib>
#include <new>
#include <stdexcept>

namespace ggml::potal {

namespace {

using clock_type = std::chrono::steady_clock;

double seconds_since(clock_type::time_point t0) {
    if (t0 == clock_type::time_point{}) return 0;
    return std::chrono::duration<double>(clock_type::now() - t0).count();
}

size_t budget_from_env() {
    const char * v = std::getenv("GGML_POTAL_WEIGHT_CACHE_MB");
    if (v == nullptr || *v == '\0') {
        return size_t(1024) << 20;
    }
    char *             end = nullptr;
    const unsigned long long mb = std::strtoull(v, &end, 10);
    if (end == v || *end != '\0' || mb > (SIZE_MAX >> 20)) {
        return size_t(1024) << 20; // malformed: keep the default
    }
    return (size_t) mb << 20;
}

} // namespace

const char * weight_cache_outcome_name(weight_cache_outcome o) {
    switch (o) {
        case weight_cache_outcome::hit:      return "HIT";
        case weight_cache_outcome::miss:     return "MISS";
        case weight_cache_outcome::bypass:   return "BYPASS";
        case weight_cache_outcome::uncached: return "UNCACHED";
        case weight_cache_outcome::failed:   return "FAILED";
    }
    return "unknown";
}

weight_cache::weight_cache() : weight_cache(budget_from_env()) {
    eager_grow_ = std::getenv("GGML_POTAL_WEIGHT_CACHE_MB") == nullptr;
}

weight_cache::weight_cache(size_t budget_bytes) {
    c_.budget = budget_bytes;
}

bool weight_cache::eligible(const ggml_tensor * src0) {
    const ggml_tensor * base = src0->view_src ? src0->view_src : src0;
    return base->buffer != nullptr && ggml_backend_buffer_get_usage(base->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
}

weight_cache::key weight_cache::key_of(const ggml_tensor * t) {
    const ggml_tensor * base = t->view_src ? t->view_src : t;
    return { static_cast<const void *>(base->buffer), potal_buffer_generation(base->buffer), t->data, (int) t->type,
             t->ne[0], t->ne[1], t->ne[2], t->ne[3], t->nb[0], t->nb[1], t->nb[2], t->nb[3] };
}

ggml_status weight_cache::prepare(const ggml_tensor * src0) {
    try {
        std::lock_guard<std::mutex> lock(mutex_);
        if (src0 == nullptr || !eligible(src0) || c_.budget == 0) return GGML_STATUS_SUCCESS;
        const ggml_tensor * base = src0->view_src ? src0->view_src : src0;
        if (!ggml_backend_buffer_is_host(base->buffer)) return GGML_STATUS_SUCCESS;
        const key k = key_of(src0);
        if (entries_.find(k) != entries_.end()) return GGML_STATUS_SUCCESS;

        npu_matmul_args args;
        if (!init_weight_args(src0, args)) {
            log::failure("potal.init", "[POTAL_INIT] weight=%s invalid shape/layout or missing static data",
                         src0->name);
            return GGML_STATUS_FAILED;
        }
        size_t bytes;
        if (!hp1_weight::required_bytes(src0->ne[1], src0->ne[0], bytes) || bytes > SIZE_MAX - c_.bytes) {
            return GGML_STATUS_ALLOC_FAILED;
        }
        const size_t total = c_.bytes + bytes;
        if (!eager_grow_ && total > c_.budget) return GGML_STATUS_ALLOC_FAILED;

        std::unique_ptr<hp1_weight> w(new (std::nothrow) hp1_weight);
        if (!w || !w->allocate(src0->ne[1], src0->ne[0])) return GGML_STATUS_ALLOC_FAILED;
        auto reader = ggml::npu::quants::wreader::WeightReaderStatus::Success;
        const auto status = prepare_weight(args, *w, &reader);
        if (status != quants::weight::hp1::status::ok) {
            log::failure("potal.init", "[POTAL_INIT] weight=%s prepare=%s reader=%s",
                         src0->name, quants::weight::hp1::status_name(status),
                         ggml::npu::quants::wreader::weight_reader_status_name(reader));
            return GGML_STATUS_FAILED;
        }
        log::debug("potal.init", "[POTAL_INIT] prepared weight=%s bytes=%zu", src0->name, bytes);
        const auto inserted = entries_.emplace(k, std::move(w));
        if (!inserted.second) return GGML_STATUS_ALLOC_FAILED;
        c_.entries++;
        c_.bytes = total;
        if (eager_grow_ && total > c_.budget) c_.budget = total;
        return GGML_STATUS_SUCCESS;
    } catch (const std::bad_alloc &) {
        return GGML_STATUS_ALLOC_FAILED;
    } catch (const std::length_error &) {
        return GGML_STATUS_ALLOC_FAILED;
    }
}

weight_lookup weight_cache::get(const ggml_tensor * src0, const npu_matmul_args & args) {
    weight_lookup r;
    const bool    cacheable = eligible(src0) && c_.budget != 0;
    // Held through the preparation: a second request for the same weight waits
    // for the first instead of preparing it again.
    std::lock_guard<std::mutex> lock(mutex_);
    const key                   k = key_of(src0);
    if (cacheable) {
        const auto it = entries_.find(k);
        if (it != entries_.end()) {
            c_.hits++;
            r.outcome = weight_cache_outcome::hit;
            r.weight  = it->second.get();
            return r;
        }
    }

    const bool measure = log::timing_enabled();
    auto t = measure ? clock_type::now() : clock_type::time_point{};
    std::unique_ptr<hp1_weight> w(new (std::nothrow) hp1_weight);
    const bool                  allocated = w != nullptr && w->allocate(src0->ne[1], src0->ne[0]);
    r.alloc_s                             = seconds_since(t);
    if (!allocated) {
        c_.failures++;
        r.alloc_failed = true;
        return r;
    }
    t          = measure ? clock_type::now() : clock_type::time_point{};
    r.status   = prepare_weight(args, *w, &r.reader);
    r.prepare_s = seconds_since(t);
    if (r.status != quants::weight::hp1::status::ok) {
        c_.failures++;
        return r; // never cached: the next use validates again
    }

    const size_t bytes = w->bytes();
    if (cacheable && bytes <= c_.budget - c_.bytes) {
        r.weight = w.get();
        try {
            entries_.emplace(k, std::move(w));
        } catch (const std::bad_alloc &) {
            // w was not moved from on a failed emplace: keep it for this call only.
            r.outcome   = weight_cache_outcome::bypass;
            r.transient = std::move(w);
            c_.bypasses++;
            return r;
        }
        c_.misses++;
        c_.entries++;
        c_.bytes += bytes;
        r.outcome = weight_cache_outcome::miss;
        return r;
    }
    r.outcome   = cacheable ? weight_cache_outcome::bypass : weight_cache_outcome::uncached;
    (cacheable ? c_.bypasses : c_.uncached)++;
    r.weight    = w.get();
    r.transient = std::move(w);
    return r;
}

void weight_cache::abandon(const hp1_weight * w) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto it = entries_.begin(); it != entries_.end(); ++it) {
        if (it->second.get() == w) {
            c_.bytes -= it->second->bytes();
            c_.entries--;
            c_.abandoned++;
            it->second->abandon();
            entries_.erase(it);
            return;
        }
    }
}

void weight_cache::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    entries_.clear();
    c_.entries = 0;
    c_.bytes   = 0;
}

weight_cache::counters weight_cache::snapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return c_;
}

} // namespace ggml::potal
