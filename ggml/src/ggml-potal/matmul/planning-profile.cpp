#include "planning-profile.hpp"

#ifdef POTAL_PLANNING_DIAGNOSTICS

#include <cstdlib>
#include <inttypes.h>
#include <mutex>

namespace ggml::potal::planning_profile {
namespace {

thread_local Root * active_root = nullptr;
thread_local uint64_t next_generation = 0;
std::mutex totals_mutex;
Snapshot totals;

constexpr const char * phase_names[] = {"route", "executor"};
constexpr const char * component_names[] = {
    "validation", "copy", "allocation", "deallocation", "construction", "other", "instrumentation"
};
constexpr const char * site_names[] = {
    "route_overhead", "fragment_build", "fragment_input", "fragment_rows",
    "tile_build", "tile_input", "tile_frozen",
    "execution_build", "execution_input", "execution_validate", "executor_request",
    "supplied_match", "builder_tile_copy", "builder_rows_copy", "report_plan_copy",
    "report_tile_copy", "payload_inventory", "allocate", "deallocate"
};
static_assert(sizeof(phase_names) / sizeof(*phase_names) == static_cast<size_t>(Phase::count));
static_assert(sizeof(component_names) / sizeof(*component_names) == static_cast<size_t>(Component::count));
static_assert(sizeof(site_names) / sizeof(*site_names) == static_cast<size_t>(Site::count));

template <typename T>
constexpr size_t index(T value) {
    return static_cast<size_t>(value);
}

bool enabled() {
    const char * value = std::getenv("POTAL_PLANNING_PROFILE");
    return value != nullptr && value[0] == '1' && value[1] == '\0';
}

uint64_t elapsed(std::chrono::steady_clock::time_point from,
                 std::chrono::steady_clock::time_point to) {
    // Quantize timestamps before subtraction so adjacent intervals telescope
    // exactly even on a clock whose native duration is finer than nanoseconds.
    const auto first = std::chrono::duration_cast<std::chrono::nanoseconds>(from.time_since_epoch()).count();
    const auto last = std::chrono::duration_cast<std::chrono::nanoseconds>(to.time_since_epoch()).count();
    return static_cast<uint64_t>(last - first);
}

void merge(PhaseStats & destination, const PhaseStats & source) {
    if (destination.roots == 0) {
        destination.n = source.n;
        destination.k = source.k;
    } else if (destination.n != source.n || destination.k != source.k) {
        destination.mixed_shape = true;
    }
    if (destination.mixed_shape) {
        destination.n = 0;
        destination.k = 0;
    }
    destination.roots += source.roots;
    destination.total_ns += source.total_ns;
    destination.rows_sum += source.rows_sum;
    for (size_t i = 0; i < destination.components.size(); ++i) {
        auto & d = destination.components[i];
        const auto & s = source.components[i];
        d.calls += s.calls;
        d.exclusive_ns += s.exclusive_ns;
        d.bytes += s.bytes;
        d.objects += s.objects;
    }
    for (size_t i = 0; i < destination.sites.size(); ++i) {
        auto & d = destination.sites[i];
        const auto & s = source.sites[i];
        d.calls += s.calls;
        d.exclusive_ns += s.exclusive_ns;
        d.inclusive_ns += s.inclusive_ns;
        d.bytes += s.bytes;
        d.objects += s.objects;
    }
}

} // namespace

Root::Root(Phase phase, size_t rows, size_t n, size_t k) : phase_(phase) {
    if (active_root != nullptr || !enabled()) {
        return;
    }
    generation_ = ++next_generation;
    stats_.roots = 1;
    stats_.rows_sum = rows;
    stats_.n = n;
    stats_.k = k;
    stats_.components[index(Component::other)].calls = 1;
    stats_.sites[index(Site::route_overhead)].calls = 1;
    active_ = true;
    active_root = this;
    start_ = last_ = Clock::now();
}

Root::~Root() {
    finish();
}

void Root::charge(Clock::time_point now) {
    const Component component = top_ ? top_->component_ : Component::other;
    const Site site = top_ ? top_->site_ : Site::route_overhead;
    const uint64_t ns = elapsed(last_, now);
    stats_.components[index(component)].exclusive_ns += ns;
    stats_.sites[index(site)].exclusive_ns += ns;
    last_ = now;
}

void Root::bookkeeping(Clock::time_point now, Site site) {
    auto & component = stats_.components[index(Component::instrumentation)];
    const uint64_t ns = elapsed(last_, now);
    ++component.calls;
    component.exclusive_ns += ns;
    stats_.sites[index(site)].exclusive_ns += ns;
    last_ = now;
}

void Root::finish() {
    if (!active_) {
        return;
    }
    const auto now = Clock::now();
    charge(now);
    // Close inclusive intervals for scopes that outlive this root. Their later
    // destructors use the TLS identity check and never dereference this root.
    for (Scope * scope = top_; scope != nullptr; scope = scope->parent_) {
        stats_.sites[index(scope->site_)].inclusive_ns += elapsed(scope->start_, now);
    }
    const auto end = Clock::now();
    bookkeeping(end, Site::route_overhead);
    stats_.total_ns = elapsed(start_, end);
    stats_.sites[index(Site::route_overhead)].inclusive_ns += stats_.total_ns;
    active_root = nullptr;
    active_ = false;
    std::lock_guard<std::mutex> lock(totals_mutex);
    merge(totals.phases[index(phase_)], stats_);
}

Scope::Scope(Site site, Component component) : site_(site), component_(component) {
    if (active_root == nullptr) {
        return;
    }
    root_ = active_root;
    generation_ = root_->generation_;
    start_ = Root::Clock::now();
    root_->charge(start_);
    parent_ = root_->top_;
    root_->top_ = this;
    ++root_->stats_.sites[index(site_)].calls;
    ++root_->stats_.components[index(component_)].calls;
    root_->bookkeeping(Root::Clock::now(), site_);
}

Scope::~Scope() {
    finish();
}

bool Scope::attached() const {
    // Test TLS first: root_ may refer to an object whose lifetime has ended.
    return root_ != nullptr && active_root == root_ &&
           active_root->generation_ == generation_;
}

void Scope::finish() {
    if (finished_ || !attached()) {
        finished_ = true;
        return;
    }
    const auto now = Root::Clock::now();
    root_->charge(now);
    Scope ** link = &root_->top_;
    while (*link != this) {
        link = &(*link)->parent_;
    }
    *link = parent_;
    finished_ = true;
    const auto end = Root::Clock::now();
    root_->bookkeeping(end, site_);
    root_->stats_.sites[index(site_)].inclusive_ns += elapsed(start_, end);
}

void Scope::add_bytes(uint64_t bytes, uint64_t objects) {
    if (!attached()) {
        return;
    }
    root_->charge(Root::Clock::now());
    auto & site = root_->stats_.sites[index(site_)];
    auto & component = root_->stats_.components[index(component_)];
    site.bytes += bytes;
    site.objects += objects;
    component.bytes += bytes;
    component.objects += objects;
    const Site current_site = root_->top_ ? root_->top_->site_ : Site::route_overhead;
    root_->bookkeeping(Root::Clock::now(), current_site);
}

void reset() {
    std::lock_guard<std::mutex> lock(totals_mutex);
    totals = Snapshot{};
}

Snapshot snapshot() {
    std::lock_guard<std::mutex> lock(totals_mutex);
    return totals;
}

void dump(FILE * stream, const char * case_name, const char * route_name) {
    if (!enabled()) {
        return;
    }
    const Snapshot data = snapshot();
    for (size_t p = 0; p < data.phases.size(); ++p) {
        const auto & phase = data.phases[p];
        std::fprintf(stream,
                     "[POTAL_PLANNING_TOTAL] phase=%s case=%s route=%s roots=%" PRIu64
                     " total_ns=%" PRIu64 " rows_sum=%" PRIu64 " n=%" PRIu64 " k=%" PRIu64 "\n",
                     phase_names[p], case_name, route_name, phase.roots, phase.total_ns,
                     phase.rows_sum, phase.n, phase.k);
        for (size_t c = 0; c < phase.components.size(); ++c) {
            const auto & component = phase.components[c];
            std::fprintf(stream,
                         "[POTAL_PLANNING_COMPONENT] phase=%s case=%s route=%s component=%s"
                         " calls=%" PRIu64 " exclusive_ns=%" PRIu64 " bytes=%" PRIu64 " objects=%" PRIu64 "\n",
                         phase_names[p], case_name, route_name, component_names[c],
                         component.calls, component.exclusive_ns, component.bytes, component.objects);
        }
        for (size_t s = 0; s < phase.sites.size(); ++s) {
            const auto & site = phase.sites[s];
            std::fprintf(stream,
                         "[POTAL_PLANNING_SITE] phase=%s case=%s route=%s site=%s"
                         " calls=%" PRIu64 " exclusive_ns=%" PRIu64 " inclusive_ns=%" PRIu64
                         " bytes=%" PRIu64 " objects=%" PRIu64 "\n",
                         phase_names[p], case_name, route_name, site_names[s],
                         site.calls, site.exclusive_ns, site.inclusive_ns, site.bytes, site.objects);
        }
    }
}

} // namespace ggml::potal::planning_profile

// These are ABI call-through wrappers, not replacement allocation operators.
// In particular, a throwing allocation unwinds Scope and reaches its caller.
extern "C" {
void * __real__Znwm(std::size_t bytes);
void * __real__Znam(std::size_t bytes);
void __real__ZdlPv(void * pointer) noexcept;
void __real__ZdaPv(void * pointer) noexcept;
void __real__ZdlPvm(void * pointer, std::size_t bytes) noexcept;
void __real__ZdaPvm(void * pointer, std::size_t bytes) noexcept;
void * __wrap__Znwm(std::size_t bytes);
void * __wrap__Znam(std::size_t bytes);
void __wrap__ZdlPv(void * pointer) noexcept;
void __wrap__ZdaPv(void * pointer) noexcept;
void __wrap__ZdlPvm(void * pointer, std::size_t bytes) noexcept;
void __wrap__ZdaPvm(void * pointer, std::size_t bytes) noexcept;

void * __wrap__Znwm(std::size_t bytes) {
    using namespace ggml::potal::planning_profile;
    if (active_root == nullptr) {
        return __real__Znwm(bytes);
    }
    Scope scope(Site::allocate, Component::allocation);
    scope.add_bytes(bytes);
    return __real__Znwm(bytes);
}

void * __wrap__Znam(std::size_t bytes) {
    using namespace ggml::potal::planning_profile;
    if (active_root == nullptr) {
        return __real__Znam(bytes);
    }
    Scope scope(Site::allocate, Component::allocation);
    scope.add_bytes(bytes);
    return __real__Znam(bytes);
}

void __wrap__ZdlPv(void * pointer) noexcept {
    using namespace ggml::potal::planning_profile;
    if (active_root == nullptr) {
        __real__ZdlPv(pointer);
        return;
    }
    Scope scope(Site::deallocate, Component::deallocation);
    scope.add_bytes(0);
    __real__ZdlPv(pointer);
}

void __wrap__ZdaPv(void * pointer) noexcept {
    using namespace ggml::potal::planning_profile;
    if (active_root == nullptr) {
        __real__ZdaPv(pointer);
        return;
    }
    Scope scope(Site::deallocate, Component::deallocation);
    scope.add_bytes(0);
    __real__ZdaPv(pointer);
}

void __wrap__ZdlPvm(void * pointer, std::size_t bytes) noexcept {
    using namespace ggml::potal::planning_profile;
    if (active_root == nullptr) {
        __real__ZdlPvm(pointer, bytes);
        return;
    }
    Scope scope(Site::deallocate, Component::deallocation);
    scope.add_bytes(bytes);
    __real__ZdlPvm(pointer, bytes);
}

void __wrap__ZdaPvm(void * pointer, std::size_t bytes) noexcept {
    using namespace ggml::potal::planning_profile;
    if (active_root == nullptr) {
        __real__ZdaPvm(pointer, bytes);
        return;
    }
    Scope scope(Site::deallocate, Component::deallocation);
    scope.add_bytes(bytes);
    __real__ZdaPvm(pointer, bytes);
}
} // extern "C"

#endif // POTAL_PLANNING_DIAGNOSTICS
