// Standalone native accounting test. The parent supplies the diagnostics define,
// planning-profile.cpp, thread support, and exactly the six documented wraps.
#include "matmul/planning-profile.hpp"

#ifdef POTAL_PLANNING_DIAGNOSTICS

#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <thread>
#include <vector>

extern "C" {
void * __wrap__Znwm(std::size_t bytes);
void * __wrap__Znam(std::size_t bytes);
void __wrap__ZdlPv(void * pointer) noexcept;
void __wrap__ZdaPv(void * pointer) noexcept;
void __wrap__ZdlPvm(void * pointer, std::size_t bytes) noexcept;
void __wrap__ZdaPvm(void * pointer, std::size_t bytes) noexcept;
}

using namespace ggml::potal::planning_profile;

namespace {

template <typename T>
constexpr size_t index(T value) {
    return static_cast<size_t>(value);
}

void check(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "planning-profile-test: %s\n", message);
        std::abort();
    }
}

// The escaped pointer makes the allocation observable even in optimized builds.
// GNU compiler syntax matches the GNU linker wrapping required by this test.
__attribute__((noinline)) void consume(const void * pointer) {
    asm volatile("" : : "g"(pointer) : "memory");
}

void enable(const char * value) {
    check(setenv("POTAL_PLANNING_PROFILE", value, 1) == 0, "setenv failed");
}

const ComponentStats & component(const PhaseStats & phase, Component value) {
    return phase.components[index(value)];
}

const SiteStats & site(const PhaseStats & phase, Site value) {
    return phase.sites[index(value)];
}

void check_partition(const PhaseStats & phase) {
    uint64_t component_ns = 0;
    uint64_t site_ns = 0;
    uint64_t component_bytes = 0;
    uint64_t site_bytes = 0;
    uint64_t component_objects = 0;
    uint64_t site_objects = 0;
    for (const auto & value : phase.components) {
        component_ns += value.exclusive_ns;
        component_bytes += value.bytes;
        component_objects += value.objects;
    }
    for (const auto & value : phase.sites) {
        site_ns += value.exclusive_ns;
        site_bytes += value.bytes;
        site_objects += value.objects;
        check(value.exclusive_ns <= value.inclusive_ns, "site exclusive time exceeds inclusive time");
    }
    check(component_ns == phase.total_ns, "components do not partition total time");
    check(site_ns == phase.total_ns, "sites do not partition total time");
    check(component_bytes == site_bytes, "component and site bytes differ");
    check(component_objects == site_objects, "component and site objects differ");
    check(site(phase, Site::route_overhead).inclusive_ns == phase.total_ns,
          "implicit root scope does not contain the full interval");
}

void check_empty(const Snapshot & data) {
    for (const auto & phase : data.phases) {
        check(phase.roots == 0 && phase.total_ns == 0 && phase.rows_sum == 0 &&
                  phase.n == 0 && phase.k == 0 && !phase.mixed_shape,
              "disabled or reset root accounting is nonzero");
        for (const auto & value : phase.components) {
            check(value.calls == 0 && value.exclusive_ns == 0 && value.bytes == 0 && value.objects == 0,
                  "disabled or reset component accounting is nonzero");
        }
        for (const auto & value : phase.sites) {
            check(value.calls == 0 && value.exclusive_ns == 0 && value.inclusive_ns == 0 &&
                      value.bytes == 0 && value.objects == 0,
                  "disabled or reset site accounting is nonzero");
        }
    }
}

void test_disabled() {
    FILE * output = std::tmpfile();
    check(output != nullptr, "tmpfile failed");
    check(unsetenv("POTAL_PLANNING_PROFILE") == 0, "unsetenv failed");
    reset();
    {
        Root root(Phase::route, 1, 2, 3);
        Scope scope(Site::fragment_input, Component::validation);
        scope.add_bytes(4);
    }
    check_empty(snapshot());
    for (const char * value : {"0", "", "01", "1x", "true"}) {
        enable(value);
        {
            Root root(Phase::route, 1, 2, 3);
            Scope scope(Site::fragment_build, Component::construction);
            std::vector<int> values(64, 7);
            consume(values.data());
            scope.add_bytes(64);
            scope.finish();
            root.finish();
        }
        check_empty(snapshot());
        dump(output, "disabled", "supplied");
        check(std::ftell(output) == 0, "disabled dump emitted output");
    }
    std::fclose(output);
}

void test_nested_accounting() {
    enable("1");
    reset();
    {
        Root root(Phase::route, 2, 8, 32);
        Scope build(Site::fragment_build, Component::construction);
        {
            Scope input(Site::fragment_input, Component::validation);
            input.add_bytes(64, 2);
            input.finish();
            input.finish();
        }
        {
            Scope rows(Site::fragment_rows, Component::validation);
            rows.add_bytes(12);
        }
        {
            Root nested(Phase::executor, 999, 999, 999);
            Scope copy(Site::builder_rows_copy, Component::copy);
            copy.add_bytes(16, 4);
            nested.finish();
        }
        build.finish();
        build.finish();
        Scope tile(Site::tile_build, Component::construction);
        Scope frozen(Site::tile_frozen, Component::validation);
        // Root.finish must close both remaining inclusive intervals once.
        root.finish();
        root.finish();
        frozen.add_bytes(999);
        frozen.finish();
        tile.finish();
    }
    const Snapshot data = snapshot();
    const auto & route = data.phases[index(Phase::route)];
    check(route.roots == 1 && route.rows_sum == 2 && route.n == 8 && route.k == 32,
          "outer root dimensions or count differ");
    check(data.phases[index(Phase::executor)].roots == 0, "nested root emitted accounting");
    check(component(route, Component::construction).calls == 2, "construction count differs");
    check(component(route, Component::validation).calls == 3, "validation count differs");
    check(component(route, Component::copy).calls == 1, "copy count differs");
    check(component(route, Component::other).calls == 1, "implicit scope count differs");
    check(component(route, Component::instrumentation).calls == 14, "bookkeeping count differs");
    check(component(route, Component::allocation).calls == 0 &&
              component(route, Component::deallocation).calls == 0,
          "accounting itself allocated");
    check(component(route, Component::validation).bytes == 76 &&
              component(route, Component::validation).objects == 3,
          "validation inventory differs");
    check(component(route, Component::copy).bytes == 16 && component(route, Component::copy).objects == 4,
          "copy inventory differs");
    for (size_t i = 0; i < route.sites.size(); ++i) {
        const Site value = static_cast<Site>(i);
        const bool used = value == Site::route_overhead || value == Site::fragment_build ||
                          value == Site::fragment_input || value == Site::fragment_rows ||
                          value == Site::builder_rows_copy || value == Site::tile_build || value == Site::tile_frozen;
        check(route.sites[i].calls == (used ? 1u : 0u), "site count differs");
    }
    check(site(route, Site::fragment_input).bytes == 64 && site(route, Site::fragment_input).objects == 2 &&
              site(route, Site::fragment_rows).bytes == 12 && site(route, Site::fragment_rows).objects == 1 &&
              site(route, Site::builder_rows_copy).bytes == 16 && site(route, Site::builder_rows_copy).objects == 4,
          "site inventory differs");
    check(site(route, Site::fragment_build).inclusive_ns >=
              site(route, Site::fragment_input).inclusive_ns + site(route, Site::fragment_rows).inclusive_ns +
                  site(route, Site::builder_rows_copy).inclusive_ns,
          "fragment children escape inclusive parent");
    check(site(route, Site::tile_build).inclusive_ns >= site(route, Site::tile_frozen).inclusive_ns,
          "frozen scope escapes inclusive parent");
    check(route.total_ns >=
              site(route, Site::fragment_build).inclusive_ns + site(route, Site::tile_build).inclusive_ns,
          "root does not contain disjoint scopes");
    check_partition(route);
}

void test_finished_copy_inventory() {
    reset();
    {
        Root root(Phase::route, 1, 8, 32);
        Scope copy(Site::builder_tile_copy, Component::copy);
        copy.finish();
        {
            Scope inventory(Site::payload_inventory, Component::instrumentation);
            copy.add_bytes(96, 3);
        }
        copy.finish();
        root.finish();
        copy.add_bytes(999);
    }
    const auto phase = snapshot().phases[index(Phase::route)];
    const auto & copy = site(phase, Site::builder_tile_copy);
    const auto & inventory = site(phase, Site::payload_inventory);
    check(copy.calls == 1 && copy.bytes == 96 && copy.objects == 3, "finished copy lost its inventory");
    check(copy.exclusive_ns == copy.inclusive_ns, "post-finish inventory changed the copy timer");
    check(inventory.calls == 1 && inventory.exclusive_ns == inventory.inclusive_ns,
          "inventory traversal escaped its instrumentation site");
    check(component(phase, Component::copy).calls == 1 &&
              component(phase, Component::copy).bytes == 96 && component(phase, Component::copy).objects == 3,
          "finished copy component inventory differs");
    check(component(phase, Component::instrumentation).calls == 7,
          "inventory scope or bookkeeping count differs");
    check_partition(phase);
}

void test_root_reuse_and_explicit_finish() {
    reset();
    alignas(Root) unsigned char storage[sizeof(Root)];
    Root * first = new (storage) Root(Phase::route, 3, 8, 32);
    Scope stale(Site::fragment_build, Component::construction);
    first->~Root();
    Root * second = new (storage) Root(Phase::executor, 4, 8, 32);
    stale.add_bytes(999);
    stale.finish();
    {
        Scope outer(Site::execution_build, Component::construction);
        Scope inner(Site::execution_validate, Component::validation);
        outer.finish();
        inner.add_bytes(7);
        inner.finish();
        inner.finish();
    }
    second->finish();
    second->~Root();
    const Snapshot data = snapshot();
    const auto & route = data.phases[index(Phase::route)];
    const auto & executor = data.phases[index(Phase::executor)];
    check(route.roots == 1 && executor.roots == 1 && executor.rows_sum == 4,
          "separate root lifetime accounting differs");
    check(site(executor, Site::fragment_build).calls == 0, "stale scope attached to a replacement root");
    check(component(executor, Component::validation).bytes == 7, "explicit unlink lost the child scope");
    check(component(route, Component::construction).bytes == 0, "finished root accepted new inventory");
    check_partition(route);
    check_partition(executor);
}

void test_vector_and_wrappers() {
    reset();
    {
        Root root(Phase::executor, 1, 64, 32);
        {
            std::vector<int> values(64, 7);
            consume(values.data());
        }
    }
    const PhaseStats vector = snapshot().phases[index(Phase::executor)];
    check(component(vector, Component::allocation).calls == 1 &&
              component(vector, Component::allocation).bytes == 64 * sizeof(int) &&
              component(vector, Component::allocation).objects == 1,
          "actual vector allocation was not wrapped exactly once");
    check(component(vector, Component::deallocation).calls == 1 &&
              component(vector, Component::deallocation).objects == 1,
          "actual vector destruction was not wrapped exactly once");
    const uint64_t deleted_bytes = component(vector, Component::deallocation).bytes;
    check(deleted_bytes == 0 || deleted_bytes == 64 * sizeof(int),
          "vector delete reported bytes not supplied by its ABI");
    check_partition(vector);

    reset();
    {
        Root root(Phase::executor, 1, 8, 32);
        void * scalar = __wrap__Znwm(17);
        void * array = __wrap__Znam(19);
        consume(scalar);
        consume(array);
        __wrap__ZdlPv(scalar);
        __wrap__ZdaPv(array);
        scalar = __wrap__Znwm(23);
        array = __wrap__Znam(29);
        consume(scalar);
        consume(array);
        __wrap__ZdlPvm(scalar, 23);
        __wrap__ZdaPvm(array, 29);
    }
    const PhaseStats wrappers = snapshot().phases[index(Phase::executor)];
    check(component(wrappers, Component::allocation).calls == 4 &&
              component(wrappers, Component::allocation).bytes == 88 &&
              component(wrappers, Component::allocation).objects == 4,
          "scalar and array allocation requests differ");
    check(component(wrappers, Component::deallocation).calls == 4 &&
              component(wrappers, Component::deallocation).bytes == 52 &&
              component(wrappers, Component::deallocation).objects == 4,
          "sized and unsized deletion inventory differs");
    check(site(wrappers, Site::allocate).calls == 4 && site(wrappers, Site::deallocate).calls == 4,
          "allocation wrapper site counts differ");
    check_partition(wrappers);
}

void test_exception_propagation() {
    reset();
    bool caught = false;
    {
        Root root(Phase::executor, 1, 8, 32);
        Scope outer(Site::executor_request, Component::construction);
        try {
            // An impossible request exercises real throwing new, not a mock or
            // a test replacement allocator. No custom new_handler is installed.
            void * pointer = __wrap__Znwm((std::numeric_limits<size_t>::max)());
            consume(pointer);
            __wrap__ZdlPv(pointer);
        } catch (const std::bad_alloc &) {
            caught = true;
        }
        Scope after(Site::supplied_match, Component::validation);
        after.add_bytes(3);
    }
    check(caught, "allocation exception did not propagate");
    const PhaseStats phase = snapshot().phases[index(Phase::executor)];
    check(component(phase, Component::allocation).calls == 1 &&
              component(phase, Component::allocation).bytes == (std::numeric_limits<size_t>::max)() &&
              component(phase, Component::allocation).objects == 1,
          "throwing request inventory differs");
    check(component(phase, Component::deallocation).calls == 0, "throwing allocation produced a delete");
    check(site(phase, Site::supplied_match).bytes == 3, "unwinding corrupted the active scope stack");
    check_partition(phase);
}

void test_thread_aggregation() {
    reset();
    auto worker = [](size_t rows, size_t n) {
        Root root(Phase::route, rows, n, 32);
        Scope scope(Site::fragment_rows, Component::validation);
        scope.add_bytes(rows);
    };
    std::thread producer(worker, 2, 8);
    std::thread owner(worker, 3, 8);
    producer.join();
    owner.join();
    auto phase = snapshot().phases[index(Phase::route)];
    check(phase.roots == 2 && phase.rows_sum == 5 && phase.n == 8 && phase.k == 32 && !phase.mixed_shape,
          "producer and owner roots did not merge");
    check(site(phase, Site::fragment_rows).calls == 2 && site(phase, Site::fragment_rows).bytes == 5,
          "thread-local scope totals differ");
    check_partition(phase);
    worker(1, 16);
    phase = snapshot().phases[index(Phase::route)];
    check(phase.roots == 3 && phase.rows_sum == 6 && phase.mixed_shape && phase.n == 0 && phase.k == 0,
          "mixed shape reporting differs");
    check_partition(phase);
}

void test_reset_snapshot_and_dump() {
    FILE * output = std::tmpfile();
    check(output != nullptr, "tmpfile failed");
    // Prepare the C stream before accounting; dump belongs outside planning.
    char buffer[32768];
    check(std::setvbuf(output, buffer, _IOFBF, sizeof(buffer)) == 0, "setvbuf failed");
    reset();
    check_empty(snapshot());
    {
        Root root(Phase::route, 5, 8, 32);
        check_empty(snapshot()); // Active, unmerged roots are not published.
        // reset requires idle workers, so it is intentionally not called here.
    }
    dump(output, "fixture", "supplied");
    check(std::fflush(output) == 0 && std::ftell(output) > 0, "enabled dump produced no output");
    std::rewind(output);
    char line[1024];
    size_t totals = 0;
    size_t components = 0;
    size_t sites = 0;
    while (std::fgets(line, sizeof(line), output) != nullptr) {
        char phase[32], case_name[32], route[32], name[64];
        unsigned long long calls, exclusive, inclusive, bytes, objects, roots, total, rows, n, k;
        if (std::sscanf(line,
                        "[POTAL_PLANNING_TOTAL] phase=%31s case=%31s route=%31s roots=%llu total_ns=%llu"
                        " rows_sum=%llu n=%llu k=%llu",
                        phase, case_name, route, &roots, &total, &rows, &n, &k) == 8) {
            ++totals;
            check(std::strcmp(phase, "route") == 0 || std::strcmp(phase, "executor") == 0, "unknown output phase");
            if (std::strcmp(phase, "route") == 0) {
                check(roots == 1 && rows == 5 && n == 8 && k == 32, "dump root fields differ");
            } else {
                check(roots == 0, "dump invented executor roots");
            }
        } else if (std::sscanf(line,
                               "[POTAL_PLANNING_COMPONENT] phase=%31s case=%31s route=%31s component=%63s"
                               " calls=%llu exclusive_ns=%llu bytes=%llu objects=%llu",
                               phase, case_name, route, name, &calls, &exclusive, &bytes, &objects) == 8) {
            ++components;
        } else if (std::sscanf(line,
                               "[POTAL_PLANNING_SITE] phase=%31s case=%31s route=%31s site=%63s"
                               " calls=%llu exclusive_ns=%llu inclusive_ns=%llu bytes=%llu objects=%llu",
                               phase, case_name, route, name, &calls, &exclusive, &inclusive, &bytes, &objects) == 9) {
            ++sites;
        } else {
            check(false, "unparseable diagnostic record");
        }
        check(std::strcmp(case_name, "fixture") == 0 && std::strcmp(route, "supplied") == 0,
              "dump case or route field differs");
    }
    check(!std::ferror(output), "reading dump failed");
    check(totals == index(Phase::count) && components == index(Phase::count) * index(Component::count) &&
              sites == index(Phase::count) * index(Site::count),
          "dump record counts differ");
    // reset and dump execute only at the idle graph boundaries. A following
    // allocation-free root also checks that no diagnostic state leaks into TLS.
    {
        Root root(Phase::executor, 1, 8, 32);
    }
    const auto phase = snapshot().phases[index(Phase::executor)];
    check(component(phase, Component::allocation).calls == 0 &&
              component(phase, Component::deallocation).calls == 0,
          "diagnostic management left accounting active");
    std::fclose(output);
    reset();
    check_empty(snapshot());
}

} // namespace

int main() {
    test_disabled();
    enable("1");
    test_nested_accounting();
    test_finished_copy_inventory();
    test_root_reuse_and_explicit_finish();
    test_vector_and_wrappers();
    test_exception_propagation();
    test_thread_aggregation();
    test_reset_snapshot_and_dump();
    check(unsetenv("POTAL_PLANNING_PROFILE") == 0, "unsetenv failed");
    return 0;
}

#endif // POTAL_PLANNING_DIAGNOSTICS
