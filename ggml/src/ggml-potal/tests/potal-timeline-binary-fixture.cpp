// Real shutdown writer exercised in both formats over the same captured rings.
#include "trace/timeline.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <unistd.h>

namespace tl = potal::timeline;

int main(int argc, char ** argv) {
    if (argc != 2 || !tl::enabled()) return 2;
    const unsigned count = unsigned(std::stoul(argv[1]));
    const std::string path = std::getenv("POTAL_TIMELINE");
    const auto record = [count](tl::role role) {
        tl::register_thread(role, -1, -1);
        const auto run = tl::begin_run(tl::op_type::qk, tl::phase::prefill, 5);
        for (unsigned j = 0; j < count; ++j) {
            const auto stage = j % 256 == 0 ? tl::stage::exsia_folding : tl::stage::acc_main_issue;
            tl::record(stage, tl::event::begin, 0, run, 5, j / 256, j, 1, 9, UINT64_MAX - j);
        }
        // Exercise the CSV writer's missing-op/phase fallback and exact 64-bit counters.
        auto & ring = *tl::detail::t_ring;
        ring.events[0].op_type = 0;
        ring.events[0].phase = 0;
        if (ring.cpu_samples != nullptr) {
            ring.cpu_samples[0].cycles = UINT64_MAX - 1;
            ring.cpu_samples[0].instructions = (uint64_t(1) << 60) + 123;
            ring.cpu_samples[0].cpu = -1;
            // Synthetic serializer fixture: retain flags independently of host PMU availability.
            ring.cpu_samples[0].thread_valid = false;
            ring.cpu_samples[0].process_valid = true;
            ring.cpu_samples[0].counters_valid = true;
        }
        tl::end_run(run);
    };
    record(tl::role::caller);
    std::thread worker([&] { record(tl::role::exsia_fold); });
    worker.join();
    if (::access(path.c_str(), F_OK) == 0) return 3; // no writes during recording
    for (const char * format : {"binary", "csv"}) {
        if (::setenv("POTAL_TIMELINE_FORMAT", format, 1) != 0) return 4;
        const auto begin = std::chrono::steady_clock::now();
        if (!tl::flush()) return 5;
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
        const std::string destination = path + "." + format;
        if (std::rename(path.c_str(), destination.c_str()) != 0) return 6;
        if (format == std::string("csv") && tl::detail::g_cpu &&
            std::rename((path + ".cpu.csv").c_str(), (destination + ".cpu.csv").c_str()) != 0) return 7;
        std::printf("%s events=%u flush_seconds=%.6f\n", format, 2 * count, seconds);
    }
    return 0;
}
