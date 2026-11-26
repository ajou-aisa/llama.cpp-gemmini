#include "../ggml/src/ggml-npu-common/profiling/include/npu/cycle_reader.hpp"
#include <cstdio>

int main() {
    using namespace ggml::npu::cycle;
    reset_read_count_for_test();
    const auto counter = read();
    const auto timestamp = timestamp_ns();
#if LOG_CYCLE
    const bool enabled = true;
#else
    const bool enabled = false;
#endif
    if (read_count_for_test() != (enabled ? 2u : 0u) ||
        (!enabled && (counter != 0 || timestamp != 0)) ||
        (enabled && timestamp == 0)) {
        std::fprintf(stderr, "reader gate mismatch: expected enabled=%d\n", enabled);
        return 1;
    }
    return 0;
}
