#pragma once

#include <cstdint>
#include <string>

namespace ggml::gemmini::cycle {

uint64_t host_thread_id() noexcept;
std::string serialize_host_timing(uint64_t start_ns, uint64_t end_ns,
                                  uint64_t start_tid, uint64_t end_tid);

}
