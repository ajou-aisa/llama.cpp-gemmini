#pragma once

#include "ggml.h"
#include <cstdint>
#include <functional>
#include <string>

namespace ggml::gemmini::sensitivity {
int anchor(const ggml_tensor * node);
int reported_anchor();
bool profiling();
bool shadow();
std::string key(const ggml_tensor * node);
struct Snapshot {
    uint64_t main = 0, residual = 0, saturations = 0;
    double seconds = 0;
};
Snapshot snapshot();
// The replay borrows the original inputs and restores every output byte.
void observe(ggml_tensor * node, Snapshot before, const std::function<void()> & replay);
bool finish();
} // namespace ggml::gemmini::sensitivity
