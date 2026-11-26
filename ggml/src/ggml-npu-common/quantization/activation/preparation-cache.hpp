#pragma once

#include "../../buffer/activation.hpp"
#include "exsia/types.hpp"
#include <string>
#include <tuple>

struct ggml_tensor;
struct npu_matmul_args;

namespace ggml::npu::quants {
// One projection family per backend; never shared across concurrent backends.
class PreparationCache {
  public:
    struct Result { bool ok; bool reused; };
    Result prepare(const ggml_tensor * projection, npu_matmul_args & args,
                   bool (*reuse_disabled)() = nullptr);
    void clear();

  private:
    using Key = std::tuple<std::string, const ggml_tensor *, const void *, size_t, size_t,
                           size_t, int, int, int>;
    Key key_{};
    unsigned roles_ = 0;
    std::vector<uint8_t> input_;
    act::QuantizedActivationBuffer main_;
    act::exsia::Meta metadata_;
};
} // namespace ggml::npu::quants
