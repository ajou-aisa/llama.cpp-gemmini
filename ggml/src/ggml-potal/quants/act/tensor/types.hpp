#pragma once

// Copied from ggml-gemmini/quants/act/tensor/types.hpp without the RMD /
// direct residual payloads (error compensation is not part of ggml-potal).

namespace ggml::potal::quants::act::tensor {

struct Meta {
    float scale = 1.0f;

    inline void reset() {
        scale = 1.0f;
    }
};

} // namespace ggml::potal::quants::act::tensor
