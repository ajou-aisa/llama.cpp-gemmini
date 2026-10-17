#pragma once

// Copied from ggml-gemmini/quants/common/tensor_util.hpp (Q8_0 helper removed).

#include "../../ggml-potal-args.h"

namespace ggml::potal {

inline const void * tensor_data(const ggml_tensor * tensor) {
    if (!tensor)
        return nullptr;

    const char * base =
        reinterpret_cast<const char *>(tensor->view_src ? tensor->view_src->data : tensor->data);
    const size_t offs = tensor->view_src ? tensor->view_offs : 0;
    return base + offs;
}

inline const float * activation_data(const ggml_tensor * tensor) {
    return reinterpret_cast<const float *>(tensor_data(tensor));
}

inline const block_q8_hp1 * weight_hp1_base(const ggml_tensor * tensor) {
    return reinterpret_cast<const block_q8_hp1 *>(tensor_data(tensor));
}

} // namespace ggml::potal
