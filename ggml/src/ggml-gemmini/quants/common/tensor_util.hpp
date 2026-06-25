#pragma once

#include "../../ggml-gemmini-args.h"

namespace ggml::gemmini {

inline const float *activation_data(const ggml_tensor *tensor) {
    if (!tensor)
        return nullptr;

    const char *base = reinterpret_cast<const char *>(tensor->view_src ? tensor->view_src->data : tensor->data);
    const size_t offs = tensor->view_src ? tensor->view_offs : 0;
    return reinterpret_cast<const float *>(base + offs);
}

inline const block_q8_0 *weight_block_base(const ggml_tensor *tensor) {
    if (!tensor)
        return nullptr;

    const char *base = reinterpret_cast<const char *>(tensor->view_src ? tensor->view_src->data : tensor->data);
    const size_t offs = tensor->view_src ? tensor->view_offs : 0;
    return reinterpret_cast<const block_q8_0 *>(base + offs);
}

}
