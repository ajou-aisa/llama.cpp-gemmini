#pragma once

#if defined(GGML_GEMMINI_PB)
#include "pb-exec.hpp"

namespace ggml::gemmini::pb {
bool attention_supported(const ggml_tensor * op);
Status execute_attention(ggml_tensor * op);
}
#endif
