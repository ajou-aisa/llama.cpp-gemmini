#pragma once

#include <npu_config.h>
#include <cstdint>
#include <type_traits>

// CPU logical storage is distinct from architecture-specific packed DMA storage.
using npu_element_t = std::conditional_t<GGML_GEMMINI_ACTIVATION_BITS == 16, int16_t, int8_t>;
