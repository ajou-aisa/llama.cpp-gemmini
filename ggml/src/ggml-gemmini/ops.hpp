#pragma once

#include <string>

#include "ggml-backend.h"

struct ggml_backend_gemmini_context {
    std::string model_arch;
};

void ggml_backend_gemmini_mul_mat(ggml_backend_gemmini_context * ctx, struct ggml_tensor * dst);

void ggml_backend_gemmini_get_rows_q8_channel(const ggml_tensor * src0,
                                              const ggml_tensor * src1,
                                              ggml_tensor *       dst);

bool ggml_backend_gemmini_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op);

void setup_gemmini_log_outputs_if_needed(void);
