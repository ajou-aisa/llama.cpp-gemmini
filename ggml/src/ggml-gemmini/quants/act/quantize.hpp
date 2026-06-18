#pragma once

#include "exsia/types.hpp"
#include "types.hpp"

#include <vector>

struct ggml_tensor;
struct ggml_gemmini_args_t;

namespace ggml::gemmini::quants
{
void quantize_activation(const ggml_tensor *src, ggml_gemmini_args_t &args);

void reset_activation_quant_state(ggml_gemmini_args_t &args);

void init_exsia_meta(ggml_gemmini_args_t &args);
const act::exsia::Meta &get_exsia_meta(const ggml_gemmini_args_t &args);
act::exsia::Meta &get_exsia_meta_mut(ggml_gemmini_args_t &args);
std::vector<QactOutlier> activation_outliers(const ggml_gemmini_args_t &args);

} // namespace ggml::gemmini::quants
