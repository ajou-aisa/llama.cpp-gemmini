// Only this executable redirects ops.cpp. Do not compile ops.cpp separately.
#include "matmul/stripe.hpp"

namespace ggml::potal {
potal_status e4c1_tile_select(size_t, size_t, size_t, uint32_t, potal_tile_plan *);
pipeline_report e4c1_run_pipeline(const ggml_tensor *, ggml_gemmini_args_t &, const hp1_weight &,
                                 const potal_tile_plan &, const pipeline_config &, stripe_owner &);
}

#define potal_tile_select e4c1_tile_select
#define run_stripe_pipeline e4c1_run_pipeline
#include "../ops.cpp"
#undef run_stripe_pipeline
#undef potal_tile_select
