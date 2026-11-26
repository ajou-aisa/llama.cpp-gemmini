#pragma once

#include "../ggml-npu-common/types/matmul.hpp"
#include <gemmini_params.h>

#if defined(GGML_GEMMINI_CONFIGURED_DIM)
static_assert(DIM == GGML_GEMMINI_CONFIGURED_DIM,
              "Gemmini parameter header DIM does not match configured DIM");
#endif

static_assert(sizeof(elem_t) == GGML_GEMMINI_ACTIVATION_STORAGE_BYTES,
              "elem_t must match configured activation transport storage");
static_assert(sizeof(elem_t) == GGML_GEMMINI_WEIGHT_STORAGE_BYTES,
              "elem_t must match configured weight transport storage");
static_assert(GGML_GEMMINI_ACTIVATION_BITS == 16 ? std::is_same_v<elem_t, int16_t>
                                                 : std::is_same_v<elem_t, int8_t>,
              "elem_t must be int16_t only for A16/W16, otherwise int8_t");


enum tiled_matmul_type_t : int { OS, WS, CPU };
struct gemmini_transport;

namespace ggml::gemmini { using namespace ggml::npu; }

// Hardware invocation state stays with the Gemmini executor.
struct ggml_gemmini_args_t : npu_matmul_args {
    // Optional owned command context. Copies/slices retain it until their final drain.
    // An aliasing shared_ptr must retain the transport context and its backing buffers.
    // Empty selects the immutable RoCC transport; it never selects another invocation.
    std::shared_ptr<const gemmini_transport> transport;
    // scales, gemmini input val.
    scale_t     scale_B    = 1.f;
    scale_acc_t scale_D    = 1;
    int         act        = 0; // default NO_ACTIVATION
    acc_scale_t scale      = 1.0f;
    acc_scale_t bert_scale = 1.0f;
    bool repeating_bias = false;
    bool transpose_A    = false;
    bool full_C         = true;
    bool low_D          = false;
    // for weight checking
    uint8_t             weightA           = 0;
    tiled_matmul_type_t tiled_matmul_type = static_cast<tiled_matmul_type_t>(0);
};

#if defined(GGML_GEMMINI_TEST_OBSERVER)
namespace ggml::gemmini {
using test_i_observer_t = void (*)(const char * consumer, size_t I, void * user_data);

GGML_API void set_test_i_observer(test_i_observer_t observer, void * user_data);
} // namespace ggml::gemmini
#endif

#if defined(GGML_GEMMINI_TESTING) || defined(GGML_GEMMINI_TEST_OBSERVER)
namespace ggml::gemmini {
enum class TestSemanticLayerSite : uint8_t {
    fp_facade,
    physical_auto_fp,
    physical_set_tile_ws,
    physical_block_quantized,
    physical_block_quantized_auto,
    physical_baseline_dense,
};
using test_semantic_layer_observer_t = bool (*)(TestSemanticLayerSite site,
                                                const char *          layer,
                                                void *                user_data);

GGML_API void set_test_semantic_layer_observer(test_semantic_layer_observer_t observer,
                                               void *                         user_data);
GGML_API bool test_observe_semantic_layer(TestSemanticLayerSite site, const char * layer);
GGML_API bool test_probe_physical_layer_sites(const ggml_gemmini_args_t & args);
GGML_API bool test_probe_physical_null_args();
GGML_API bool test_probe_fp_facade_layer(const std::string & layer);
} // namespace ggml::gemmini
#endif

#if defined(GGML_GEMMINI_TESTING)
namespace ggml::gemmini {
GGML_API bool test_hp1_native_weight_admission_contract();
GGML_API std::string test_resolve_backend_matmul_layer(std::string_view model_arch,
                                                       std::string_view weight_name,
                                                       std::string_view input_name,
                                                       std::string_view consumer_name);
GGML_API void        test_reset_unclassified_matmul_diagnostics();
GGML_API size_t      test_unclassified_matmul_diagnostic_count();
} // namespace ggml::gemmini
#endif
