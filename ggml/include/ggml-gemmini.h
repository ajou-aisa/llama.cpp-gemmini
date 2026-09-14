#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

// Optional FPGA_UART runtime query via reg.get_proc_address("ggml_gemmini_fpga_stats_v1").
// The caller must synchronize execution before taking its final snapshot.
struct ggml_gemmini_fpga_stats_v1 {
    uint64_t assigned;   // MUL_MAT nodes actually visited by graph_compute
    uint64_t attempted;  // calls to the SCU adapter; preflight may reject before UART
    uint64_t completed;  // adapter success after output commit and normal RELEASE
    uint64_t failed;     // failed assigned nodes, with no numerical retry
};
typedef bool (*ggml_gemmini_fpga_stats_v1_fn)(struct ggml_gemmini_fpga_stats_v1 *, size_t);

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_gemmini_init(void);

// GGML_BACKEND_API bool ggml_backend_is_gemmini(ggml_backend_t backend);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_gemmini_reg(void);


#ifdef  __cplusplus
}
#endif
