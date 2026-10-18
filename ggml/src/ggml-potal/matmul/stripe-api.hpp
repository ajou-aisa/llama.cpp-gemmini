#pragma once

// potal_tiled_matmul_stripe() and its session (potal.h) on the tiled WS path
// of tiling.hpp. The opaque potal_stripe_session is a potal_ws_session: the
// issuer owns exactly one (execution.cpp) and runs the I8 GEMM, the FULL tiled
// jobs and the planned stripe GEMMs on it, so all of them share one scale
// generation sequence and one set of work counters. Tests bind an emulator by
// constructing the session with their own ops.

#include "potal.h"
#include "tiling.hpp"

struct potal_stripe_session {
    ggml::potal::potal_ws_session ws;
};
