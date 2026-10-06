#pragma once

#include "types.hpp"
#include "exsia-event.hpp"
#include "exsia-profile.hpp"
#include "exsia-state.hpp"
#include "local.hpp"
#include "folding.hpp"

#include "../../../residual/residual-capture.hpp"

#include <algorithm>
#include <atomic>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include <gemmini/layer.hpp>
#include <gemmini/cpu-timing.h>
#include <gemmini/cpu_log_context.hpp>
#include <gemmini/evaluation_metrics.hpp>
#if CYCLE_SIM
#include <gemmini/cycle_sim_log.hpp>
#endif

#ifndef GGML_GEMMINI_EXSIA_DEFAULT_MODE_VALUE
#define GGML_GEMMINI_EXSIA_DEFAULT_MODE_VALUE 0
#endif

struct ggml_tensor;
struct ggml_gemmini_args_t;

namespace ggml::gemmini::quants::act::exsia {
uint64_t next_exsia_run_id();

std::array<ExSIAState::ExecutionModeAvailability, 3> execution_mode_availability();

const char * failure_code_name(ExSIAState::FailureCode code) noexcept;
const char * failure_origin_name(ExSIAState::FailureCode code) noexcept;

class ExSIA {
  private:
    ExSIAState state_;
#if GGML_GEMMINI_EXSIA_DEFAULT_MODE_VALUE == 1
    ExSIAState::ExecutionMode requested_mode_ = ExSIAState::ExecutionMode::LocalParallel;
#elif GGML_GEMMINI_EXSIA_DEFAULT_MODE_VALUE == 2
    ExSIAState::ExecutionMode requested_mode_ = ExSIAState::ExecutionMode::LocalFoldingPipeline;
#else
    ExSIAState::ExecutionMode requested_mode_ = ExSIAState::ExecutionMode::Sequential;
#endif
    std::array<StripePipelineSlot, EXSIA_PIPELINE_SLOT_COUNT> pipeline_slots_;
    LocalExecutionWorkspace                                   local_workspace_;
    LocalStage                                                local_;
    StripeFolding                                             folding_;
    std::atomic<ExSIAState::FailureCode> first_failure_code_{ExSIAState::FailureCode::None};
    std::atomic<size_t>                  first_failure_stripe_{ExSIAState::no_failure_stripe};

    void reset_failure_state();
    void record_failure(ExSIAState::FailureCode code,
                        size_t                  stripe = ExSIAState::no_failure_stripe);

  public:
    void set_execution_mode(ExSIAState::ExecutionMode mode) {
        requested_mode_ = mode;
    }

    bool run(Meta & meta, const ggml_tensor * src, ggml_gemmini_args_t & args);

    bool run(Meta &                  meta,
             const ggml_tensor *     src,
             ggml_gemmini_args_t &   args,
             const StripeReadySink * sink);

    const ExSIAState & state() const {
        return state_;
    }

    bool ownership_ready() const {
        for (const StripePipelineSlot & slot : pipeline_slots_) {
            if (slot.lifecycle != StripePipelineSlotState::Released)
                return false;
        }
        return local_workspace_.workers.size() == EXSIA_LOCAL_WORKER_COUNT;
    }
};

bool dequantize_activation(float *                     dst,
                           size_t                      dst_row_stride,
                           size_t                      dst_col_stride,
                           size_t                      rows,
                           size_t                      cols,
                           const ggml_gemmini_args_t & args);

} // namespace ggml::gemmini::quants::act::exsia
