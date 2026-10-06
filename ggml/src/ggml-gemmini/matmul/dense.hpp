#pragma once

#include "../ggml-gemmini-args.h"
#include "types.hpp"

namespace ggml::gemmini {

class MatmulStripeJob;
class MatmulExecution;
class MatmulStripeInput;
struct MatmulStatus;

class MatMul {
  public:
    explicit MatMul(ggml_gemmini_args_t args);
    explicit MatMul(ggml_gemmini_args_t * args);
    MatMul(MatMul && other) noexcept;
    MatMul & operator=(MatMul && other) noexcept;
    ~MatMul();
    MatMul(const MatMul &)             = delete;
    MatMul & operator=(const MatMul &) = delete;

    MatMulResult run_dense();
    MatMulResult run_full();
    MatMulStatus begin_stripes();
    MatMulStatus run_stripe(MatMulStripe stripe);
    MatMulStatus finish_stripes();

    static MatMulCapability stripe_capability(const ggml_gemmini_args_t & args);
    MatMulState             state() const;

  private:
    friend class MatmulExecution;
    friend class MatmulStripeCollector;
    friend MatmulStatus    execute_full(MatmulExecution &);
    friend MatmulStatus    finish_execution(MatmulExecution &);
    friend MatmulStripeJob capture_stripe(MatmulExecution &, MatmulStripeInput);
    friend MatmulStripeJob
    capture_stripe(MatmulExecution &, MatmulStripeInput, rmd::StripePacketHandle);
    friend MatmulStripeJob capture_stripe(MatmulExecution &,
                                          MatmulStripeInput,
                                          residual::DirectStripePayloadHandle,
                                          rmd::StripePacketHandle);
    friend MatmulStatus    execute_dense_stripe(MatmulStripeJob &);
    friend MatmulStatus    accept_external_dense_completion(MatmulStripeJob &);
    friend MatmulStatus    execute_rmd_stripe(MatmulStripeJob &);
    friend MatmulStatus    compose_rmd_stripe(MatmulStripeJob &);
    friend MatmulStatus    finalize_stripe(MatmulStripeJob &);

    MatMulResult run_dense(bool transactional);
    MatMulStatus run_stripe(MatMulStripe stripe, size_t stripe_id);
    MatMulStatus run_staged_stripe(MatMulStripe              stripe,
                                   size_t                    stripe_id,
                                   const quants::act::Meta & activation_metadata);
    MatMulStatus begin_output_transaction();
    void         commit_output_transaction();
    void         discard_output_transaction();

    ggml_gemmini_args_t &       args();
    const ggml_gemmini_args_t & args() const;

    ggml_gemmini_args_t   owned_args_{};
    ggml_gemmini_args_t * args_ptr_           = nullptr;
    size_t                first_row_          = 0;
    size_t                last_row_begin_     = 0;
    size_t                last_row_end_       = 0;
    size_t                covered_rows_       = 0;
    bool                  has_stripes_        = false;
    MatMulState           state_              = MatMulState::idle;
    float *               output_destination_ = nullptr;
    size_t                output_row_stride_  = 0;
    size_t                output_col_stride_  = 0;
    std::vector<float>    output_stage_;
};

} // namespace ggml::gemmini
