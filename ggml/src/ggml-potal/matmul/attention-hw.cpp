#include "attention.hpp"

#include "stripe-api.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <new>
#include <stdexcept>
#include <vector>

namespace ggml::potal::attention {

planned_result run_corrected(const potal_stripe_gemm_desc & desc, const int32_t * bias, size_t columns,
                             const potal_paired_sink * sink) {
    planned_result result;
    potal_paired_session * session = nullptr;
    using clock = std::chrono::steady_clock;
    const auto seconds = [](clock::time_point start) {
        return std::chrono::duration<double>(clock::now() - start).count();
    };
    struct group {
        std::vector<potal_paired_loop_ws> loops;
        std::vector<const uint32_t *> carriers;
        potal_acc_bias bias{};
        potal_acc_readback output{};
    };
    try {
        prepared_stripe_gemm prepared;
        result.status = prepare_stripe_gemm(desc, prepared, &result.stage);
        if (result.status != POTAL_OK) {
            return result;
        }
        if (columns != desc.n || bias == nullptr) {
            result.status = POTAL_ERR_INVALID_ARGUMENT;
            result.stage = "attention_bias_operand";
            return result;
        }
        const auto & plan = prepared.plan;
        std::vector<group> groups;
        // All descriptor, capacity and pointer checks precede the first command.
        for (size_t i = 0; i < desc.m; i += plan.tile_I * POTAL_HW_DIM) {
            const size_t rows = std::min(plan.tile_I * POTAL_HW_DIM, size_t(desc.m) - i);
            for (size_t j = 0; j < desc.n; j += plan.tile_J * POTAL_HW_DIM) {
                const size_t cols = std::min(plan.tile_J * POTAL_HW_DIM, size_t(desc.n) - j);
                group current;
                current.bias = { 0, uint32_t((rows + POTAL_HW_DIM - 1) / POTAL_HW_DIM),
                                 uint32_t((cols + POTAL_HW_DIM - 1) / POTAL_HW_DIM),
                                 uint32_t((POTAL_HW_DIM - rows % POTAL_HW_DIM) % POTAL_HW_DIM),
                                 uint32_t((POTAL_HW_DIM - cols % POTAL_HW_DIM) % POTAL_HW_DIM), bias + j };
                current.output = { 0, current.bias.i_tiles, current.bias.j_tiles,
                                   current.bias.pad_i, current.bias.pad_j,
                                   desc.c + i * desc.ldc + j, desc.ldc };
                result.status = potal_acc_bias_check(&current.bias);
                if (result.status == POTAL_OK) {
                    result.status = potal_acc_readback_check(&current.output);
                }
                if (result.status != POTAL_OK) {
                    result.stage = "attention_bias_preflight";
                    return result;
                }
                for (size_t k = 0; k < desc.k; k += plan.tile_K * POTAL_HW_DIM) {
                    const size_t count = std::min(plan.tile_K * POTAL_HW_DIM, size_t(desc.k) - k);
                    potal_paired_loop_ws request{};
                    auto & loop = request.loop;
                    loop.a = desc.a + i * desc.lda + k;
                    loop.b = desc.b + k * desc.ldb + j;
                    loop.lda = desc.lda;
                    loop.ldb = desc.ldb;
                    loop.i_tiles = current.bias.i_tiles;
                    loop.j_tiles = current.bias.j_tiles;
                    loop.k_tiles = uint32_t(count / POTAL_HW_DIM);
                    loop.pad_i = current.bias.pad_i;
                    loop.pad_j = current.bias.pad_j;
                    loop.scale_generation = 1;
                    loop.passes_per_block = uint32_t(desc.scale_block_k / POTAL_HW_DIM);
                    loop.flags = k == 0 ? 0 : POTAL_GEMM_ACCUMULATE;
                    request.path = POTAL_PAIRED_MAIN; // Known physical ACC0, also for upper P digits.
                    result.status = potal_paired_loop_ws_check(&request);
                    if (result.status != POTAL_OK) {
                        result.stage = "attention_pv_loop_preflight";
                        return result;
                    }
                    current.loops.push_back(request);
                    current.carriers.push_back(desc.carriers + (k / desc.scale_block_k) * desc.carrier_stride + j);
                }
                groups.push_back(std::move(current));
            }
        }
        result.status = potal_paired_session_open(sink, &session);
        if (result.status != POTAL_OK) {
            result.stage = "attention_pv_open";
            return result;
        }
        auto accept = [&](potal_status status, const char * stage) {
            if (status != POTAL_OK) {
                result.status = status;
                result.stage = stage;
                throw status;
            }
        };
        accept(potal_paired_tlb_flush(session), "attention_pv_tlb");
        accept(potal_paired_fence(session), "attention_pv_fence");
        result.tlb_flushes = 1;
        uint32_t generation = 0;
        for (auto & current : groups) {
            for (size_t chunk = 0; chunk < current.loops.size(); ++chunk) {
                auto & request = current.loops[chunk];
                auto & loop = request.loop;
                generation = generation % POTAL_GENERATION_MAX + 1;
                loop.scale_generation = generation;
                result.report.last_generation = generation;
                const uint32_t blocks = loop.k_tiles / loop.passes_per_block;
                const auto valid_columns = [&](uint32_t j) {
                    return POTAL_HW_DIM - (j + 1 == loop.j_tiles ? loop.pad_j : 0);
                };
                auto start = clock::now();
                for (uint32_t block = 0; block < blocks; ++block) {
                    for (uint32_t j = 0; j < loop.j_tiles; ++j) {
                        for (uint32_t col = 0; col < valid_columns(j); ++col) {
                            accept(potal_paired_scale_write(session, block * loop.j_tiles + j, generation, col,
                                current.carriers[chunk][block * desc.carrier_stride + j * POTAL_HW_DIM + col]),
                                "attention_pv_scale_write");
                            ++result.report.scale_writes;
                        }
                    }
                }
                result.report.scale_s += seconds(start);
                start = clock::now();
                accept(potal_paired_loop_ws_issue(session, &request), "attention_pv_issue");
                result.report.issue_s += seconds(start);
                ++result.report.loop_ws;
                start = clock::now();
                accept(potal_paired_fence(session), "attention_pv_fence");
                result.report.fence_s += seconds(start);
                ++result.report.fences;
                start = clock::now();
                for (uint32_t block = 0; block < blocks; ++block) {
                    for (uint32_t j = 0; j < loop.j_tiles; ++j) {
                        for (uint32_t col = 0; col < valid_columns(j); ++col) {
                            accept(potal_paired_scale_release(session, block * loop.j_tiles + j, generation, col),
                                   "attention_pv_scale_release");
                            ++result.report.scale_releases;
                        }
                    }
                }
                result.report.scale_s += seconds(start);
            }
            // All digit passes are complete. One full-width INT32 MVIN adds
            // the already-scaled exact bias per output lane, then readback.
            auto start = clock::now();
            accept(potal_paired_acc_bias_issue(session, &current.bias), "attention_pv_bias");
            result.report.issue_s += seconds(start);
            start = clock::now();
            accept(potal_paired_fence(session), "attention_pv_fence");
            result.report.fence_s += seconds(start);
            ++result.report.fences;
            start = clock::now();
            accept(potal_paired_acc_readback_issue(session, &current.output), "attention_pv_readback");
            result.report.issue_s += seconds(start);
            start = clock::now();
            accept(potal_paired_fence(session), "attention_pv_fence");
            result.report.fence_s += seconds(start);
            ++result.report.fences;
        }
        result.status = potal_paired_session_close(session);
        session = nullptr;
        if (result.status == POTAL_OK) {
            result.ok = true;
            result.stage = nullptr;
        } else {
            result.stage = "attention_pv_close";
            result.fatal = true;
        }
    } catch (potal_status) {
        // Preserve the original status and stage. Cleanup never retries a
        // failed fence; a still-live DMA/Scale entry keeps accelerator ownership.
    } catch (const std::bad_alloc &) {
        result.status = POTAL_ERR_CAPACITY;
        result.stage = "attention_pv_allocation";
    } catch (const std::length_error &) {
        result.status = POTAL_ERR_CAPACITY;
        result.stage = "attention_pv_allocation";
    }
    if (session) {
        const potal_status cleanup = potal_paired_session_cleanup(session);
        const potal_paired_session_status remaining = potal_paired_session_query(session);
        result.fatal = remaining.outstanding_loops || remaining.outstanding_readbacks ||
                       remaining.outstanding_acc_updates || remaining.live_scale_slots ||
                       (cleanup != POTAL_OK && remaining.error_stage &&
                        std::strcmp(remaining.error_stage, "fence") == 0);
        const potal_status close = potal_paired_session_close(session);
        if (result.status == POTAL_OK) {
            result.status = cleanup != POTAL_OK ? cleanup : close;
        }
    }
    result.report.status = result.status;
    result.report.stage = result.stage;
    result.report.fatal = result.fatal;
    return result;
}

} // namespace ggml::potal::attention
