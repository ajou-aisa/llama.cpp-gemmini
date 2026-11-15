// H3-6E: unchanged paired-auto planning and checked C11 translation inventory.
// Build like potal-paired-execution-test with POTAL_BACKEND_SOURCES,
// potal_add_backend_settings, the tests include directory, and PRIVATE ggml.
// Run on the host with EXSIA and RMD enabled. No command reaches hardware:
// every checked session command is forwarded to support/paired-h2-emulator.hpp.
//
// Fixtures reproduce potal-paired-smoke.cpp bench_one, not its run_case sequence:
// reset mt19937(20261023) per case; generate and quantize weights before activations.
// All ranges are half-open. K tile counts use POTAL_HW_DIM elements per tile.
// "logical" footprints and "command_declared" read extents are not actual DMA
// transactions, bus traffic, elapsed time, concurrency, or evidence of hardware PASS.

#include "ggml-cpp.h"
#include "ggml-quants.h"
#include "matmul/dense.hpp"
#include "matmul/paired-auto.hpp"
#include "quants/act/exsia/types.hpp"
#include "support/paired-h2-emulator.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <tuple>
#include <variant>
#include <vector>

using namespace ggml::potal;

namespace {

constexpr size_t D = POTAL_HW_DIM;
namespace exsia = ggml::gemmini::quants::act::exsia;

struct bench_case {
    const char * name;
    size_t m, n, k, step;
};

struct free_dma {
    void operator()(void * p) const { std::free(p); }
};

std::ostream & record(const bench_case & c, size_t stripe, const char * kind) {
    return std::cout << "POTAL_INVENTORY case=" << c.name << " stripe=" << stripe << " record=" << kind;
}

bool failure(const bench_case & c, size_t stripe, const char * stage) {
    record(c, stripe, "failure") << " stage=" << stage << '\n';
    return false;
}

// Keep distributions separate, including their cached normal sample, exactly
// as hp1_weight_tensor() and activations() in the smoke benchmark.
void fill_tensors(const bench_case & c, ggml_tensor * w, ggml_tensor * x) {
    std::mt19937 rng(20261023);
    std::normal_distribution<float> weight_normal(0.0f, 1.0f);
    std::vector<float> f(c.n * c.k);
    for (size_t n = 0; n < c.n; ++n) {
        for (size_t k = 0; k < c.k; ++k) {
            const float mag[] = { 1.0f, 8.0f, 0.125f, 64.0f };
            f[n * c.k + k] = weight_normal(rng) * 0.02f * mag[(k / 32 + n) % 4];
        }
    }
    ggml_quantize_chunk(GGML_TYPE_Q8_HP1, f.data(), w->data, 0, c.n, c.k, nullptr);
    std::normal_distribution<float> act_normal(0.0f, 1.0f);
    auto * a = static_cast<float *>(x->data);
    for (size_t m = 0; m < c.m; ++m) {
        for (size_t k = 0; k < c.k; ++k) {
            a[m * c.k + k] = act_normal(rng) * (1.0f + 0.5f * (float) (m % 3));
        }
        if (m % c.step == 0) {
            for (size_t b = 0; b < 2; ++b) {
                a[m * c.k + (m * 7 + b * 37) % c.k] = 400.0f * ((m + b) % 4 ? -1.0f : 1.0f);
            }
        }
    }
}

struct inventory_totals {
    size_t stripes = 0, panels = 0, residencies = 0, windows = 0;
    size_t compact = 0, padded = 0, fillers = 0, index_writes = 0, mapping_errors = 0;
    size_t scale_slots = 0, scale_entries = 0, main_loops = 0, residual_loops = 0;
    uint64_t a_payload = 0, b_payload = 0, a_span_sum = 0, b_span_sum = 0;
    // Counts executed windows (including repeated J groups), not unique packets.
    std::map<std::tuple<size_t, size_t, size_t, size_t>, size_t> histogram;
};

void print_plan(const bench_case & c, size_t stripe, const potal_tile_plan & serial,
                const potal_paired_execution_plan & plan, const residual::decoded_packet * decoded,
                inventory_totals & totals) {
    const auto & t = plan.tile;
    record(c, stripe, "tile")
        << " units=DIM_tiles serial_I=" << serial.tile_I << " serial_J=" << serial.tile_J
        << " serial_K=" << serial.tile_K << " paired_I=" << t.tile_I << " paired_J=" << t.tile_J
        << " requested_main_K=" << t.tile_K_main << " hardware_only_K=" << t.tile_K_hardware_limit
        << " residual_limit_K=" << t.tile_K_residual_limit << " final_K=" << t.tile_K
        << " weight_limit_K=" << t.tile_K_weight_limit << " activation_limit_K=" << t.tile_K_activation_limit
        << " scale_limit_K=" << t.tile_K_scale_limit << " panels=" << t.k_panels
        << " j_groups=" << t.j_groups << " unique_windows=" << t.residual_windows
        << " lane_rows=" << t.residual_lane_rows << " residual_row_tiles=" << t.residual_row_tiles
        << " residual_compact_K=" << t.residual_k_compact << '\n';
    record(c, stripe, "footprint")
        << " basis=logical main_A_sp_rows=" << t.main_a_rows << " residual_A_sp_rows=" << t.residual_a_rows
        << " residual_A_offset_tiles=" << t.residual_a_offset_tiles << " B_sp_rows=" << t.weight_rows
        << " main_A_bytes=" << t.main_a_rows * D << " residual_A_bytes=" << t.residual_a_rows * D
        << " B_bytes=" << t.weight_rows * D << " main_ACC_rows=" << t.main_acc_rows
        << " residual_ACC_rows=" << t.residual_acc_rows << " main_ACC_bytes=" << t.main_acc_rows * D * sizeof(int32_t)
        << " residual_ACC_bytes=" << t.residual_acc_rows * D * sizeof(int32_t) << '\n';
    ++totals.stripes;
    totals.panels += t.panels.size();
    totals.residencies += plan.residencies.size();
    totals.main_loops += plan.main_gemms;
    totals.residual_loops += plan.residual_gemms;
    for (const auto & p : t.panels) {
        record(c, stripe, "panel") << " panel=" << p.index << " original_K_begin=" << p.k_begin
            << " original_K_end=" << p.k_begin + p.k_rows << " K_tiles=" << p.k_tiles
            << " compact_K=" << p.residual_k_compact << " windows=" << p.windows.size() << '\n';
    }
    for (const auto & r : plan.residencies) {
        const auto & op = plan.ops[r.main_gemm_op];
        const auto & p = t.panels[r.panel];
        const size_t j_begin = r.j_group * t.tile_J * D;
        record(c, stripe, "residency") << " residency=" << r.index << " pair=" << r.pair
            << " panel=" << r.panel << " j_group=" << r.j_group
            << " B_sp_begin=" << op.b_sp_begin << " B_sp_end=" << op.b_sp_begin + op.b_sp_rows
            << " original_K_begin=" << p.k_begin << " original_K_end=" << p.k_begin + p.k_rows
            << " J_begin=" << j_begin << " J_end=" << j_begin + r.cols << " J_tiles=" << r.j_tiles
            << " main_A_sp_begin=" << op.a_sp_begin << " main_A_sp_end=" << op.a_sp_begin + op.a_sp_rows
            << " ACC_epoch=" << op.acc_epoch << " consumers=" << r.consumers << " windows=" << r.windows << '\n';
    }
    for (const auto & op : plan.ops) {
        if (op.kind != potal_paired_op_kind::index_program) {
            continue;
        }
        const auto & p = t.panels[op.panel];
        const auto & w = p.windows[op.window];
        const auto & r = plan.residencies[op.residency];
        const auto & layout = potal_paired_pair(op.pair);
        size_t active = 0, errors = 0, original_min = SIZE_MAX, original_max = 0;
        size_t resident_min = SIZE_MAX, resident_max = 0;
        for (size_t i = op.entry_begin; i < op.entry_end; ++i) {
            const auto & en = plan.index_entries[i];
            original_min = std::min(original_min, en.original_k);
            original_max = std::max(original_max, en.original_k);
            resident_min = std::min(resident_min, en.resident_k);
            resident_max = std::max(resident_max, en.resident_k);
            errors += en.slot != i - op.entry_begin || en.original_k < p.k_begin ||
                      en.original_k >= p.k_begin + p.k_rows || en.resident_k != en.original_k - p.k_begin ||
                      en.resident_k >= op.index_source_k_rows;
            if (!en.filler) {
                ++active;
                const auto & fragment = t.fragments[en.fragment];
                errors += en.compact_k < fragment.compact_begin ||
                          en.compact_k >= fragment.compact_begin + fragment.count;
                if (decoded != nullptr && en.compact_k >= fragment.compact_begin &&
                    en.compact_k < fragment.compact_begin + fragment.count) {
                    errors += en.original_k != potal_paired_fragment_original_k(
                        *decoded, fragment, en.compact_k - fragment.compact_begin, potal_paired_hw_geometry());
                } else {
                    ++errors;
                }
            } else {
                errors += en.compact_k != potal_paired_no_index;
            }
        }
        const potal_paired_op * gemm = nullptr;
        for (const auto & candidate : plan.ops) {
            if (candidate.kind == potal_paired_op_kind::residual_gemm &&
                candidate.residency == op.residency && candidate.window == op.window) {
                gemm = &candidate;
                break;
            }
        }
        if (gemm == nullptr || op.entry_begin == op.entry_end) {
            ++totals.mapping_errors;
            continue;
        }
        const auto & scale = plan.scale_windows[gemm->scale_window];
        const auto & tail = plan.index_entries[op.entry_end - 1];
        const size_t padded = w.k_tiles * D;
        errors += active + w.filler_slots != w.active_rows;
        errors += w.active_rows + w.pad_k != padded;
        record(c, stripe, "window") << " basis=logical residency=" << op.residency
            << " panel=" << op.panel << " window=" << op.window << " pair=" << op.pair
            << " resident_K=" << p.k_rows << " active_compact_K=" << active
            << " indexed_slots=" << w.active_rows << " padded_K=" << padded
            << " filler_slots=" << w.filler_slots << " tail_pad_K=" << w.pad_k
            << " utilization=" << (double) active / padded
            << " I_tiles=" << gemm->i_tiles << " J_tiles=" << gemm->j_tiles
            << " lane_rows=" << t.residual_lane_rows << " row_tiles=" << t.residual_row_tiles
            << " original_K_min=" << original_min << " original_K_end=" << original_max + 1
            << " resident_K_min=" << resident_min << " resident_K_end=" << resident_max + 1
            << " tail_original_K=" << tail.original_k << " tail_resident_K=" << tail.resident_k
            << " tail_filler=" << tail.filler << " index_writes=" << op.entry_end - op.entry_begin
            << " scale_slots=" << scale.slots << " scale_entries=" << scale.blocks * r.cols
            << " scale_window=" << scale.index << " scale_generation=" << scale.generation
            << " dependent_residency=" << gemm->residency << " ACC_bank=" << gemm->acc_bank
            << " ACC_epoch=" << gemm->acc_epoch << " first=" << gemm->first << " last=" << gemm->last
            << " A_sp_begin=" << layout.activation_begin + gemm->a_offset_tiles * D
            << " A_sp_rows=" << gemm->a_sp_rows << " A_bytes=" << gemm->a_sp_rows * D
            << " ACC_rows=" << gemm->i_tiles * gemm->j_tiles * D
            << " ACC_bytes=" << gemm->i_tiles * gemm->j_tiles * D * D * sizeof(int32_t)
            << " mapping_errors=" << errors << '\n';
        for (const auto & dep : gemm->deps) {
            record(c, stripe, "window_dependency") << " residency=" << op.residency << " window=" << op.window
                << " consumer_op=" << gemm->id << " producer_op=" << dep.op
                << " requires_completion=" << dep.complete << '\n';
        }
        ++totals.windows;
        totals.compact += active;
        totals.padded += padded;
        totals.fillers += w.filler_slots;
        totals.index_writes += op.entry_end - op.entry_begin;
        totals.mapping_errors += errors;
        ++totals.histogram[{ active, padded, gemm->i_tiles, gemm->j_tiles }];
    }
    for (const auto & scale : plan.scale_windows) {
        totals.scale_slots += scale.slots;
        totals.scale_entries += scale.blocks * plan.residencies[scale.residency].cols;
    }
}

// Observe actual encoded fields at the checked C11 sink, then forward unchanged.
// A span includes stride gaps; payload excludes them. Neither is a bus measurement.
struct command_inventory {
    const bench_case & c;
    size_t stripe;
    const potal_paired_execution_plan & plan;
    const potal_paired_request & request;
    inventory_totals & totals;
    potal_test::h2_emulator emu;
    std::vector<const potal_paired_op *> loops, indices;
    size_t loop_cursor = 0, index_cursor = 0, written = 0;
    const potal_paired_op * current_index = nullptr;
    size_t resident[2] = { SIZE_MAX, SIZE_MAX };
    bool valid = true;

    command_inventory(const bench_case & c_, size_t s, const potal_paired_execution_plan & p,
                      const potal_paired_request & q, inventory_totals & t) :
        c(c_), stripe(s), plan(p), request(q), totals(t) {
        for (const auto & op : plan.ops) {
            if (op.kind == potal_paired_op_kind::main_gemm || op.kind == potal_paired_op_kind::residual_gemm) {
                loops.push_back(&op);
            } else if (op.kind == potal_paired_op_kind::index_program) {
                indices.push_back(&op);
            }
        }
    }

    static potal_status command(void * ctx, const potal_paired_cmd_record * raw) {
        auto & self = *static_cast<command_inventory *>(ctx);
        self.observe(*raw);
        return self.emu.command(*raw);
    }

    void observe(const potal_paired_cmd_record & raw) {
        if (raw.funct == POTAL_FUNCT_INDEX_CONFIG) {
            valid &= current_index == nullptr && index_cursor < indices.size();
            if (index_cursor >= indices.size()) {
                return;
            }
            current_index = indices[index_cursor++];
            written = 0;
            valid &= (raw.rs1 & 0xffffffffull) == current_index->index_b_base &&
                     (raw.rs1 >> 32) == current_index->index_j_tiles &&
                     (raw.rs2 & 0xffffffffull) == current_index->index_source_k_rows &&
                     (raw.rs2 >> 32) == current_index->index_active_rows;
        } else if (raw.funct == POTAL_FUNCT_INDEX_WRITE) {
            if (current_index == nullptr || written >= current_index->entry_end - current_index->entry_begin) {
                valid = false;
                return;
            }
            const auto & en = plan.index_entries[current_index->entry_begin + written++];
            valid &= raw.rs1 == en.slot && raw.rs2 == en.resident_k;
        } else if (raw.funct == POTAL_FUNCT_LOOP_WS) {
            if (loop_cursor >= loops.size()) {
                valid = false;
                return;
            }
            const auto & op = *loops[loop_cursor++];
            const bool residual = ((raw.rs1 >> 62) & 1) != 0;
            const bool indexed = ((raw.rs1 >> 63) & 1) != 0;
            const size_t a_id = (raw.rs1 >> 18) & 3, b_id = (raw.rs1 >> 16) & 3;
            const size_t I = emu.bounds_rs2 & 0xffff, J = (emu.bounds_rs2 >> 16) & 0xffff;
            const size_t K = (emu.bounds_rs2 >> 32) & 0xffff;
            const size_t pi = emu.bounds_rs1 & 0xffff, pj = (emu.bounds_rs1 >> 16) & 0xffff;
            const size_t pk = (emu.bounds_rs1 >> 32) & 0xffff;
            const bool geometry_ok = I == op.i_tiles && J == op.j_tiles && K == op.k_tiles &&
                                     pi == op.pad_i && pj == op.pad_j && pk == op.pad_k;
            valid &= geometry_ok && ((raw.rs1 >> 61) & 1) && a_id == op.pair + 1 && b_id == a_id &&
                     (raw.rs2 >> 3) == op.a_offset_tiles && ((raw.rs1 & 1) == !op.first) &&
                     residual == (op.kind == potal_paired_op_kind::residual_gemm) && indexed == residual;
            if (residual) {
                valid &= emu.b_dram == 0 && resident[op.pair] == op.residency &&
                         current_index != nullptr && current_index->residency == op.residency &&
                         current_index->window == op.window &&
                         written == current_index->entry_end - current_index->entry_begin;
                current_index = nullptr;
            } else {
                valid &= emu.b_dram != 0 && current_index == nullptr;
                const auto & panel = plan.tile.panels[op.panel];
                valid &= emu.a_dram == reinterpret_cast<uintptr_t>(request.a + panel.k_begin) &&
                         emu.b_dram == reinterpret_cast<uintptr_t>(request.weight->b +
                             panel.k_begin * request.weight->ldb + op.j_group * plan.tile.tile_J * D) &&
                         emu.a_stride == request.lda && emu.b_stride == request.weight->ldb;
                resident[op.pair] = op.residency;
            }
            if (!geometry_ok) {
                return; // Do not compute extents from malformed bounds.
            }
            const uint64_t rows = I * D - pi, cols = J * D - pj, k = K * D - pk;
            const uint64_t a_payload = rows * k, b_payload = emu.b_dram ? k * cols : 0;
            const uint64_t a_span = (rows - 1) * emu.a_stride + k;
            const uint64_t b_span = emu.b_dram ? (k - 1) * emu.b_stride + cols : 0;
            totals.a_payload += a_payload;
            totals.b_payload += b_payload;
            totals.a_span_sum += a_span;
            totals.b_span_sum += b_span;
            record(c, stripe, "loop") << " basis=command_declared path=" << (residual ? "residual" : "main")
                << " sequence=" << loop_cursor - 1 << " op=" << op.id << " residency=" << op.residency
                << " pair=" << op.pair << " A_spad_id=" << a_id << " B_spad_id=" << b_id
                << " indexed=" << indexed << " I_tiles=" << I << " J_tiles=" << J << " K_tiles=" << K
                << " pad_I=" << pi << " pad_J=" << pj << " pad_K=" << pk
                << " A_address=" << emu.a_dram << " B_address=" << emu.b_dram
                << " A_stride_bytes=" << emu.a_stride << " B_stride_bytes=" << emu.b_stride
                << " A_read_payload_bytes=" << a_payload << " B_read_payload_bytes=" << b_payload
                << " A_read_span_bytes=" << a_span << " B_read_span_bytes=" << b_span
                << " A_end_address=" << emu.a_dram + a_span
                << " B_end_address=" << (emu.b_dram ? emu.b_dram + b_span : 0)
                << " translation_checks_ok=" << valid << '\n';
        }
    }
};

bool run_case(const bench_case & c) {
    ggml_init_params ip = { 4 * (c.n * c.k + 2 * c.m * c.n + c.m * c.k) + 64 * 1024 * 1024, nullptr, false };
    ggml_context_ptr ctx(ggml_init(ip));
    if (!ctx) {
        return failure(c, 0, "context");
    }
    auto * w = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_Q8_HP1, c.k, c.n);
    auto * x = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, c.k, c.m);
    fill_tensors(c, w, x);
    auto * y = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, c.n, c.m);
    const hp1_shape shape = { (int64_t) c.m, (int64_t) c.n, (int64_t) c.k };
    potal_tile_plan serial;
    ggml_gemmini_args_t args;
    hp1_operands operands;
    quants::weight::hp1::stats weight_stats;
    if (potal_tile_select(c.m, c.n, c.k, 32 / D, &serial) != POTAL_OK ||
        !operands.allocate(shape) ||
        !init_args(w, y, shape, "", "golden", ggml::gemmini::residual::ResidualRoute::ws_packet, serial, args) ||
        unpack_weights(args, operands, &weight_stats) != quants::weight::hp1::status::ok ||
        !quantize_activations(x, args, operands)) {
        return failure(c, 0, "prepare");
    }
    const auto * meta = std::get_if<exsia::Meta>(&args.act_quant.storage());
    const size_t stripe_rows = serial.tile_I * D;
    const size_t stripes = (c.m + stripe_rows - 1) / stripe_rows;
    if (meta == nullptr || meta->theta.size() != stripes) {
        return failure(c, 0, "EXSIA_required");
    }
    std::map<size_t, const ggml::gemmini::rmd::StripePacket *> packets;
    for (const auto & packet : meta->rmd_packets) {
        if (!packet || packet->stripe_id >= stripes || !packets.emplace(packet->stripe_id, packet.get()).second) {
            return failure(c, 0, "packet_identity");
        }
    }
    if (packets.empty()) {
        return failure(c, 0, "RMD_required");
    }
    record(c, 0, "fixture") << " seed=20261023 M=" << c.m << " N=" << c.n << " K=" << c.k
        << " outlier=400 blocks=2 step=" << c.step << " DIM=" << D << " stripes=" << stripes
        << " packets=" << packets.size() << " ranges=half_open execution=H2_command_emulator\n";
    const auto hw = potal_paired_hw_geometry();
    record(c, 0, "geometry") << " DIM=" << hw.dim << " banks=" << hw.bank_num << " bank_sp_rows=" << hw.bank_rows
        << " ACC_rows=" << hw.acc_rows << " index_entries_per_pair=" << hw.index_entries
        << " scale_slots=" << hw.scale_entries << " work_ids=" << hw.work_ids << " block_K=" << hw.block_k << '\n';
    inventory_totals totals;
    for (size_t s = 0; s < stripes; ++s) {
        const size_t row_begin = s * stripe_rows, rows = std::min(stripe_rows, c.m - row_begin);
        residual::decoded_packet decoded;
        const residual::decoded_packet * decoded_ptr = nullptr;
        const auto packet = packets.find(s);
        if (packet != packets.end()) {
            if (residual::decode_packet(*packet->second, operands.own, decoded) !=
                ggml::gemmini::rmd::RmdStatus::success) {
                return failure(c, s, "decode");
            }
            decoded_ptr = &decoded;
        }
        potal_paired_tile_plan tile;
        potal_paired_execution_plan plan;
        std::string why;
        if (potal_paired_plan_stripe(serial, rows, c.n, c.k, decoded_ptr, tile, plan, &why) != potal_paired_status::ok) {
            std::cerr << c.name << ": " << why << '\n';
            return failure(c, s, "plan");
        }
        print_plan(c, s, serial, plan, decoded_ptr, totals);
        std::unique_ptr<int32_t, free_dma> residual_c(static_cast<int32_t *>(
            dma_alloc(align_up(std::max<size_t>(decoded.rows.size(), 1), D) * operands.ldc * sizeof(int32_t))));
        if (!residual_c) {
            return failure(c, s, "residual_C");
        }
        uint32_t generation = 0;
        potal_paired_request request;
        request.stripe_id = s;
        request.row_begin = row_begin;
        request.rows = rows;
        request.n = c.n;
        request.k = c.k;
        request.main_plan = &serial;
        request.a = operands.a + row_begin * operands.lda;
        request.lda = operands.lda;
        request.weight = &operands.own;
        request.residual = decoded_ptr;
        request.c_main = operands.c + row_begin * operands.ldc;
        request.ldc_main = operands.ldc;
        request.c_residual = residual_c.get();
        request.ldc_residual = operands.ldc;
        request.generation = &generation;
        request.plan = &plan;
        command_inventory commands(c, s, plan, request, totals);
        potal_paired_sink sink = { &command_inventory::command, &commands };
        potal_paired_session * session = nullptr;
        if (potal_paired_session_open(&sink, &session) != POTAL_OK) {
            return failure(c, s, "session_open");
        }
        potal_paired_report report;
        const auto status = potal_paired_matmul_auto(request, session, report);
        const auto state = potal_paired_session_query(session);
        const auto closed = potal_paired_session_close(session);
        const auto & emu = commands.emu;
        bool scales_clear = true;
        for (const auto & slot : emu.scales) {
            for (const auto & column : slot) {
                scales_clear &= !column.valid;
            }
        }
        const bool safe = !state.failed && state.outstanding_loops == 0 && state.outstanding_readbacks == 0 &&
                          state.index_transaction_open == 0 && state.live_scale_slots == 0 && closed == POTAL_OK;
        const bool counts = emu.main_loops == plan.main_gemms && emu.residual_loops == plan.residual_gemms &&
                            emu.b_loads == plan.main_gemms && emu.index_configs == plan.index_programs &&
                            emu.index_writes == plan.index_writes && emu.main_loops == report.counts.main_loop_ws &&
                            emu.residual_loops == report.counts.residual_loop_ws &&
                            emu.index_writes == report.counts.index_writes &&
                            emu.scale_writes == report.counts.scale_writes && emu.scale_releases == emu.scale_writes;
        const bool ok = status == potal_paired_status::ok && !report.fatal && safe && counts && scales_clear &&
                        commands.valid && commands.loop_cursor == commands.loops.size() &&
                        commands.index_cursor == commands.indices.size() && commands.current_index == nullptr &&
                        emu.errors.empty() && !emu.scale_miss && totals.mapping_errors == 0;
        record(c, s, "execution") << " model=H2_command_emulator status=" << potal_paired_status_name(status)
            << " main_LOOP_WS=" << emu.main_loops << " indexed_residual_LOOP_WS=" << emu.residual_loops
            << " B_loads=" << emu.b_loads << " index_configs=" << emu.index_configs
            << " index_writes=" << emu.index_writes << " scale_writes=" << emu.scale_writes
            << " scale_releases=" << emu.scale_releases << " readback_tiles=" << emu.mvouts
            << " fences=" << emu.fences << " outstanding_loops=" << state.outstanding_loops
            << " outstanding_readbacks=" << state.outstanding_readbacks << " live_scale_slots=" << state.live_scale_slots
            << " index_transaction_open=" << state.index_transaction_open << " failed=" << state.failed
            << " close_status=" << potal_status_name(closed) << " safe_completion=" << safe
            << " translation_checks_ok=" << commands.valid << " count_checks_ok=" << counts
            << " emulator_errors=" << !emu.errors.empty() << " scale_miss=" << emu.scale_miss
            << " diagnostic_ok=" << ok << '\n';
        if (!ok) {
            std::cerr << c.name << ": stage=" << (report.stage ? report.stage : "-")
                      << " detail=" << report.detail << " emulator=" << emu.errors << '\n';
            if (report.fatal) {
                residual_c.release();
                operands.abandon();
            }
            return false;
        }
    }
    for (const auto & bucket : totals.histogram) {
        record(c, 0, "window_histogram") << " scope=case_executed_windows active_compact_K=" << std::get<0>(bucket.first)
            << " padded_K=" << std::get<1>(bucket.first) << " I_tiles=" << std::get<2>(bucket.first)
            << " J_tiles=" << std::get<3>(bucket.first) << " count=" << bucket.second << '\n';
    }
    record(c, 0, "totals") << " scope=case stripes=" << totals.stripes << " panels=" << totals.panels
        << " residencies=" << totals.residencies << " executed_windows=" << totals.windows
        << " main_LOOP_WS=" << totals.main_loops << " residual_LOOP_WS=" << totals.residual_loops
        << " logical_compact_K_sum=" << totals.compact << " logical_padded_K_sum=" << totals.padded
        << " logical_utilization=" << (totals.padded ? (double) totals.compact / totals.padded : 0)
        << " logical_filler_slots=" << totals.fillers << " logical_index_writes=" << totals.index_writes
        << " logical_scale_slots_programmed=" << totals.scale_slots
        << " logical_scale_entries_programmed=" << totals.scale_entries << " mapping_errors=" << totals.mapping_errors
        << " command_declared_A_read_payload_bytes=" << totals.a_payload
        << " command_declared_B_read_payload_bytes=" << totals.b_payload
        << " command_declared_A_read_span_sum_bytes=" << totals.a_span_sum
        << " command_declared_B_read_span_sum_bytes=" << totals.b_span_sum << " diagnostic_ok=1\n";
    return true;
}

} // namespace

int main() {
    ggml_time_init();
    // Fail closed instead of letting optional per-window snapshots change the
    // command stream being inventoried. The caller can rerun without that mode.
    const char * window_diag = std::getenv("POTAL_PAIRED_WINDOW_DIAG");
    if (window_diag != nullptr && std::string(window_diag) == "1") {
        std::cerr << "Unset POTAL_PAIRED_WINDOW_DIAG: inventory requires the unchanged command stream.\n";
        return 1;
    }
    const bench_case cases[] = {
        { "decode_1x768x768", 1, 768, 768, 1 },
        { "gpt2_ffn_up_16x3072x768", 16, 3072, 768, 2 },
    };
    bool ok = true;
    for (const auto & c : cases) {
        ok = run_case(c) && ok;
    }
    return ok ? 0 : 1;
}
