// Same fixture helpers and production graph entry point as the hardware smoke.
// Link support/e4c1-sequential.cpp and the backend adapter, not another ops.cpp.
int e4c1_original_smoke_main(int, char **);
#define main e4c1_original_smoke_main
#include "potal-paired-smoke.cpp"
#undef main

#include "matmul/stripe.hpp"
#include "support/e4c1-sequential.hpp"

#include <climits>
#include <stdexcept>
#include <utility>

namespace seq = potal_test::e4c1;

namespace ggml::potal {
potal_status e4c1_tile_select(size_t, size_t, size_t, uint32_t, potal_tile_plan *);
pipeline_report e4c1_run_pipeline(const ggml_tensor *, ggml_gemmini_args_t &, const hp1_weight &,
                                 const potal_tile_plan &, const pipeline_config &, stripe_owner &);
}

namespace {
enum class bench_mode { serial48, serial28, paired28, sequential28 };
bench_mode selected_mode = bench_mode::serial48;
seq::options sequential_options;
seq::statistics aggregate_statistics;
bool diagnostic_run = false;
bool capture_prepared = false;
bool small_gate = false;
uint64_t captured_hash = 0;
size_t observed_main_k = 0;
size_t sequential_calls = 0;
bool paired_geometry_ok = true;
bool sequential_statistics_ok = true;

void require_e4c1(bool value, const char * message) {
    if (!value) throw std::runtime_error(message);
}

const char * benchmark_mode_name(bench_mode mode) {
    switch (mode) {
        case bench_mode::serial48: return "serial_originalK48";
        case bench_mode::serial28: return "serial_forcedK28";
        case bench_mode::paired28: return "paired_wholeK28";
        case bench_mode::sequential28: return "sequentialK28";
    }
    return "invalid";
}

// E3's serialization, independently fed from this executable's preparations.
// Host ABI is intentionally part of this same-build fingerprint.
struct prepared_hash {
    uint64_t value = UINT64_C(14695981039346656037);
    void bytes(const void * data, size_t size) {
        const auto * p = static_cast<const unsigned char *>(data);
        for (size_t i = 0; i < size; ++i) {
            value ^= p[i];
            value *= UINT64_C(1099511628211);
        }
    }
    template<class T> void scalar(const T & x) { bytes(&x, sizeof(x)); }
    template<class T> void vector(const std::vector<T> & x) {
        scalar(x.size());
        bytes(x.data(), x.size() * sizeof(T));
    }
};

uint64_t fingerprint_prepared(const ggml_gemmini_args_t & args) {
    prepared_hash f;
    f.scalar(args.I); f.scalar(args.J); f.scalar(args.K);
    f.scalar(args.activation_rows_per_stripe);
    f.bytes(args.A.raw_data(), args.A.raw_size());
    const auto * meta = std::get_if<ggml::gemmini::quants::act::exsia::Meta>(&args.act_quant.storage());
    require_e4c1(meta != nullptr, "missing ExSIA metadata");
    f.vector(meta->theta);
    f.scalar(meta->rmd_packets.size());
    for (const auto & handle : meta->rmd_packets) {
        require_e4c1(bool(handle), "null sealed packet");
        const auto & p = *handle;
#define FIELD(x) f.scalar(p.x)
        FIELD(version); FIELD(digit_bits); FIELD(lane_capacity); FIELD(digit_storage);
        FIELD(stripe_id); FIELD(row_begin); FIELD(row_count); FIELD(logical_k); FIELD(logical_j);
        FIELD(j_padded); FIELD(block_size); FIELD(array_dim); FIELD(activation_value_count);
        FIELD(residual_event_count); FIELD(residual_min); FIELD(residual_max); FIELD(required_planes);
        FIELD(digit_nnz); FIELD(active_original_rows); FIELD(active_original_rows_valid);
        FIELD(residual_observations_valid); FIELD(total_output_values);
#undef FIELD
        f.vector(p.k_indices);
        f.vector(p.stacked_activation.signed_int8);
        f.vector(p.stacked_activation.signed_int16);
        f.scalar(p.blocks.size());
        for (const auto & b : p.blocks) {
#define FIELD(x) f.scalar(b.x)
            FIELD(block_id); FIELD(global_k_begin); FIELD(compact_k_count); FIELD(padded_k_count);
            FIELD(active_lane_mask); FIELD(active_lane_count); FIELD(lane_ids); FIELD(lane_k_masks);
            FIELD(k_index_offset); FIELD(activation_offset); FIELD(activation_byte_offset);
            FIELD(activation_byte_count); FIELD(output_value_offset); FIELD(rows_padded);
            FIELD(lane_stride_values);
#undef FIELD
            f.scalar(b.groups.size());
            for (const auto & g : b.groups) {
                f.vector(g.lane_positions); f.scalar(g.row_offsets); f.vector(g.row_ids);
                f.scalar(g.k_mask); f.scalar(g.padded_k_count); f.scalar(g.activation_offset);
                f.scalar(g.activation_byte_offset); f.scalar(g.activation_byte_count);
            }
        }
    }
    return f.value;
}

// Independent wide arithmetic, never the production SCU/reference/lowering.
__extension__ typedef __int128 wide_integer;
int32_t bounded_i32(wide_integer x) {
    if (x > INT32_MAX) return INT32_MAX;
    if (x < INT32_MIN) return INT32_MIN;
    return static_cast<int32_t>(x);
}
int32_t scaled_partial(int64_t dot, uint32_t carrier) {
    if (carrier == POTAL_CARRIER_ZERO || dot == 0) return 0;
    require_e4c1(carrier <= 32767, "invalid golden carrier");
    // Beyond 31, every nonzero signed integer saturates; avoid unbounded shifts.
    if (carrier > 31) return dot < 0 ? INT32_MIN : INT32_MAX;
    return bounded_i32(wide_integer(dot) * (wide_integer(1) << carrier));
}

struct raw_stripe {
    size_t row_begin = 0, rows = 0, cols = 0;
    std::vector<int32_t> main;
    struct lane_result {
        uint8_t lane;
        size_t row;
        std::vector<int32_t> values;
    };
    std::vector<lane_result> lanes;
    std::vector<int64_t> correction;
    bool main_seen = false, residual_seen = false, reconstruction_seen = false;
};
std::vector<raw_stripe> * active_raw = nullptr;
bool raw_ok = true;

std::vector<raw_stripe> packet_direct_golden(const ggml_gemmini_args_t & args, const hp1_operands & o) {
    namespace rmd = ggml::gemmini::rmd;
    const auto * meta = std::get_if<ggml::gemmini::quants::act::exsia::Meta>(&args.act_quant.storage());
    require_e4c1(meta != nullptr, "raw golden metadata");
    const size_t per = args.activation_rows_per_stripe;
    std::vector<raw_stripe> result((args.I + per - 1) / per);
    for (size_t s = 0; s < result.size(); ++s) {
        auto & g = result[s];
        g.row_begin = s * per;
        g.rows = std::min(per, size_t(args.I) - g.row_begin);
        g.cols = args.J;
        g.main.assign(g.rows * g.cols, 0);
        g.correction.assign(g.rows * g.cols, 0);
        for (size_t row = 0; row < g.rows; ++row) {
            for (size_t col = 0; col < g.cols; ++col) {
                int32_t acc = 0;
                for (size_t k = 0; k < size_t(args.K); k += 16) {
                    int64_t dot = 0;
                    for (size_t t = k; t < std::min(k + 16, size_t(args.K)); ++t)
                        dot += int64_t(o.a[(g.row_begin + row) * o.lda + t]) * o.b[t * o.ldb + col];
                    acc = bounded_i32(wide_integer(acc) +
                                      scaled_partial(dot, o.carriers[(k / 32) * o.cstride + col]));
                }
                g.main[row * g.cols + col] = acc;
            }
        }
    }
    for (const auto & handle : meta->rmd_packets) {
        require_e4c1(handle && handle->stripe_id < result.size(), "raw packet stripe");
        const auto & p = *handle;
        auto & g = result[p.stripe_id];
        require_e4c1(p.row_begin == g.row_begin && p.row_count == g.rows, "raw packet extent");
        std::vector<wide_integer> sums(g.rows * g.cols, 0);
        for (uint8_t lane = 0; lane < p.lane_capacity; ++lane) {
            for (size_t row = 0; row < g.rows; ++row) {
                std::vector<std::vector<int32_t>> digits;
                bool active = false;
                for (const auto & b : p.blocks) {
                    digits.emplace_back(b.compact_k_count, 0);
                    for (uint8_t pos = 0; pos < b.active_lane_count; ++pos) {
                        if (b.lane_ids[pos] != lane) continue;
                        for (size_t k = 0; k < b.compact_k_count; ++k) {
                            int32_t d = 0;
                            require_e4c1(rmd::read_packet_digit(p, b, pos, row, k, d) == rmd::RmdStatus::success,
                                         "packet-direct digit");
                            digits.back()[k] = d;
                            active = active || d != 0;
                        }
                    }
                }
                if (!active) continue;
                raw_stripe::lane_result lr{lane, row, std::vector<int32_t>(g.cols, 0)};
                for (size_t col = 0; col < g.cols; ++col) {
                    int32_t acc = 0;
                    for (size_t block = 0; block < p.blocks.size(); ++block) {
                        const auto & b = p.blocks[block];
                        for (size_t f = 0; f < b.compact_k_count; f += 16) {
                            int64_t dot = 0;
                            for (size_t k = f; k < std::min(f + 16, size_t(b.compact_k_count)); ++k) {
                                const size_t original = size_t(b.block_id) * 32 + p.k_indices[b.k_index_offset + k];
                                require_e4c1(original < size_t(args.K), "packet-direct original K");
                                dot += int64_t(digits[block][k]) * o.b[original * o.ldb + col];
                            }
                            acc = bounded_i32(wide_integer(acc) +
                                              scaled_partial(dot, o.carriers[b.block_id * o.cstride + col]));
                        }
                    }
                    lr.values[col] = acc;
                    wide_integer place = 1;
                    require_e4c1(lane < 8, "radix lane outside INT64 contract");
                    for (uint8_t l = 0; l < lane; ++l) place *= 256;
                    sums[row * g.cols + col] += wide_integer(acc) * place;
                }
                g.lanes.push_back(std::move(lr));
            }
        }
        for (size_t i = 0; i < sums.size(); ++i) {
            require_e4c1(sums[i] >= INT64_MIN && sums[i] <= INT64_MAX, "golden reconstruction overflow");
            g.correction[i] = static_cast<int64_t>(sums[i]);
        }
    }
    return result;
}

// Only owner-thread raw/reconstruction events are inspected. No producer data
// races, oracle work, profile clocks or trace observer in measured samples.
void inspect_raw(const stripe_event & ev) {
    if (!active_raw || ev.stripe_id == no_stripe) return;
    if (ev.kind != stripe_event_kind::main_end && ev.kind != stripe_event_kind::residual_end &&
        ev.kind != stripe_event_kind::paired_end && ev.kind != stripe_event_kind::radix_reconstruct_end) return;
    if (ev.stripe_id >= active_raw->size()) { raw_ok = false; return; }
    auto & g = (*active_raw)[ev.stripe_id];
    auto compare = [&](const int32_t * actual, size_t stride, size_t rows, bool main) {
        const size_t want_rows = main ? g.rows : g.lanes.size();
        if (rows != want_rows || ev.cols != g.cols || (rows && (!actual || stride < g.cols))) {
            raw_ok = false;
            return;
        }
        for (size_t row = 0; row < rows; ++row) {
            for (size_t col = 0; col < g.cols; ++col) {
                const int32_t expected = main ? g.main[row * g.cols + col] : g.lanes[row].values[col];
                if (actual[row * stride + col] != expected) {
                    if (raw_ok)
                        std::fprintf(stderr, "E4C1_RAW_MISMATCH stripe=%zu stage=%s row=%zu col=%zu got=%d want=%d\n",
                                     ev.stripe_id, main ? "Main" : "Residual", row, col,
                                     actual[row * stride + col], expected);
                    raw_ok = false;
                }
            }
        }
    };
    if (ev.kind == stripe_event_kind::main_end || ev.kind == stripe_event_kind::paired_end) {
        compare(ev.c, ev.ldc, ev.rows, true);
        g.main_seen = true;
    }
    if (ev.kind == stripe_event_kind::residual_end || ev.kind == stripe_event_kind::paired_end) {
        if (ev.kind == stripe_event_kind::paired_end) {
            compare(ev.residual_c, ev.residual_ldc, ev.residual_rows, false);
            if (!ev.paired || ev.paired->plan.residual_rows.size() != g.lanes.size()) raw_ok = false;
            else for (size_t i = 0; i < g.lanes.size(); ++i) {
                const auto & r = ev.paired->plan.residual_rows[i];
                if (r.lane != g.lanes[i].lane || r.source_row != g.lanes[i].row) raw_ok = false;
            }
        } else compare(ev.c, ev.ldc, ev.rows, false);
        g.residual_seen = true;
    }
    if (ev.kind == stripe_event_kind::radix_reconstruct_end) {
        if (!ev.correction || ev.rows != g.rows || ev.cols != g.cols) raw_ok = false;
        else for (size_t i = 0; i < g.correction.size(); ++i)
            if (ev.correction[i] != g.correction[i]) raw_ok = false;
        g.reconstruction_seen = true;
    }
}

// No new hardware thread. Real owner's safety/affinity/serialization and normal
// consumer fatal outcome stay in force. Only the paired callable is substituted.
class sequential_owner final : public stripe_owner {
    stripe_owner & real_;
  public:
    explicit sequential_owner(stripe_owner & real) : real_(real) {}
    bool start(stripe_consumer consume, std::string & why, potal_status & status) override {
        return real_.start([consume = std::move(consume)](const owner_runners & supplied) {
            owner_runners runners = supplied;
            if (selected_mode == bench_mode::sequential28) {
                runners.paired = [](const potal_paired_request & request, potal_paired_report & report) {
                    seq::statistics stats;
                    const auto status = seq::run(request, nullptr, sequential_options, report, stats);
                    ++sequential_calls;
                    // Return fatal reports before aggregation can allocate/throw.
                    // The normal consumer must see fatal and quarantine its DMA.
                    if (report.fatal) return status;
                    seq::add(aggregate_statistics, stats);
                    sequential_statistics_ok = sequential_statistics_ok && !stats.software_only &&
                        stats.peak_scale_slots <= 256 && stats.peak_index_entries <= 32 &&
                        stats.peak_b_residencies <= 2 && stats.b_loads == stats.b_retirements &&
                        stats.main_scale_writes == stats.main_scale_releases &&
                        stats.residual_scale_writes == stats.residual_scale_releases;
                    paired_geometry_ok = paired_geometry_ok && report.tile.main.tile_I == 1 &&
                        report.tile.main.tile_J == 9 && report.tile.main.tile_K == 28;
                    return status;
                };
            }
            return consume(runners);
        }, why, status);
    }
    void join() override { real_.join(); }
    bool pin_producer(bool one_cpu, std::string & why) override { return real_.pin_producer(one_cpu, why); }
    void unpin_producer() override { real_.unpin_producer(); }
};

std::vector<uint64_t> stable_counts(const route_run & r) {
    std::vector<uint64_t> out;
#define COUNT(x) out.push_back(r.after.x - r.before.x)
    COUNT(main_logical_gemms); COUNT(main_loop_ws); COUNT(main_fences);
    COUNT(residual_logical_gemms); COUNT(residual_loop_ws); COUNT(residual_fences);
    COUNT(paired_stripes); COUNT(paired_main_loop_ws); COUNT(paired_residual_loop_ws);
    COUNT(paired_index_configs); COUNT(paired_index_writes); COUNT(paired_readbacks);
    COUNT(paired_readback_tiles); COUNT(paired_fences); COUNT(paired_drains);
    COUNT(scale_writes); COUNT(scale_releases); COUNT(paired_fallback_stripes);
    COUNT(stripes_semantic_completed); COUNT(merges_completed);
#undef COUNT
    return out;
}

void print_counts(const char * name, bench_mode mode, const route_run & r) {
    std::printf("E4C1_COUNTS case=%s mode=%s",name,benchmark_mode_name(mode));
#define COUNT(x) std::printf(" " #x "=%llu", static_cast<unsigned long long>(r.after.x-r.before.x))
    COUNT(main_logical_gemms); COUNT(main_loop_ws); COUNT(main_fences);
    COUNT(residual_logical_gemms); COUNT(residual_loop_ws); COUNT(residual_fences);
    COUNT(paired_stripes); COUNT(paired_main_loop_ws); COUNT(paired_residual_loop_ws);
    COUNT(paired_index_configs); COUNT(paired_index_writes); COUNT(paired_readbacks);
    COUNT(paired_readback_tiles); COUNT(paired_fences); COUNT(paired_drains);
    COUNT(scale_writes); COUNT(scale_releases); COUNT(paired_fallback_stripes);
    COUNT(stripes_semantic_completed); COUNT(merges_completed);
    COUNT(weight_cache_hits); COUNT(weight_cache_misses); COUNT(weight_packs);
#undef COUNT
    std::printf("\n");
}

void print_statistics(const char * name, const seq::statistics & s) {
    std::printf("E4C1_STATS case=%s main_k_tiles=%u Scale_limit=256 Index_limit=32", name,
                sequential_options.main_k_tiles);
#define STAT(x) std::printf(" " #x "=%llu", static_cast<unsigned long long>(s.x))
    STAT(main_microtiles); STAT(residual_microtiles); STAT(main_partials); STAT(residual_partials);
    STAT(main_to_residual); STAT(residual_to_main); STAT(main_only_tail); STAT(residual_only_tail);
    STAT(b_loads); STAT(b_mvins); STAT(b_retirements); STAT(scale_programs); STAT(scale_slot_instances);
    STAT(main_scale_writes); STAT(residual_scale_writes);
    STAT(main_scale_releases); STAT(residual_scale_releases);
    STAT(scale_generations); STAT(generation_wraps); STAT(release_boundaries);
    STAT(index_configs); STAT(index_writes); STAT(fences); STAT(readbacks); STAT(mvouts);
    STAT(views); STAT(loops); STAT(commands); STAT(drains);
    STAT(peak_scale_slots); STAT(peak_index_entries); STAT(peak_b_residencies);
    STAT(skipped_zero_group_fragments);
#undef STAT
    std::printf(" software_only=%d extents=command_declared physical_traffic=unmeasured\n", s.software_only);
    if (sequential_options.profile) {
        static const char * names[] = {"planning", "prepare", "b_issue", "b_completion", "scale_program",
            "index_program", "main_issue", "residual_issue", "completion", "scale_release", "readback",
            "retirement", "close"};
        static_assert(sizeof(names) / sizeof(names[0]) == static_cast<size_t>(seq::scope::count),
                      "update scope names with scheduler contract");
        for (size_t i = 0; i < s.intervals.size(); ++i)
            std::printf("E4C1_CPU_INTERVAL case=%s scope=%s ns=%llu calls=%llu hardware_wait=unmeasured\n",
                        name, names[i], static_cast<unsigned long long>(s.intervals[i].ns),
                        static_cast<unsigned long long>(s.intervals[i].calls));
    }
    for (const auto & e : s.events)
        std::printf("E4C1_EVENT case=%s kind=%u pair=%u bank=%u base=%u pitch=%u acc_tile=%u "
                    "k_rows=%u columns=%u scale_base=%u scale_generation=%u original_k=%llu original_j=%llu "
                    "sequence=%llu group=%u indexed=%u first=%u last=%u extents=command_declared\n",
                    name, unsigned(e.kind), e.pair, e.bank, e.base, e.pitch, e.acc_tile, e.k_rows,
                    e.columns, e.scale_base, e.scale_generation,
                    static_cast<unsigned long long>(e.original_k), static_cast<unsigned long long>(e.original_j),
                    static_cast<unsigned long long>(e.sequence), e.group, e.indexed, e.first, e.last);
    // API does not expose an epoch-begin counter. Do not relabel loops as epochs.
    std::puts("E4C1_EPOCHS checked_by=resident_API immutable_epoch_bindings=1 epoch_counter=unavailable");
}

void probe_only() {
    issuer_owner owner;
    std::string why;
    potal_status start_status = POTAL_OK;
    bool matched = false;
    consume_outcome result;
    const bool started = owner.start([&](const owner_runners &) {
        potal_paired_session * session = nullptr;
        const auto opened = potal_paired_resident_open(nullptr, &session);
        if (opened != POTAL_OK) {
            result = {false, opened, "resident_open"};
            return result;
        }
        const auto q = potal_paired_resident_query(session);
        matched = q.capability_matched && !q.software_only;
        const auto closed = potal_paired_resident_close(&session);
        result = {session != nullptr, closed, "resident_close"};
        return result;
    }, why, start_status);
    require_e4c1(started, why.c_str());
    owner.join();
    require_e4c1(matched && !result.fatal && result.status == POTAL_OK, "resident signature/close refused");
    std::printf("E4C1_PROBE PASS signature=%016llx software_only=0 extension_commands=0\n",
                static_cast<unsigned long long>(POTAL_RESIDENT_WS_CAPABILITY));
}

void controlled_case(ggml_backend_t backend, const bench_case & c, int repeats, bool gate) {
    // Fresh engine per case; distinct smoke helper distributions, weights first.
    std::mt19937 rng(20261023);
    ggml_init_params ip{size_t(4 * (c.n*c.k + 2*c.m*c.n + c.m*c.k) + 64*1024*1024), nullptr, false};
    ggml_context_ptr context(ggml_init(ip));
    require_e4c1(bool(context), "context allocation");
    auto * w = hp1_weight_tensor(context.get(), c.k, c.n, rng);
    auto * x = activations(context.get(), c.k, c.m, rng, c.outlier, c.blocks, c.step);
    auto * expected = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, c.n, c.m);
    potal_tile_plan original, frozen;
    require_e4c1(potal_tile_select(c.m, c.n, c.k, 2, &original) == POTAL_OK, "original plan");
    if (small_gate)
        require_e4c1(potal_frozen_plan(c.m, c.n, c.k, 1, 9, 28, 2, &original) == POTAL_OK, "gate I1J9K28");
    require_e4c1(potal_frozen_plan(c.m, c.n, c.k, original.tile_I, original.tile_J,
                                  std::min(size_t(28), original.tile_K), 2, &frozen) == POTAL_OK, "frozen plan");
    if (!small_gate)
        require_e4c1(original.tile_I == 1 && original.tile_J == 9 && original.tile_K == 48 &&
                     frozen.tile_I == original.tile_I && frozen.tile_J == original.tile_J,
                     "target requires original I1J9K48 and retained I/J K28");
    ggml_gemmini_args_t args;
    hp1_operands operands;
    hp1::stats unpack_stats;
    require_e4c1(operands.allocate({c.m,c.n,c.k}) &&
        init_args(w, expected, {c.m,c.n,c.k}, "", "e4c1_prepare",
                  ggml::gemmini::residual::ResidualRoute::ws_packet, original, args) &&
        unpack_weights(args, operands, &unpack_stats) == hp1::status::ok &&
        quantize_activations(x, args, operands), "independent preparation");
    const uint64_t expected_hash = fingerprint_prepared(args);
    if (!std::strcmp(c.name, "decode_1x768x768"))
        require_e4c1(expected_hash == UINT64_C(0xe460e0fce470e311), "Decode E3 fingerprint differs");
    if (!std::strcmp(c.name, "gpt2_ffn_up_16x3072x768"))
        require_e4c1(expected_hash == UINT64_C(0x18e2b73e075c3bbe), "FFN E3 fingerprint differs");
    golden_result gold;
    std::vector<float> expected_values;
    gold.ok = potal_test::pipeline_golden(args, operands, expected_values, &gold.counts);
    require_e4c1(gold.ok, "final-output golden");
    gold.stripes = (c.m + original.tile_I*16 - 1) / (original.tile_I*16);
    std::memcpy(expected->data, expected_values.data(), expected_values.size()*sizeof(float));
    auto raw = diagnostic_run ? packet_direct_golden(args, operands) : std::vector<raw_stripe>{};
    prepared_hash weight_before, activation_before;
    weight_before.bytes(w->data, ggml_nbytes(w));
    activation_before.bytes(x->data, ggml_nbytes(x));
    ggml_init_params wip{2*ggml_tensor_overhead(), nullptr, true};
    ggml_context_ptr weight_context(ggml_init(wip));
    require_e4c1(bool(weight_context), "weight context allocation");
    auto * cached = ggml_new_tensor_2d(weight_context.get(), GGML_TYPE_Q8_HP1, c.k, c.n);
    auto * buft = ggml_backend_cpu_buffer_type();
    ggml_backend_buffer_ptr buffer(ggml_backend_buft_alloc_buffer(buft, ggml_backend_buft_get_alloc_size(buft,cached)));
    require_e4c1(buffer && ggml_backend_tensor_alloc(buffer.get(),cached,ggml_backend_buffer_get_base(buffer.get())) ==
                 GGML_STATUS_SUCCESS, "weight buffer");
    ggml_backend_buffer_set_usage(buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_tensor_set(cached, w->data, 0, ggml_nbytes(w));
    auto * y = ggml_mul_mat(context.get(), cached, x);
    auto * graph = ggml_new_graph(context.get());
    ggml_build_forward_expand(graph, y);
    std::array<std::vector<uint64_t>,4> counts;
    std::array<std::vector<double>,4> samples;
    std::array<std::vector<uint64_t>,4> scheduler_counts;
    long issuer_tid = -1;
    int issuer_cpu = -1;
    auto execute = [&](int mode, bool warmup) {
        selected_mode = static_cast<bench_mode>(mode);
        sequential_calls = 0;
        paired_geometry_ok = sequential_statistics_ok = true;
        aggregate_statistics = {};
        raw_ok = true;
        for (auto & r : raw) r.main_seen = r.residual_seen = r.reconstruction_seen = false;
        active_raw = diagnostic_run ? &raw : nullptr;
        capture_prepared = warmup;
        captured_hash = 0;
        const auto run = run_route(backend,c.name,mode >= 2 ? ROUTE_PAIRED : ROUTE_SERIAL,
                                   graph,y,expected,gold,false);
        capture_prepared = false;
        active_raw = nullptr; // owner joined before graph_compute returns
        require_e4c1(run.ok && !run.fallback, "route/output failed or paired fallback attempted");
        require_e4c1(run.after.hp1_completed-run.before.hp1_completed == 1 &&
                     run.after.stripe_pipeline_nodes-run.before.stripe_pipeline_nodes == 1 &&
                     run.after.route_completed[GGML_POTAL_ROUTE_EXSIA_BLOCK]-
                         run.before.route_completed[GGML_POTAL_ROUTE_EXSIA_BLOCK] == 1 &&
                     run.after.rejected_nodes == run.before.rejected_nodes &&
                     run.after.compensation_failed == run.before.compensation_failed,
                     "expected one production ExSIA pipeline MatMul");
        require_e4c1(run.after.issuer_ready && run.after.issuer_cpu >= 0 && run.after.issuer_tid > 0,
                     "real issuer not established");
        if (issuer_tid < 0) {
            issuer_tid = run.after.issuer_tid;
            issuer_cpu = run.after.issuer_cpu;
        }
        require_e4c1(run.after.issuer_tid == issuer_tid && run.after.issuer_cpu == issuer_cpu,
                     "issuer changed between modes");
        require_e4c1(observed_main_k == (small_gate || mode == 1 ? 28u : 48u), "Main tile selector not bound");
        require_e4c1(mode < 2 || run.paired == gold.stripes, "paired case ran Serial");
        require_e4c1(mode < 2 || paired_geometry_ok, "paired execution did not retain I1J9K28");
        require_e4c1(mode != 3 || (sequential_calls == gold.stripes &&
                     sequential_statistics_ok), "sequential scheduler/resource guard");
        require_e4c1(mode == 3 || sequential_calls == 0, "sequential runner leaked into baseline");
        if (diagnostic_run) {
            for (const auto & r : raw)
                raw_ok = raw_ok && r.main_seen &&
                    (r.lanes.empty() || (r.residual_seen && r.reconstruction_seen));
            require_e4c1(raw_ok, "raw Main/Residual/reconstruction mismatch");
        }
        if (warmup) require_e4c1(captured_hash == expected_hash, "pipeline quantization/packet fingerprint differs");
        else require_e4c1(run.after.weight_cache_hits-run.before.weight_cache_hits == 1 &&
                          run.after.weight_cache_misses == run.before.weight_cache_misses &&
                          run.after.weight_packs == run.before.weight_packs &&
                          run.after.weight_validations == run.before.weight_validations &&
                          run.after.weight_cache_entries == run.before.weight_cache_entries &&
                          run.after.weight_cache_bytes == run.before.weight_cache_bytes, "timed cache guard");
        if (warmup) counts[mode] = stable_counts(run);
        else require_e4c1(stable_counts(run) == counts[mode], "route counts changed");
        if (mode == 3) {
            const auto & s = aggregate_statistics;
            const std::vector<uint64_t> current{s.main_microtiles,s.residual_microtiles,s.main_partials,
                s.residual_partials,s.main_to_residual,s.residual_to_main,s.b_loads,s.b_mvins,
                s.b_retirements,s.scale_programs,s.index_configs,s.index_writes,s.readbacks,s.mvouts,
                s.loops,s.commands,s.drains,s.peak_scale_slots,s.peak_index_entries,s.peak_b_residencies};
            if (warmup) scheduler_counts[mode] = current;
            else require_e4c1(current == scheduler_counts[mode], "scheduler counts changed");
        }
        prepared_hash after_w, after_x;
        after_w.bytes(cached->data, ggml_nbytes(cached));
        after_x.bytes(x->data, ggml_nbytes(x));
        require_e4c1(after_w.value == weight_before.value && after_x.value == activation_before.value, "input mutation");
        return run;
    };
    std::printf("E4C1_FIXTURE case=%s seed=20261023 weights_hash=%016llx activations_hash=%016llx "
                "prepared_hash=%016llx I=%zu J=%zu originalK=%zu forcedK=%zu\n", c.name,
                static_cast<unsigned long long>(weight_before.value),
                static_cast<unsigned long long>(activation_before.value),
                static_cast<unsigned long long>(expected_hash),original.tile_I,original.tile_J,original.tile_K,frozen.tile_K);
    const int first = gate ? 3 : 0;
    for (int mode = first; mode < 4; ++mode) {
        const auto r = execute(mode,true);
        print_counts(c.name,static_cast<bench_mode>(mode),r);
        if (mode == 3) print_statistics(c.name,aggregate_statistics);
    }
    if (gate) {
        const uint64_t expected_loads = ((uint64_t(c.n)+143)/144)*((uint64_t(c.k)+447)/448);
        require_e4c1(aggregate_statistics.b_loads == expected_loads &&
                     aggregate_statistics.b_mvins > 0 && aggregate_statistics.main_to_residual > 0 &&
                     aggregate_statistics.index_configs > 0 && aggregate_statistics.scale_generations > 0,
                     "gate did not exercise resident sequential Main/Residual");
        std::printf("E4C1_GATE PASS case=%s raw_integer_exact=1 expected_b_loads=%llu hardware_timing=unmeasured\n",
                    c.name,static_cast<unsigned long long>(expected_loads));
    } else if (!diagnostic_run) {
        for (int iteration = 0; iteration < repeats; ++iteration) {
            for (int position = 0; position < 4; ++position) {
                const int mode = (iteration + (iteration%2 ? 3-position : position))%4;
                const auto r = execute(mode,false);
                samples[mode].push_back(r.compute_ms);
                std::printf("E4C1_SAMPLE case=%s mode=%s sample=%d cpu_compute_ms=%.6f cpu_pipeline_ms=%.6f "
                            "diagnostics=OFF hardware_wait=unmeasured\n",c.name,
                            benchmark_mode_name(static_cast<bench_mode>(mode)),iteration+1,r.compute_ms,r.wall_ms);
            }
        }
        for (int mode = 0; mode < 4; ++mode) {
            const double med = median(samples[mode]);
            std::vector<double> deviations;
            for (double value : samples[mode]) deviations.push_back(std::abs(value-med));
            std::printf("E4C1_SUMMARY case=%s mode=%s n=%zu cpu_median_ms=%.6f cpu_mad_ms=%.6f\n",
                        c.name,benchmark_mode_name(static_cast<bench_mode>(mode)),
                        samples[mode].size(),med,median(deviations));
        }
    } else {
        std::printf("E4C1_DIAGNOSTIC PASS case=%s modes=4 raw_integer_exact=1 performance_sample=0\n",c.name);
    }
    ggml_backend_potal_clear_weight_cache(backend);
}
} // namespace

namespace ggml::potal {
potal_status e4c1_tile_select(size_t m,size_t n,size_t k,uint32_t ppb,potal_tile_plan * plan) {
    const auto status = potal_tile_select(m,n,k,ppb,plan);
    if (status != POTAL_OK) return status;
    if (small_gate || selected_mode == bench_mode::serial28) {
        potal_tile_plan frozen;
        const auto checked = potal_frozen_plan(m,n,k,small_gate ? 1 : plan->tile_I,
            small_gate ? 9 : plan->tile_J,small_gate ? 28 : std::min(size_t(28),plan->tile_K),ppb,&frozen);
        if (checked != POTAL_OK) return checked;
        *plan = frozen;
    }
    observed_main_k = plan->tile_K;
    return POTAL_OK;
}

pipeline_report e4c1_run_pipeline(const ggml_tensor * x,ggml_gemmini_args_t & args,const hp1_weight & w,
                                 const potal_tile_plan & plan,const pipeline_config & config,stripe_owner & owner) {
    auto cfg = config;
    if (diagnostic_run || capture_prepared) {
        const auto observer = cfg.observer;
        cfg.observer = [observer](const stripe_event & event) {
            if (observer) observer(event);
            if (diagnostic_run) inspect_raw(event);
            if (event.kind == stripe_event_kind::paired_end && event.paired)
                paired_geometry_ok = paired_geometry_ok && event.paired->tile.main.tile_I == 1 &&
                    event.paired->tile.main.tile_J == 9 && event.paired->tile.main.tile_K == 28;
        };
    }
    sequential_owner proxy(owner);
    auto report = run_stripe_pipeline(x,args,w,plan,cfg,proxy);
    if (capture_prepared && report.ok) captured_hash = fingerprint_prepared(args);
    return report;
}
} // namespace ggml::potal

int main(int argc,char ** argv) {
    try {
        bool bench = false, gates = false, profile = false, trace = false, probe = false;
        const char * which = "all";
        int repeats = 7;
        for (int i = 1; i < argc; ++i) {
            const char * arg = argv[i];
            if (!std::strcmp(arg,"--bench")) bench = true;
            else if (!std::strcmp(arg,"--gates")) gates = true;
            else if (!std::strcmp(arg,"--profile")) profile = true;
            else if (!std::strcmp(arg,"--trace")) trace = true;
            else if (!std::strcmp(arg,"--probe-only")) probe = true;
            else if ((!std::strcmp(arg,"--case") || !std::strcmp(arg,"--repeat") || !std::strcmp(arg,"--group")) &&
                     i+1 < argc) {
                const char * value = argv[++i];
                if (!std::strcmp(arg,"--case")) which = value;
                else {
                    char * end = nullptr;
                    errno = 0;
                    const long n = std::strtol(value,&end,10);
                    require_e4c1(!errno && end != value && !*end && n > 0 && n <= 10000, "invalid integer option");
                    if (!std::strcmp(arg,"--repeat")) repeats = int(n);
                    else {
                        require_e4c1(n == 1 || n == 2 || n == 4, "--group requires 1, 2 or 4");
                        sequential_options.main_k_tiles = uint32_t(n);
                    }
                }
            } else throw std::runtime_error("usage: --bench --case decode|ffn|all --repeat 7 --group 1|2|4; "
                                             "separate --profile, --trace, --gates, or --probe-only");
        }
        require_e4c1(!std::strcmp(which,"decode") || !std::strcmp(which,"ffn") || !std::strcmp(which,"all"), "invalid case");
        require_e4c1(!(bench && (gates || profile || trace || probe)) &&
                     !(probe && (gates || profile || trace)), "measurement and diagnostics must be separate runs");
        require_e4c1(bench || gates || profile || trace || probe, "select --bench, --gates, --profile, --trace or --probe-only");
        require_e4c1(!bench || repeats >= 7, "measured target requires n >= 7");
#if LOG_DEBUG || LOG_INFO || LOG_TRACE || CYCLE_DETAIL
        require_e4c1(!bench, "measured build requires LOG_DEBUG/INFO/TRACE=0 and CYCLE_DETAIL=0");
#endif
#if !defined(__riscv)
        std::fprintf(stderr,"E4C1 REFUSED: native hardware CLI requires RISC-V; no emulation or CPU fallback.\n");
        return 2;
#endif
        // Refuse externally enabled diagnostics even for our diagnostic runs.
        // Their controlled flags below are the only benchmark inputs.
        for (const char * key : {"POTAL_PLANNING_PROFILE","POTAL_RAW_TRACE","POTAL_PAIRED_PROFILE",
             "POTAL_PAIRED_WINDOW_DIAG","POTAL_PAIRED_DIAG_INDEX_ENTRIES","POTAL_PAIRED_STAGE_DIAG",
             "POTAL_PAIRED_PROBE"})
            require_e4c1(std::getenv(key) == nullptr, key);
        if (probe) { probe_only(); return 0; }
        diagnostic_run = gates || profile || trace;
        sequential_options.poison_b = diagnostic_run;
        sequential_options.profile = profile;
        sequential_options.trace = trace || gates;
        if (profile) setenv("POTAL_PAIRED_PROFILE","1",1);
        if (trace) setenv("POTAL_RAW_TRACE","1",1);
        require_e4c1(select_pipeline_mode(), "stripe pipeline configuration");
        ggml_backend_ptr backend(ggml_backend_potal_init());
        require_e4c1(bool(backend), "backend initialization");
        if (gates) {
            constexpr bench_case progressive[] = {
                {"single_residency_16x144x448",16,144,448,400.0f,2,2},
                {"K28_K20_16x144x768",16,144,768,400.0f,2,2},
                {"J9_tail_16x155x768",16,155,768,400.0f,2,2}
            };
            small_gate = true;
            for (const auto & c : progressive) controlled_case(backend.get(),c,0,true);
            small_gate = false;
        }
        for (const auto & c : bench_cases) {
            const bool decode = !std::strcmp(c.name,"decode_1x768x768");
            const bool ffn = !std::strcmp(c.name,"gpt2_ffn_up_16x3072x768");
            if ((decode || ffn) && (gates || !std::strcmp(which,"all") ||
                (decode && !std::strcmp(which,"decode")) || (ffn && !std::strcmp(which,"ffn"))))
                controlled_case(backend.get(),c,repeats,gates);
        }
        std::printf("E4C1_%s PASS failures=%d raw_integer_verified=%d timing=CPU_observed "
                    "hardware_timing=unmeasured physical_traffic=unmeasured\n",
                    gates ? "GATES" : diagnostic_run ? "DIAGNOSTIC" : "BENCH",failures,diagnostic_run);
        return failures ? 1 : 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr,"E4C1 FAIL %s\n",error.what());
        return 1;
    }
}
