// Included by the ExSIA behavioral test owner inside its anonymous namespace.
bool test_direct_stripe_rn() {
    using namespace quants::act::exsia;
    RestorePolicy policy;
    const int original_mode = std::fegetround();
    struct RestoreMode { int mode; ~RestoreMode() { std::fesetround(mode); } } restore{original_mode};
    const float factor = std::ldexp(1.0f, 4 - GGML_GEMMINI_ACTIVATION_BITS);
    std::vector<std::vector<float>> fixtures;
    for (float sign : {1.0f, -1.0f}) {
        std::vector<float> coarse(128, 4.0f);
        coarse[0] = 8; coarse[1] = sign * 1.25f * factor;
        std::fill(coarse.begin() + 32, coarse.begin() + 64, 32);
        std::fill(coarse.begin() + 64, coarse.begin() + 96, 16);
        std::fill(coarse.begin() + 96, coarse.end(), 8);
        fixtures.push_back(coarse);
        std::vector<float> fine(96, 4.0f);
        fine[0] = 8; fine[1] = sign * 0.625f * factor;
        std::fill(fine.begin() + 32, fine.begin() + 64, 8);
        std::fill(fine.begin() + 64, fine.end(), 2);
        fixtures.push_back(fine);
    }
    fixtures.emplace_back(32, GGML_GEMMINI_ACTIVATION_BITS == 4 ? 1.9f : 1.999f);
    fixtures.emplace_back(33, 0.0f);
    fixtures.push_back({0, -0.0f, INFINITY, -INFINITY, NAN,
                        std::numeric_limits<float>::denorm_min(),
                        -std::numeric_limits<float>::denorm_min()});
    fixtures.push_back({256, 8, 1, 1, 1, 1, 1}); // sigma changes the final block exponent
    fixtures.push_back({256, 2, 2, 2, 2, 2, 3}); // same-scale sigma branch
    fixtures.push_back({1, 1, 1, 1, 1, 1, 1, 1}); // one bucket
    fixtures.push_back({2147483648.0f, -2147483648.0f, 1, 1, 1, 1, 1, 1,
                        std::numeric_limits<float>::max(), -std::numeric_limits<float>::max()});
    std::vector<float> ties(128, std::ldexp(1.0f, GGML_GEMMINI_ACTIVATION_BITS - 2));
    const float tie_values[] = {0.5f, -0.5f, 1.5f, -1.5f, 2.5f, -2.5f,
                               std::nextafter(0.5f, 0.0f), std::nextafter(0.5f, 1.0f),
                               std::nextafter(-0.5f, 0.0f), std::nextafter(-0.5f, -1.0f)};
    std::copy(std::begin(tie_values), std::end(tie_values), ties.begin() + 1);
    ties[0] *= 2;
    std::fill(ties.begin() + 32, ties.begin() + 64, ties[0] * 2);
    std::fill(ties.begin() + 64, ties.begin() + 96, ties[0]);
    fixtures.push_back(ties); // final theta zero, both signed ties and adjacent F32 values
    std::vector<float> two_buckets(64, 1);
    std::fill(two_buckets.begin(), two_buckets.begin() + 32, 2);
    fixtures.push_back(two_buckets);
    std::vector<float> repeated(128, 16);
    std::fill(repeated.begin(), repeated.begin() + 32, 32);
    std::fill(repeated.begin() + 96, repeated.end(), 8);
    fixtures.push_back(repeated);
    std::vector<float> intermediate_clip(128, std::ldexp(1.0f, -28));
    intermediate_clip[0] = 4;
    std::fill(intermediate_clip.begin() + 32, intermediate_clip.begin() + 64, 4);
    std::fill(intermediate_clip.begin() + 64, intermediate_clip.begin() + 96, 2);
    std::fill(intermediate_clip.begin() + 96, intermediate_clip.end(), 1);
    fixtures.push_back(intermediate_clip); // block signed32 loss must not reach the final stripe
    const size_t top2_witness = fixtures.size();
    for (float sign : {1.0f, -1.0f}) {
        std::vector<float> coarse(96, 4.0f);
        coarse[0] = 8; coarse[1] = sign * 2.5f * factor;
        std::fill(coarse.begin() + 32, coarse.begin() + 64, 32);
        std::fill(coarse.begin() + 64, coarse.end(), 16);
        fixtures.push_back(coarse);
    }
    // Tail columns and a tail stripe use different values at each logical row.
    // These sizes also force slot reuse and catch reads with a padded source stride.
    std::vector<float> tail((2 * DIM + 1) * 99);
    for (size_t row = 0; row < 2 * DIM + 1; ++row) {
        const auto & base = fixtures[row % 4];
        for (size_t col = 0; col < 99; ++col)
            tail[row * 99 + col] = base[col % base.size()];
    }
    fixtures.push_back(tail);
    size_t runs = 0, distinctions[2] = {};
    for (int mode : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
        if (std::fesetround(mode) != 0) return false;
        for (int anchor : {2, 3}) {
        // Unset is tested as well as explicit 1, and every run re-reads the policy.
        for (const char * arm : {"0", "1", static_cast<const char *>(nullptr)}) {
        if (!policy.set(arm)) return false;
        const bool direct = arm == nullptr || std::strcmp(arm, "1") == 0;
        for (size_t f = 0; f < fixtures.size(); ++f) {
            const auto & source = fixtures[f];
            const size_t rows = f + 1 == fixtures.size() ? 2 * DIM + 1 : 1;
            const size_t cols = source.size() / rows;
            const auto expected = original_stripe_oracle(source, rows, cols, anchor, direct);
            if ((anchor == 3 && f < 4) ||
                (anchor == 2 && (f == top2_witness || f == top2_witness + 1))) {
                const int sign = anchor == 3 ? (f < 2 ? 1 : -1)
                                            : (f == top2_witness ? 1 : -1);
                // Coarse fixtures round to zero through Local; fine fixtures
                // round to +/-2. Original-FP final RN is always +/-1.
                const int old = anchor == 3 && f % 2 == 1 ? 2 * sign : 0;
                if (!check(expected.dense[1] == (direct ? sign : old),
                           "reachable fixture distinguishes literal old/direct codes"))
                    return false;
                ++distinctions[anchor - 2];
            }
            // Prove analysis Local keeps only preliminary worker integers, including
            // forced recomputation, while publishing the unchanged masks/exponents.
            LocalStage local;
            local.set_force_recompute(true);
            ExSIAState local_state;
            local_state.B_size = 32;
            StripeScratch scratch;
            if (!scratch.prepare(32)) return false;
            for (size_t row = 0; row < rows; ++row)
                for (size_t b = 0; b < (cols + 31) / 32; ++b) {
                    uint64_t word = 0;
                    BlockMask mask(&word, 32);
                    int16_t exp = 0;
                    Meta meta;
#if EXSIA_BRANCH_COUNTS_ENABLED
                    LocalBlockCycleSample sample;
#endif
                    if (!local.run_optimized(meta, local_state, source.data() + row * cols + b * 32,
                                             std::min<size_t>(32, cols - b * 32), 32, row, b,
                                             scratch, mask, nullptr, exp
#if EXSIA_BRANCH_COUNTS_ENABLED
                                             , sample
#endif
                                             ) ||
                        !check(exp == expected.block_exp[row * ((cols + 31) / 32) + b],
                               "analysis Local publishes independent block exponent"))
                        return false;
                    for (size_t i = 0; i < std::min<size_t>(32, cols - b * 32); ++i) {
                        const size_t p = row * cols + b * 32 + i;
                        if (!check(mask.is_set(i) == expected.local_selected[p] &&
                                   scratch.sigma_q[i] == expected.preliminary[p],
                                   "analysis preserves independent local mask and preliminary integers"))
                            return false;
                    }
#if EXSIA_BRANCH_COUNTS_ENABLED
                    if (!check(sample.replay_overwrite_count == 0,
                               "analysis Local performs no final replay")) return false;
#if EXSIA_STAGE_PROFILE_ENABLED
                    if (!check(sample.forced_recompute_count == 0,
                               "analysis Local performs no forced replay")) return false;
#endif
#endif
#if GGML_GEMMINI_ACT_QUANT_METRICS
                    if (!check(!scratch.actual_requantized, "analysis Local records no replay")) return false;
#endif
                }
            for (const auto & execution : execution_mode_availability()) {
                if (!execution.available) continue;
                for (auto route : {residual::ResidualRoute::ws_packet, residual::ResidualRoute::cpu_direct}) {
                    ggml_tensor tensor{};
                    tensor.type = GGML_TYPE_F32;
                    tensor.data = const_cast<float *>(source.data());
                    ggml_gemmini_args_t args{};
                    args.I = rows; args.J = 3; args.K = cols; args.sA = cols;
                    args.tile_I = args.tile_J = args.tile_K = 1;
                    args.activation_rows_per_stripe = DIM;
                    args.folding_anchor = anchor;
                    args.residual_route = route;
                    if (!args.A.allocate(rows, cols, GGML_GEMMINI_ACTIVATION_BITS)) return false;
                    ExSIA exsia;
                    Meta meta;
                    exsia.set_execution_mode(execution.mode);
                    if (!check(exsia.run(meta, &tensor, args) && meta.theta == expected.theta &&
                               exsia.ownership_ready(), "actual direct run publishes independent theta"))
                        return false;
                    std::vector<int64_t> plane(rows * cols);
                    for (const auto & packet : meta.rmd_packets) {
                        if (rmd::validate_packet(*packet) != rmd::RmdStatus::success) return false;
                        for (const auto & block : packet->blocks)
                            for (size_t k = 0; k < block.compact_k_count; ++k)
                                for (size_t lane = 0; lane < block.active_lane_count; ++lane)
                                    for (size_t row = 0; row < packet->row_count; ++row) {
                                        int32_t digit = 0;
                                        if (rmd::read_packet_digit(*packet, block, lane, row, k, digit) !=
                                            rmd::RmdStatus::success) return false;
                                        const size_t col = block.global_k_begin +
                                            packet->k_indices[block.k_index_offset + k];
                                        if (!check(col < cols, "packet never publishes padded columns")) return false;
                                        plane[(packet->row_begin + row) * cols + col] +=
                                            int64_t(digit) * (int64_t{1} << (packet->digit_bits * block.lane_ids[lane]));
                                    }
                    }
                    for (const auto & payload : meta.direct_residuals)
                        for (const auto & event : payload->events)
                            plane[(payload->row_begin + event.local_row) * cols + event.original_k] += event.residual;
                    for (size_t row = 0; row < rows; ++row)
                        for (size_t col = 0; col < cols; ++col) {
                            const size_t p = row * cols + col;
#if EXSIA_VALIDATION
                            if (!check(exsia.state().stripe[row / DIM].outlier_mask.is_set(row % DIM, col) ==
                                       expected.selected[p],
                                       "published selection includes promotion and selected zero residuals"))
                                return false;
#endif
                            if (args.A.get(row, col) != expected.dense[p] || plane[p] != expected.residual[p]) {
                                std::fprintf(stderr, "STRIPE anchor=%d policy=%s fixture=%zu mode=%s fenv=%d row=%zu col=%zu got=%d/%lld expected=%d/%d\n",
                                    anchor, arm ? arm : "unset", f, execution.label, mode, row, col, args.A.get(row, col),
                                    (long long)plane[p], expected.dense[p], expected.residual[p]);
                                return false;
                            }
                        }
                    args.act_quant.storage().emplace<Meta>(meta);
                    std::vector<float> decoded(rows * cols);
                    if (!dequantize_activation(decoded.data(), cols, 1, rows, cols, args)) return false;
                    for (size_t p = 0; p < decoded.size(); ++p)
                        if (!check(decoded[p] == std::ldexp(float(int64_t(expected.dense[p]) + expected.residual[p]),
                                                            expected.theta[(p / cols) / DIM]),
                                   "actual consumer preserves retained integer and final clipping")) return false;
                    const auto dense_bytes = *args.A.bytes;
                    if (!check(quants::quantize_activation(&tensor, args) &&
                               *args.A.bytes == dense_bytes,
                               "public F32 activation dispatch reaches direct dense codes"))
                        return false;
                    std::vector<float> dispatched(rows * cols);
                    if (!check(quants::dequantize_activation(dispatched.data(), cols, 1, rows, cols, args) &&
                               dispatched == decoded,
                               "public consumer retains direct stripe residual result"))
                        return false;
                    if (!check(std::fegetround() == mode, "direct path preserves caller fenv")) return false;
#if !EXSIA_VALIDATION
                    if (!check(exsia.state().residual.capacity() == 0, "production retains no residual plane"))
                        return false;
#endif
                    ++runs;
                }
            }
        }
        }
        }
    }
    if (!test_direct_stripe_policy_boundary()) return false;
    StripePipelineSlot slot;
    if (!slot.prepare(100000, 4, 1, 128, 32, true) ||
        !slot.prepare(100000, 4, 1, 128, 32, false) ||
        !check(slot.q_wide.empty() && slot.q_wide.capacity() == 0,
               "direct slot allocates no wide output")) return false;
    slot.acquire(0);
    slot.reset_for_stripe(0, 0, 1, 128, 4);
    if (!check(slot.q_wide.capacity() == 0, "direct slot reset never materializes wide output")) return false;
    SigmaDetector sigma;
    if (!check(!sigma.detect(5, sigma.prepare(5, 25, 5)) &&
               sigma.detect(6, sigma.prepare(6, 36, 6)),
               "sigma equality is strict and a larger deviation selects")) return false;
    std::printf("DIRECT_STRIPE_RN PASS bits=%d validation=%d runs=%zu anchors=2,3 policy=0,1,unset distinctions=%zu,%zu original_stride=logical direct_replay=0 direct_wide_capacity=0\n",
                GGML_GEMMINI_ACTIVATION_BITS, EXSIA_VALIDATION, runs,
                distinctions[0], distinctions[1]);
    return true;
}
