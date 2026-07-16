#include "../ggml/src/ggml-gemmini/ggml-gemmini-args.h"
#include "../ggml/src/ggml-gemmini/quants/act/dispatch.hpp"
#include "../ggml/src/ggml-gemmini/quants/act/exsia/exsia.hpp"
#include "../ggml/src/ggml-gemmini/quants/act/exsia/exsia_shift.hpp"
#include "../ggml/src/ggml-gemmini/quants/act/quantize.hpp"
#include "../ggml/src/ggml-gemmini/quants/common/tensor_util.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#ifndef CYCLE_DETAIL
#define CYCLE_DETAIL 0
#endif

using namespace ggml::gemmini::quants::act::exsia;

namespace {

bool checked_mul_size(size_t lhs, size_t rhs, size_t &out) {
    if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs) {
        return false;
    }

    out = lhs * rhs;
    return true;
}

bool checked_add_size(size_t lhs, size_t rhs, size_t &out) {
    if (lhs > std::numeric_limits<size_t>::max() - rhs) {
        return false;
    }

    out = lhs + rhs;
    return true;
}

bool checked_round_up_multiple(size_t value, size_t multiple, size_t &out) {
    if (multiple == 0) {
        return false;
    }

    size_t rounded_input = 0;
    if (!checked_add_size(value, multiple - 1, rounded_input)) {
        return false;
    }

    out = (rounded_input / multiple) * multiple;
    return true;
}

std::string json_escape(const std::string &value) {
    std::string escaped;
    escaped.reserve(value.size());
    for (const char character : value) {
        switch (character) {
            case '\\': escaped += "\\\\"; break;
            case '"': escaped += "\\\""; break;
            case '\n': escaped += "\\n"; break;
            case '\r': escaped += "\\r"; break;
            case '\t': escaped += "\\t"; break;
            default: escaped += character; break;
        }
    }
    return escaped;
}

struct Coord {
    size_t row = 0;
    size_t col = 0;
};

struct CycleLabelRecord {
    bool present = false;
    uint64_t start = 0;
    uint64_t end = 0;
    uint64_t delta = 0;
};

struct TimingObservation {
    std::string status = "not-run";
    std::string cycle_log_path;
    size_t record_count = 0;
    CycleLabelRecord local_stage;
    CycleLabelRecord stripe_folding;
    CycleLabelRecord end_to_end;
    uint64_t stage_sum = 0;
    uint64_t non_stage_overhead = 0;
};

struct SelectionComparisonResult {
    bool ok = false;
    std::string criterion = "one-sided upper-tail magnitude";
    int tau = 0;
    int rho = 0;
    size_t rows = 0;
    size_t cols = 0;
    size_t mismatch_count = 0;
    std::vector<float> input_fixture;
    std::vector<uint8_t> top1_mask;
    std::vector<uint8_t> float_mask;
    std::vector<uint8_t> integer_mask;
    std::vector<Coord> float_selected;
    std::vector<Coord> integer_selected;
    std::vector<Coord> float_only;
    std::vector<Coord> integer_only;
};

#ifdef __riscv
constexpr bool cycle_reader_supported = true;
#else
constexpr bool cycle_reader_supported = false;
#endif

std::string resolve_cycle_log_path() {
    const char *path = std::getenv("GGML_GEMMINI_CYCLE_DETAIL_LOG");
    return path && path[0] ? std::string(path) : std::string("log/exsia-cycle-detail.jsonl");
}

bool extract_json_string(const std::string &line, const char *key, std::string &value) {
    const std::string needle = std::string("\"") + key + "\":\"";
    const size_t start = line.find(needle);
    if (start == std::string::npos) {
        return false;
    }

    const size_t value_start = start + needle.size();
    const size_t value_end = line.find('"', value_start);
    if (value_end == std::string::npos) {
        return false;
    }

    value = line.substr(value_start, value_end - value_start);
    return true;
}

bool extract_json_u64(const std::string &line, const char *key, uint64_t &value) {
    const std::string needle = std::string("\"") + key + "\":";
    const size_t start = line.find(needle);
    if (start == std::string::npos) {
        return false;
    }

    size_t value_start = start + needle.size();
    size_t value_end = value_start;
    while (value_end < line.size() && line[value_end] >= '0' && line[value_end] <= '9') {
        ++value_end;
    }
    if (value_end == value_start) {
        return false;
    }

    try {
        value = std::stoull(line.substr(value_start, value_end - value_start));
    } catch (const std::invalid_argument &) {
        return false;
    } catch (const std::out_of_range &) {
        return false;
    }
    return true;
}

TimingObservation parse_timing_observation(const std::string &path) {
    TimingObservation timing;
    timing.cycle_log_path = path;
    bool has_nonzero_record = false;

    std::ifstream cycle_log(path);
    if (!cycle_log) {
        timing.status = "unavailable";
        return timing;
    }

    std::string line;
    while (std::getline(cycle_log, line)) {
        std::string op;
        uint64_t start = 0;
        uint64_t end = 0;
        if (!extract_json_string(line, "op", op) ||
            !extract_json_u64(line, "start", start) ||
            !extract_json_u64(line, "end", end)) {
            continue;
        }

        CycleLabelRecord *record = nullptr;
        if (op == "exsia.local_stage_total") {
            record = &timing.local_stage;
        } else if (op == "exsia.stripe_folding_total") {
            record = &timing.stripe_folding;
        } else if (op == "exsia.end_to_end") {
            record = &timing.end_to_end;
        }

        if (!record) {
            continue;
        }

        record->present = true;
        record->start = start;
        record->end = end;
        record->delta = end >= start ? end - start : 0;
        has_nonzero_record = has_nonzero_record || start != 0 || end != 0 || record->delta != 0;
        ++timing.record_count;
    }

    if (!timing.local_stage.present && !timing.stripe_folding.present && !timing.end_to_end.present) {
        timing.status = "unavailable";
        return timing;
    }

    if (!has_nonzero_record) {
        timing.status = "unavailable";
        return timing;
    }

    timing.stage_sum = timing.local_stage.delta + timing.stripe_folding.delta;
    timing.non_stage_overhead = timing.end_to_end.delta >= timing.stage_sum
                                    ? timing.end_to_end.delta - timing.stage_sum
                                    : 0;
    timing.status = timing.local_stage.present && timing.stripe_folding.present && timing.end_to_end.present
                        ? "available"
                        : "incomplete";
    return timing;
}

int16_t exp_to_theta_for_test(int16_t exponent, int16_t rho) {
    const int16_t neg_inf = std::numeric_limits<int16_t>::min();
    return exponent == neg_inf ? neg_inf : static_cast<int16_t>(static_cast<int>(exponent) - static_cast<int>(rho));
}

long double scaled_magnitude_for_test(float value, int16_t theta) {
    const int16_t neg_inf = std::numeric_limits<int16_t>::min();
    if (theta == neg_inf || !std::isfinite(value)) {
        return 0.0L;
    }

    return std::fabs(std::ldexp(static_cast<long double>(value), -static_cast<int>(theta)));
}

bool detect_float_upper_tail(long double magnitude,
                             long double sum,
                             long double sum_squares,
                             size_t count,
                             int tau) {
    if (count == 0) {
        return false;
    }

    const long double count_ld = static_cast<long double>(count);
    const long double centered = count_ld * magnitude - sum;
    const long double variance_numer = count_ld * sum_squares - sum * sum;
    if (centered <= 0.0L || variance_numer <= 0.0L) {
        return false;
    }

    const long double tau_squared = static_cast<long double>(tau) * static_cast<long double>(tau);
    return centered * centered > tau_squared * variance_numer;
}

std::vector<float> make_timing_selection_fixture(size_t rows, size_t cols) {
    std::vector<float> fixture(rows * cols, 0.0f);
    for (size_t row = 0; row < rows; ++row) {
        for (size_t col = 0; col < cols; ++col) {
            const size_t block_col = col % BLOCK_SIZE;
            const float sign = ((row + col) % 2 == 0) ? 1.0f : -1.0f;
            float value = 0.0f;

            if (block_col % 11 == 0) {
                value = std::ldexp(sign * (1.0f + 0.125f * static_cast<float>((row + col) % 3)), 8);
            } else if (block_col % 7 == 0) {
                value = std::ldexp(sign * (1.0f + 0.03125f * static_cast<float>((row + 2 * col) % 5)), 2);
            } else if (block_col % 5 == 0) {
                value = std::ldexp(sign * (1.0f + 0.015625f * static_cast<float>((2 * row + col) % 7)), 1);
            } else {
                value = sign * (0.375f + 0.0625f * static_cast<float>((3 * row + col) % 9));
            }

            fixture[row * cols + col] = value;
        }
    }

    return fixture;
}

SelectionComparisonResult compute_selection_comparison(const std::vector<float> &fixture,
                                                       size_t rows,
                                                       size_t cols,
                                                       int tau,
                                                       int rho) {
    SelectionComparisonResult result;
    result.ok = true;
    result.tau = tau;
    result.rho = rho;
    result.rows = rows;
    result.cols = cols;
    result.input_fixture = fixture;
    result.top1_mask.assign(rows * cols, 0);
    result.float_mask.assign(rows * cols, 0);
    result.integer_mask.assign(rows * cols, 0);

    size_t padded_cols = 0;
    if (!checked_round_up_multiple(cols, BLOCK_SIZE, padded_cols)) {
        result.ok = false;
        return result;
    }

    const size_t blocks_per_row = padded_cols / BLOCK_SIZE;
    const int16_t neg_inf = std::numeric_limits<int16_t>::min();
    ExpScanner scanner;
    WideQuantizer quantizer;
    SigmaDetector detector;
    BlockState block;
    if (!block.prepare(BLOCK_SIZE)) {
        result.ok = false;
        return result;
    }

    BitMask top1_mask;
    if (!top1_mask.prepare(1, BLOCK_SIZE)) {
        result.ok = false;
        return result;
    }
    std::vector<long double> float_magnitudes(BLOCK_SIZE, 0.0L);

    for (size_t row = 0; row < rows; ++row) {
        for (size_t block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
            for (size_t i = 0; i < BLOCK_SIZE; ++i) {
                const size_t col = block_idx * BLOCK_SIZE + i;
                block.x[i] = col < cols ? fixture[row * cols + col] : 0.0f;
            }

            scanner.scan_top2_exp(block.x, block);
            top1_mask.clear_active_bits();

            const bool has_second_bucket = block.e2 != neg_inf;
            if (has_second_bucket) {
                for (size_t i = 0; i < BLOCK_SIZE; ++i) {
                    const size_t col = block_idx * BLOCK_SIZE + i;
                    if (col < cols && block.e[i] != neg_inf && block.e[i] == block.e1) {
                        top1_mask.set(0, i);
                        result.top1_mask[row * cols + col] = 1;
                    }
                }
            }

            const int16_t block_exponent = has_second_bucket ? block.e2 : block.e1;
            const int16_t theta_pre = exp_to_theta_for_test(block_exponent, static_cast<int16_t>(rho));
            const auto [q_tmp, int_sum, int_sum_squares] =
                quantizer.quantize_block(block.x, 0, 0, top1_mask, theta_pre);

            size_t candidate_count = 0;
            long double float_sum = 0.0L;
            long double float_sum_squares = 0.0L;
            for (size_t i = 0; i < BLOCK_SIZE; ++i) {
                const size_t col = block_idx * BLOCK_SIZE + i;
                if (col >= cols || top1_mask.is_set(0, i)) {
                    float_magnitudes[i] = 0.0L;
                    continue;
                }

                const long double magnitude = scaled_magnitude_for_test(block.x[i], theta_pre);
                float_magnitudes[i] = magnitude;
                float_sum += magnitude;
                float_sum_squares += magnitude * magnitude;
                ++candidate_count;
            }

            for (size_t i = 0; i < BLOCK_SIZE; ++i) {
                const size_t col = block_idx * BLOCK_SIZE + i;
                if (col >= cols || top1_mask.is_set(0, i)) {
                    continue;
                }

                const size_t flat_index = row * cols + col;
                const bool float_selected = detect_float_upper_tail(
                    float_magnitudes[i], float_sum, float_sum_squares, candidate_count, tau);
                const bool integer_selected = detector.detect_sigma(
                    q_tmp[i], int_sum, int_sum_squares, candidate_count);

                result.float_mask[flat_index] = float_selected ? 1 : 0;
                result.integer_mask[flat_index] = integer_selected ? 1 : 0;

                if (float_selected) {
                    result.float_selected.push_back({row, col});
                }
                if (integer_selected) {
                    result.integer_selected.push_back({row, col});
                }
                if (float_selected && !integer_selected) {
                    result.float_only.push_back({row, col});
                } else if (!float_selected && integer_selected) {
                    result.integer_only.push_back({row, col});
                }
            }
        }
    }

    result.mismatch_count = result.float_only.size() + result.integer_only.size();
    return result;
}

constexpr bool quantize_activation_returns_bool =
    std::is_same_v<
        decltype(ggml::gemmini::quants::quantize_activation(
            static_cast<const ggml_tensor *>(nullptr),
            std::declval<ggml_gemmini_args_t &>())),
        bool>;

constexpr bool act_quantize_returns_bool =
    std::is_same_v<
        decltype(ggml::gemmini::quants::act::quantize(
            static_cast<const ggml_tensor *>(nullptr),
            std::declval<ggml_gemmini_args_t &>())),
        bool>;

bool same_outliers(const std::vector<ggml::gemmini::quants::act::ggml_gemmini_qact_outlier> &lhs,
                   const std::vector<ggml::gemmini::quants::act::ggml_gemmini_qact_outlier> &rhs) {
    if (lhs.size() != rhs.size()) {
        return false;
    }

    for (size_t i = 0; i < lhs.size(); ++i) {
        if (lhs[i].row != rhs[i].row || lhs[i].col != rhs[i].col || lhs[i].residual != rhs[i].residual) {
            return false;
        }
    }

    return true;
}

bool run_reference_exsia(Meta &meta,
                         ExSIAState &state,
                         const ggml_tensor *tensor,
                         ggml_gemmini_args_t &args) {
    meta.reset();
    meta.sigma = GGML_GEMMINI_EXSIA_SIGMA;
    state = ExSIAState{};
    state.B_size = BLOCK_SIZE;
    state.K_logical = args.K;

    if (args.I == 0 || args.K == 0 || !checked_round_up_multiple(args.K, state.B_size, state.K_padded)) {
        return false;
    }

    state.blocks_per_row = state.K_padded / state.B_size;

    size_t rows_per_stripe = args.I;
    if (args.tile_I > 0 && !checked_mul_size(args.tile_I, DIM, rows_per_stripe)) {
        return false;
    }
    if (rows_per_stripe == 0) {
        return false;
    }

    size_t stripe_round_input = 0;
    if (!checked_add_size(args.I, rows_per_stripe - 1, stripe_round_input)) {
        return false;
    }

    const size_t num_stripes = stripe_round_input / rows_per_stripe;
    const float *src_data = ggml::gemmini::activation_data(tensor);
    if (!src_data) {
        return false;
    }

    size_t logical_elem_count = 0;
    if (!checked_mul_size(args.I, args.K, logical_elem_count)) {
        return false;
    }

    meta.theta.assign(num_stripes, std::numeric_limits<int16_t>::min());
    meta.outliers.clear();
    meta.outliers.reserve(logical_elem_count);

    state.stripe.assign(num_stripes, StripeState{});
    std::vector<std::vector<int32_t>> stripe_q_wide(num_stripes);
    std::vector<std::vector<int16_t>> stripe_block_exp(num_stripes);
    for (size_t s = 0; s < num_stripes; ++s) {
        StripeState &stripe = state.stripe[s];
        stripe.row_start = s * rows_per_stripe;
        stripe.row_end = std::min((s + 1) * rows_per_stripe, args.I);
        if (!stripe.outlier_mask.prepare(stripe.row_count(), state.K_padded) ||
            !stripe.scratch.prepare(state.B_size)) {
            return false;
        }

        size_t elem_count = 0;
        size_t block_count = 0;
        if (!checked_mul_size(stripe.row_count(), state.K_padded, elem_count) ||
            !checked_mul_size(stripe.row_count(), state.blocks_per_row, block_count)) {
            return false;
        }

        stripe_q_wide[s].assign(elem_count, 0);
        stripe_block_exp[s].assign(block_count, std::numeric_limits<int16_t>::min());
    }

    LocalStage local;
    StripeFolding folding;
    for (size_t row = 0; row < args.I; ++row) {
        const size_t stripe_idx = row / rows_per_stripe;
        StripeState &stripe = state.stripe[stripe_idx];
        for (size_t block = 0; block < state.blocks_per_row; ++block) {
            std::vector<float> &block_x = stripe.scratch.block.x;
            const size_t col_offset = block * state.B_size;
            for (size_t i = 0; i < state.B_size; ++i) {
                const size_t col = col_offset + i;
                block_x[i] = col < args.K ? src_data[row * args.K + col] : 0.0f;
            }

            uint64_t cycle_delta = 0;
            if (!local.run(meta, state, stripe, block_x, row, block,
                           stripe_q_wide[stripe_idx], stripe_block_exp[stripe_idx], cycle_delta)) {
                return false;
            }
        }
    }

    int8_t *dst = reinterpret_cast<int8_t *>(args.A);
    for (size_t s = 0; s < num_stripes; ++s) {
        uint64_t cycle_delta = 0;
        if (!folding.run(meta, state, state.stripe[s], args, s, dst,
                         stripe_q_wide[s], stripe_block_exp[s], cycle_delta)) {
            return false;
        }
    }

    return true;
}

}

int main(int argc, char **argv) {
    std::remove("exsia_test_artifact.json");

    std::string case_name = "smoke";
    size_t rows = 1;
    size_t cols = 1;
    const int compiled_tau = GGML_GEMMINI_EXSIA_SIGMA;
    int observed_meta_sigma = compiled_tau;
    int requested_tau = compiled_tau;
    bool requested_tau_present = false;
    bool requested_tau_valid = false;
    std::string failure_reason;

    bool compiled_tau_check = true;
    bool meta_sigma_check = true;
    bool quantizer_check = false;
    bool detector_check = false;
    bool invalid_zero_dimension_check = false;
    bool initial_state_check = false;
    bool finite_input_policy_check = false;
    bool shift_extremes_check = false;
    bool multi_stripe_check = false;
    bool dequantize_check = false;
    bool outlier_order_check = false;
    bool mask_layout_check = false;
    bool mask_boundary_check = false;
    bool reference_check = false;
    bool reference_case_checked = false;
    bool scratch_reuse_check = false;
    bool stream_parity_check = false;
    bool stream_invalid_shape_check = false;
    bool timing_and_selection_check = false;
    bool timing_missing_log_check = false;
    std::string timing_json = "null";
    std::string selection_comparison_json = "null";

    const auto write_bool = [](std::ofstream &artifact, bool value) {
        artifact << (value ? "true" : "false");
    };
    const auto write_coord_list = [](std::ostringstream &artifact, const std::vector<Coord> &coords) {
        artifact << "[";
        for (size_t i = 0; i < coords.size(); ++i) {
            if (i != 0) {
                artifact << ",";
            }
            artifact << "[" << coords[i].row << "," << coords[i].col << "]";
        }
        artifact << "]";
    };
    const auto write_mask_matrix = [](std::ostringstream &artifact,
                                      const std::vector<uint8_t> &mask,
                                      size_t mask_rows,
                                      size_t mask_cols) {
        artifact << "[";
        for (size_t row = 0; row < mask_rows; ++row) {
            if (row != 0) {
                artifact << ",";
            }
            artifact << "[";
            for (size_t col = 0; col < mask_cols; ++col) {
                if (col != 0) {
                    artifact << ",";
                }
                artifact << static_cast<int>(mask[row * mask_cols + col]);
            }
            artifact << "]";
        }
        artifact << "]";
    };
    const auto write_float_matrix = [](std::ostringstream &artifact,
                                       const std::vector<float> &values,
                                       size_t value_rows,
                                       size_t value_cols) {
        artifact << "[";
        for (size_t row = 0; row < value_rows; ++row) {
            if (row != 0) {
                artifact << ",";
            }
            artifact << "[";
            for (size_t col = 0; col < value_cols; ++col) {
                if (col != 0) {
                    artifact << ",";
                }
                artifact << values[row * value_cols + col];
            }
            artifact << "]";
        }
        artifact << "]";
    };
    const auto build_timing_json = [](const TimingObservation &timing) {
        std::ostringstream artifact;
        const auto write_record = [&](const char *label, const CycleLabelRecord &record) {
            artifact << "\"" << label << "\":{"
                     << "\"present\":" << (record.present ? "true" : "false") << ","
                     << "\"start\":";
            if (record.present) {
                artifact << record.start;
            } else {
                artifact << "null";
            }
            artifact << ",\"end\":";
            if (record.present) {
                artifact << record.end;
            } else {
                artifact << "null";
            }
            artifact << ",\"delta\":";
            if (record.present) {
                artifact << record.delta;
            } else {
                artifact << "null";
            }
            artifact << "}";
        };

        artifact << "{"
                 << "\"status\":\"" << json_escape(timing.status) << "\","
                 << "\"cycle_log_path\":\"" << json_escape(timing.cycle_log_path) << "\","
                 << "\"record_count\":" << timing.record_count << ","
                 << "\"stage_sum_cycles\":";
        if (timing.status == "available" || timing.status == "incomplete") {
            artifact << timing.stage_sum;
        } else {
            artifact << "null";
        }
        artifact << ",\"non_stage_cycles\":";
        if (timing.status == "available" || timing.status == "incomplete") {
            artifact << timing.non_stage_overhead;
        } else {
            artifact << "null";
        }
        artifact << ",\"records\":{";
        write_record("exsia.local_stage_total", timing.local_stage);
        artifact << ",";
        write_record("exsia.stripe_folding_total", timing.stripe_folding);
        artifact << ",";
        write_record("exsia.end_to_end", timing.end_to_end);
        artifact << "}}";
        return artifact.str();
    };
    const auto build_selection_json = [&](const SelectionComparisonResult &comparison) {
        std::ostringstream artifact;
        artifact << "{"
                 << "\"status\":\"" << (comparison.ok ? "available" : "failure") << "\","
                 << "\"criterion\":\"" << json_escape(comparison.criterion) << "\","
                 << "\"tau\":" << comparison.tau << ","
                 << "\"rho\":" << comparison.rho << ","
                 << "\"rows\":" << comparison.rows << ","
                 << "\"cols\":" << comparison.cols << ","
                 << "\"input_fixture\":";
        write_float_matrix(artifact, comparison.input_fixture, comparison.rows, comparison.cols);
        artifact << ",\"top1_preselection_mask\":";
        write_mask_matrix(artifact, comparison.top1_mask, comparison.rows, comparison.cols);
        artifact << ",\"float_upper_tail_mask\":";
        write_mask_matrix(artifact, comparison.float_mask, comparison.rows, comparison.cols);
        artifact << ",\"integer_upper_tail_mask\":";
        write_mask_matrix(artifact, comparison.integer_mask, comparison.rows, comparison.cols);
        artifact << ",\"float_selected_indices\":";
        write_coord_list(artifact, comparison.float_selected);
        artifact << ",\"integer_selected_indices\":";
        write_coord_list(artifact, comparison.integer_selected);
        artifact << ",\"mismatches\":{"
                 << "\"float_only\":";
        write_coord_list(artifact, comparison.float_only);
        artifact << ",\"integer_only\":";
        write_coord_list(artifact, comparison.integer_only);
        artifact << ",\"count\":"
                 << comparison.mismatch_count
                 << "}}";
        return artifact.str();
    };
    const auto write_artifact = [&](const char *status) {
        std::ofstream artifact("exsia_test_artifact.json");
        if (!artifact) {
            return false;
        }

        artifact << "{\n"
                 << "  \"status\": \"" << json_escape(status) << "\",\n"
                 << "  \"case\": \"" << json_escape(case_name) << "\",\n"
                 << "  \"rows\": " << rows << ",\n"
                 << "  \"cols\": " << cols << ",\n"
                 << "  \"compiled_tau\": " << compiled_tau << ",\n"
                 << "  \"meta_sigma\": " << observed_meta_sigma << ",\n"
                 << "  \"compiled_tau_check\": ";
        write_bool(artifact, compiled_tau_check);
        artifact << ",\n"
                 << "  \"meta_sigma_check\": ";
        write_bool(artifact, meta_sigma_check);
        artifact << ",\n"
                 << "  \"requested_compiled_tau\": ";
        if (requested_tau_present && requested_tau_valid) {
            artifact << requested_tau;
        } else {
            artifact << "null";
        }
        artifact << ",\n"
                 << "  \"error\": \"" << json_escape(failure_reason) << "\",\n"
                 << "  \"quantizer\": ";
        write_bool(artifact, quantizer_check);
        artifact << ",\n"
                 << "  \"detector\": ";
        write_bool(artifact, detector_check);
        artifact << ",\n"
                 << "  \"invalid_zero_dimension\": ";
        write_bool(artifact, invalid_zero_dimension_check);
        artifact << ",\n"
                 << "  \"initial_state\": ";
        write_bool(artifact, initial_state_check);
        artifact << ",\n"
                 << "  \"finite_input_policy\": ";
        write_bool(artifact, finite_input_policy_check);
        artifact << ",\n"
                 << "  \"shift_extremes\": ";
        write_bool(artifact, shift_extremes_check);
        artifact << ",\n"
                 << "  \"multi_stripe\": ";
        write_bool(artifact, multi_stripe_check);
        artifact << ",\n"
                 << "  \"dequantize\": ";
        write_bool(artifact, dequantize_check);
        artifact << ",\n"
                 << "  \"outlier_order\": ";
        write_bool(artifact, outlier_order_check);
        artifact << ",\n"
                 << "  \"mask_layout\": ";
        write_bool(artifact, mask_layout_check);
        artifact << ",\n"
                 << "  \"mask_boundary\": ";
        write_bool(artifact, mask_boundary_check);
        artifact << ",\n"
                 << "  \"scratch_reuse\": ";
        write_bool(artifact, scratch_reuse_check);
        artifact << ",\n"
                 << "  \"stream_parity\": ";
        write_bool(artifact, stream_parity_check);
        artifact << ",\n"
                 << "  \"stream_invalid_shape\": ";
        write_bool(artifact, stream_invalid_shape_check);
        artifact << ",\n"
                 << "  \"timing\": " << timing_json << ",\n"
                 << "  \"selection_comparison\": " << selection_comparison_json << ",\n"
                 << "  \"reference_status\": \""
                 << (reference_case_checked ? (reference_check ? "pass" : "failure") : "not-run")
                 << "\"\n}\n";
        artifact.flush();
        return static_cast<bool>(artifact) && artifact.tellp() > 0;
    };
    const auto fail = [&](const char *reason) {
        if (failure_reason.empty()) {
            failure_reason = reason;
        }
        write_artifact("failure");
        return 1;
    };
    const auto parse_decimal = [](const std::string &text, unsigned long long &value) {
        if (text.empty() || !std::all_of(text.begin(), text.end(), [](char character) {
                return character >= '0' && character <= '9';
            })) {
            return false;
        }

        size_t consumed = 0;
        try {
            value = std::stoull(text, &consumed, 10);
        } catch (const std::invalid_argument &) {
            return false;
        } catch (const std::out_of_range &) {
            return false;
        }
        return consumed == text.size();
    };

    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        const size_t equals = argument.find('=');
        const std::string option = argument.substr(0, equals);
        const std::string value = equals == std::string::npos
                                      ? (i + 1 < argc ? argv[++i] : "")
                                      : argument.substr(equals + 1);
        if (value.empty()) {
            if (option == "--compiled-tau") {
                compiled_tau_check = false;
            }
            return fail("missing option value");
        }

        if (option == "--case") {
            case_name = value;
        } else if (option == "--rows" || option == "--cols") {
            unsigned long long parsed = 0;
            if (!parse_decimal(value, parsed) || parsed > std::numeric_limits<size_t>::max()) {
                return fail("invalid numeric input");
            }
            if (option == "--rows") {
                rows = static_cast<size_t>(parsed);
            } else {
                cols = static_cast<size_t>(parsed);
            }
        } else if (option == "--compiled-tau") {
            requested_tau_present = true;
            unsigned long long parsed = 0;
            if (!parse_decimal(value, parsed) || parsed > std::numeric_limits<int>::max()) {
                compiled_tau_check = false;
                return fail("invalid numeric input");
            }
            requested_tau = static_cast<int>(parsed);
            requested_tau_valid = true;
            compiled_tau_check = requested_tau == compiled_tau;
            if (!compiled_tau_check) {
                return fail("compiled tau mismatch");
            }
        } else {
            return fail("unsupported option");
        }
    }

    if (case_name != "all" &&
        case_name != "smoke" && case_name != "reference" && case_name != "config" &&
        case_name != "invalid-shape" && case_name != "detector" &&
        case_name != "mask-layout" && case_name != "mask-boundary" &&
        case_name != "scratch-reuse" && case_name != "stream-parity" &&
        case_name != "stream-invalid-shape" &&
        case_name != "shift-extremes" &&
        case_name != "timing-and-selection" &&
        case_name != "timing-missing-log") {
        return fail("unsupported case");
    }

    if (case_name == "config") {
        compiled_tau_check = requested_tau == GGML_GEMMINI_EXSIA_SIGMA;
        if (!compiled_tau_check) {
            return fail("config case requires compiled tau to match the macro");
        }
    }

    const auto check = [&](bool condition, const char *reason) {
        if (!condition && failure_reason.empty()) {
            failure_reason = reason;
        }
        return condition;
    };

    WideQuantizer quantizer;
    const std::vector<float> source = {1.0f, -2.0f, 0.5f};
    const std::vector<int32_t> expected_quantized = {1, -2, 0};
    const auto quantized = quantizer.quantize_block(source, 0);
    const BitMask no_mask;
    const auto quantized_with_stats = quantizer.quantize_block(source, 0, 0, no_mask, 0);
    quantizer_check = check(
        quantized == expected_quantized &&
            std::get<0>(quantized_with_stats) == expected_quantized &&
            std::get<1>(quantized_with_stats) == static_cast<__int128_t>(3) &&
            std::get<2>(quantized_with_stats) == static_cast<__int128_t>(5),
        "quantizer statistics mismatch");

    const auto min_quantized_with_stats = quantizer.quantize_block(
        std::vector<float>{-2147483648.0f}, 0, 0, no_mask, 0);
    const __int128_t int32_min_magnitude = static_cast<__int128_t>(std::numeric_limits<int32_t>::max()) + 1;
    quantizer_check = check(
        quantizer_check &&
            std::get<0>(min_quantized_with_stats) == std::vector<int32_t>{std::numeric_limits<int32_t>::min()} &&
            std::get<1>(min_quantized_with_stats) == int32_min_magnitude &&
            std::get<2>(min_quantized_with_stats) == int32_min_magnitude * int32_min_magnitude,
        "INT32_MIN magnitude statistics mismatch");

    SigmaDetector detector;
    detector_check = check(!detector.detect_sigma(0, 0, 0, 0), "detector zero-count check failed");
    const __int128_t upper_tail_sum = 115;
    const __int128_t upper_tail_sum_squares = 10015;
    const bool positive_tail = detector.detect_sigma(100, upper_tail_sum, upper_tail_sum_squares, 16);
    const bool negative_tail = detector.detect_sigma(-100, upper_tail_sum, upper_tail_sum_squares, 16);
    const __int128_t upper_tail_sigma_squared =
        static_cast<__int128_t>(compiled_tau) * compiled_tau;
    const __int128_t upper_tail_centered = static_cast<__int128_t>(16) * 100 - upper_tail_sum;
    const __int128_t upper_tail_variance =
        static_cast<__int128_t>(16) * upper_tail_sum_squares - upper_tail_sum * upper_tail_sum;
    const bool expected_upper_tail =
        upper_tail_centered > 0 && upper_tail_variance > 0 &&
        upper_tail_centered * upper_tail_centered > upper_tail_sigma_squared * upper_tail_variance;
    detector_check = check(
        detector_check && positive_tail == expected_upper_tail &&
            negative_tail == expected_upper_tail && positive_tail == negative_tail,
        "sign-invariant magnitude detection failed");
    detector_check = check(
        detector_check && !detector.detect_sigma(1, upper_tail_sum, upper_tail_sum_squares, 16),
        "lower-tail magnitude value was selected");
    detector_check = check(
        detector_check && !detector.detect_sigma(1, 0, 0, 4),
        "zero variance was selected");
    detector_check = check(
        detector_check && !detector.detect_sigma(7, 28, 196, 4),
        "centered zero value was selected");
    const __int128_t boundary_sigma_squared =
        static_cast<__int128_t>(compiled_tau) * compiled_tau;
    const auto expected_boundary = [](int32_t q, __int128_t sigma_squared) {
        const __int128_t centered = static_cast<__int128_t>(4) * q - 10;
        return centered > 0 && centered * centered > sigma_squared * 20;
    };
    detector_check = check(
        detector_check &&
            detector.detect_sigma(5, 10, 30, 4) == expected_boundary(5, boundary_sigma_squared) &&
            detector.detect_sigma(6, 10, 30, 4) == expected_boundary(6, boundary_sigma_squared),
        "sigma cross-multiplication boundary mismatch");
    const __int128_t int32_max_magnitude = static_cast<__int128_t>(std::numeric_limits<int32_t>::max());
    const size_t large_magnitude_count =
        static_cast<size_t>(compiled_tau) * static_cast<size_t>(compiled_tau) + 1;
    detector_check = check(
        detector_check &&
            detector.detect_sigma(std::numeric_limits<int32_t>::max(), 0,
                                  int32_max_magnitude * int32_max_magnitude, large_magnitude_count) &&
            detector.detect_sigma(std::numeric_limits<int32_t>::min(), 0,
                                  int32_min_magnitude * int32_min_magnitude, large_magnitude_count),
        "large signed magnitudes were not detected safely");
    if (case_name == "detector" && !detector_check) {
        return fail("corrected detector expectation failed");
    }

    ggml_gemmini_args_t invalid_rows_args;
    invalid_rows_args.I = 0;
    invalid_rows_args.K = cols;
    Meta invalid_rows_meta;
    ExSIA invalid_rows_exsia;
    const bool invalid_rows = !invalid_rows_exsia.run(invalid_rows_meta, nullptr, invalid_rows_args);

    ggml_gemmini_args_t invalid_cols_args;
    invalid_cols_args.I = rows;
    invalid_cols_args.K = 0;
    Meta invalid_cols_meta;
    ExSIA invalid_cols_exsia;
    const bool invalid_cols = !invalid_cols_exsia.run(invalid_cols_meta, nullptr, invalid_cols_args);
    invalid_zero_dimension_check = check(invalid_rows && invalid_cols, "zero-dimension shape was accepted");

    ExSIA initial_exsia;
    initial_state_check = check(initial_exsia.state().K_logical == 0, "initial ExSIA state was not empty");

    {
        ggml_tensor finite_policy_tensor{};
        const std::vector<float> finite_policy_source = {
            0.0f,
            std::numeric_limits<float>::quiet_NaN(),
            std::numeric_limits<float>::infinity(),
            -std::numeric_limits<float>::infinity(),
        };
        std::vector<elem_t> finite_policy_quantized(finite_policy_source.size(), static_cast<elem_t>(91));
        std::vector<float> finite_policy_dequantized(finite_policy_source.size(), 1.0f);
        finite_policy_tensor.type = GGML_TYPE_F32;
        finite_policy_tensor.data = const_cast<float *>(finite_policy_source.data());

        ggml_gemmini_args_t finite_policy_args;
        finite_policy_args.I = 1;
        finite_policy_args.K = finite_policy_source.size();
        finite_policy_args.A = finite_policy_quantized.data();
        finite_policy_args.sA = finite_policy_source.size();
        auto &finite_policy_meta = finite_policy_args.act_quant.storage().emplace<Meta>();

        ExSIA finite_policy_exsia;
        const bool finite_policy_run = finite_policy_exsia.run(
            finite_policy_meta, &finite_policy_tensor, finite_policy_args);
        const bool finite_policy_dequantize = finite_policy_run && dequantize_activation(
            finite_policy_dequantized.data(),
            finite_policy_source.size(),
            1,
            1,
            finite_policy_source.size(),
            finite_policy_args);
        finite_input_policy_check = check(
            finite_policy_run &&
                finite_policy_dequantize &&
                std::all_of(finite_policy_quantized.begin(),
                            finite_policy_quantized.end(),
                            [](elem_t value) { return value == 0; }) &&
                std::all_of(finite_policy_dequantized.begin(),
                            finite_policy_dequantized.end(),
                            [](float value) { return value == 0.0f; }) &&
                finite_policy_meta.outliers.empty() &&
                finite_policy_meta.resolve_stripe_theta(0) ==
                    static_cast<int16_t>(0 - finite_policy_meta.rho),
            "non-finite input policy did not sanitize to exact zeros");
    }

    shift_extremes_check = check(
        detail::shift_q_i32(0, 0) == 0 &&
            detail::shift_q_i32(7, 0) == 7 &&
            detail::shift_q_i32(1, 31) == std::numeric_limits<int32_t>::max() &&
            detail::shift_q_i32(-1, 31) == std::numeric_limits<int32_t>::min() &&
            detail::shift_q_i32(std::numeric_limits<int32_t>::max(), 1) == std::numeric_limits<int32_t>::max() &&
            detail::shift_q_i32(std::numeric_limits<int32_t>::min(), 1) == std::numeric_limits<int32_t>::min() &&
            detail::shift_q_i32(4, -1) == 2 &&
            detail::shift_q_i32(5, -1) == 3 &&
            detail::shift_q_i32(-4, -1) == -2 &&
            detail::shift_q_i32(-5, -1) == -3 &&
            detail::shift_q_i32(7, -2) == 2 &&
            detail::shift_q_i32(-7, -2) == -2 &&
            detail::shift_q_i32(std::numeric_limits<int32_t>::max(), -31) == 1 &&
            detail::shift_q_i32(std::numeric_limits<int32_t>::min(), -31) == -1 &&
            detail::shift_q_i32(3, static_cast<int16_t>(127)) == std::numeric_limits<int32_t>::max() &&
            detail::shift_q_i32(-3, static_cast<int16_t>(127)) == std::numeric_limits<int32_t>::min() &&
            detail::shift_q_i32(3, static_cast<int16_t>(-127)) == 0 &&
            detail::shift_q_i32(-3, static_cast<int16_t>(-127)) == 0,
        "shift_q_i32 extreme-shift coverage failed");

    const auto check_invalid_shape_failure = [&](size_t requested_rows, size_t requested_cols) {
        const size_t invalid_rows = requested_rows;
        const size_t invalid_cols = requested_rows == 0 ? requested_cols : 0;
        const size_t output_cols = std::max<size_t>(requested_cols, 1);
        float invalid_source = 1.0f;
        ggml_tensor invalid_tensor{};
        invalid_tensor.type = GGML_TYPE_F32;
        invalid_tensor.data = &invalid_source;

        std::vector<elem_t> quantized(std::max<size_t>(output_cols, 1), static_cast<elem_t>(77));
        ggml_gemmini_args_t quant_args;
        quant_args.I = invalid_rows;
        quant_args.K = invalid_cols;
        quant_args.A = quantized.data();
        quant_args.sA = output_cols;
        quant_args.gemmini_call_k_logical = 999;
        quant_args.gemmini_call_k_aligned = 999;
        quant_args.gemmini_call_tile_k_elems = 999;
        quant_args.act_quant.storage().emplace<Meta>();

        const bool quantized_ok =
            ggml::gemmini::quants::quantize_activation(&invalid_tensor, quant_args);
        const bool quant_meta_cleared = std::holds_alternative<ggml::gemmini::quants::act::NoneMeta>(
            quant_args.act_quant.storage());
        const bool quant_call_state_cleared =
            quant_args.gemmini_call_k_logical == 0 &&
            quant_args.gemmini_call_k_aligned == 0 &&
            quant_args.gemmini_call_tile_k_elems == 0;
        const bool quant_dst_reset = std::all_of(
            quantized.begin(), quantized.end(), [](elem_t value) { return value == 0; });

        std::vector<elem_t> dispatch_quantized(std::max<size_t>(output_cols, 1), static_cast<elem_t>(88));
        ggml_gemmini_args_t dispatch_args;
        dispatch_args.I = invalid_rows;
        dispatch_args.K = invalid_cols;
        dispatch_args.A = dispatch_quantized.data();
        dispatch_args.sA = output_cols;
        dispatch_args.act_quant.storage().emplace<Meta>();

        const bool dispatch_ok = ggml::gemmini::quants::act::quantize(&invalid_tensor, dispatch_args);
        const bool dispatch_meta_cleared = std::holds_alternative<ggml::gemmini::quants::act::NoneMeta>(
            dispatch_args.act_quant.storage());
        const bool dispatch_dst_reset = std::all_of(
            dispatch_quantized.begin(), dispatch_quantized.end(), [](elem_t value) { return value == 0; });

        return quantize_activation_returns_bool &&
               act_quantize_returns_bool &&
               !quantized_ok &&
               !dispatch_ok &&
               quant_meta_cleared &&
               dispatch_meta_cleared &&
               quant_call_state_cleared &&
               quant_dst_reset &&
               dispatch_dst_reset;
    };

    if (case_name == "invalid-shape") {
        const bool invalid_shape_check = check(
            check_invalid_shape_failure(rows, cols),
            "invalid shape did not return false/reset cleanly");
        if (!invalid_shape_check) {
            return fail(failure_reason.empty() ? "invalid shape failed" : failure_reason.c_str());
        }

        failure_reason = failure_reason.empty() ? "invalid shape" : failure_reason;
        return fail(failure_reason.c_str());
    }

    if (case_name == "stream-invalid-shape" || case_name == "all") {
        stream_invalid_shape_check = check(
            check_invalid_shape_failure(rows, cols),
            "stream invalid-shape did not propagate failure/reset cleanly");
        if (case_name == "stream-invalid-shape" && !stream_invalid_shape_check) {
            return fail(failure_reason.empty() ? "stream invalid-shape failed" : failure_reason.c_str());
        }
    }

    if (case_name == "detector" && !detector_check) {
        return fail("corrected detector expectation failed");
    }

    if (case_name == "shift-extremes" && !shift_extremes_check) {
        return fail("shift-extremes failed");
    }

    if (case_name == "timing-and-selection" || case_name == "timing-missing-log" || case_name == "all") {
        const std::string cycle_log_path = resolve_cycle_log_path();
        std::remove(cycle_log_path.c_str());

        const std::vector<float> fixture_source = make_timing_selection_fixture(rows, cols);
        std::vector<elem_t> quantized_fixture(rows * cols, 0);
        ggml_tensor fixture_tensor{};
        fixture_tensor.type = GGML_TYPE_F32;
        fixture_tensor.data = const_cast<float *>(fixture_source.data());

        ggml_gemmini_args_t fixture_args;
        fixture_args.I = rows;
        fixture_args.K = cols;
        fixture_args.A = quantized_fixture.data();
        fixture_args.sA = cols;
        fixture_args.tile_I = 1;
        auto &fixture_meta = fixture_args.act_quant.storage().emplace<Meta>();

        ExSIA fixture_exsia;
        const bool run_ok = fixture_exsia.run(fixture_meta, &fixture_tensor, fixture_args);
        observed_meta_sigma = fixture_meta.sigma;
        meta_sigma_check = check(
            run_ok && fixture_meta.sigma == GGML_GEMMINI_EXSIA_SIGMA,
            "Meta sigma does not match compiled tau");
        if (!run_ok) {
            return fail("timing fixture ExSIA run failed");
        }

        const TimingObservation timing = parse_timing_observation(cycle_log_path);
        timing_json = build_timing_json(timing);

        const bool timing_should_be_available = CYCLE_DETAIL != 0 && cycle_reader_supported;
        if (case_name == "timing-and-selection" || case_name == "all") {
            const SelectionComparisonResult comparison = compute_selection_comparison(
                fixture_source, rows, cols, compiled_tau, fixture_meta.rho);
            selection_comparison_json = build_selection_json(comparison);
            const bool mismatch_count_check = comparison.mismatch_count ==
                comparison.float_only.size() + comparison.integer_only.size();
            const bool expected_mismatch_check =
                comparison.float_only.empty() &&
                comparison.integer_only.size() == 1 &&
                comparison.integer_only[0].row == 31 &&
                comparison.integer_only[0].col == 7;
            timing_and_selection_check = check(
                ((timing_should_be_available &&
                  timing.status == "available" &&
                  timing.local_stage.present &&
                  timing.stripe_folding.present &&
                  timing.end_to_end.present &&
                  timing.end_to_end.delta >= timing.stage_sum) ||
                 (!timing_should_be_available && timing.status == "unavailable")) &&
                    comparison.ok &&
                    mismatch_count_check &&
                    expected_mismatch_check,
                "timing or selection comparison artifact was incomplete");
            if (!timing_and_selection_check) {
                return fail(failure_reason.empty() ? "timing-and-selection failed" : failure_reason.c_str());
            }
        } else {
            timing_missing_log_check = check(
                !timing_should_be_available && timing.status == "unavailable",
                "timing instrumentation availability was not reported as unavailable");
            if (!timing_missing_log_check) {
                return fail(failure_reason.empty() ? "timing-missing-log failed" : failure_reason.c_str());
            }
        }
    }

    if (case_name == "all" ||
        case_name == "smoke" || case_name == "reference" || case_name == "config" ||
        case_name == "mask-layout" || case_name == "mask-boundary" ||
        case_name == "scratch-reuse" || case_name == "stream-parity") {
        reference_case_checked = true;
        const size_t stripe_rows = static_cast<size_t>(DIM);
        const size_t fixture_rows = std::max(rows, 2 * stripe_rows + 1);
        const size_t fixture_cols = std::max(cols, static_cast<size_t>(BLOCK_SIZE) + 5);
        const size_t fixture_size = fixture_rows * fixture_cols;

        std::vector<float> fixture_source(fixture_size, 0.0f);
        for (size_t row = 0; row < fixture_rows; ++row) {
            for (size_t col = 0; col < fixture_cols; ++col) {
                fixture_source[row * fixture_cols + col] = col < BLOCK_SIZE ? 1.0f : 1024.0f;
            }
        }

        std::vector<elem_t> quantized_fixture(fixture_size, 0);
        std::vector<float> dequantized_fixture(fixture_size, 0.0f);
        ggml_tensor fixture_tensor{};
        fixture_tensor.type = GGML_TYPE_F32;
        fixture_tensor.data = fixture_source.data();

        ggml_gemmini_args_t fixture_args;
        fixture_args.I = fixture_rows;
        fixture_args.K = fixture_cols;
        fixture_args.A = quantized_fixture.data();
        fixture_args.sA = fixture_cols;
        fixture_args.tile_I = 1;
        auto &fixture_meta = fixture_args.act_quant.storage().emplace<Meta>();
        observed_meta_sigma = fixture_meta.sigma;
        meta_sigma_check = check(
            fixture_meta.sigma == GGML_GEMMINI_EXSIA_SIGMA,
            "Meta sigma does not match compiled tau");

        ExSIA fixture_exsia;
        const bool run_ok = fixture_exsia.run(fixture_meta, &fixture_tensor, fixture_args);
        const auto &state = fixture_exsia.state();
        const bool bounded_public_scratch =
            state.x_f32.empty() && state.q_wide.empty() &&
            state.block_exp.empty() && state.residual.empty();
        const bool state_ok = run_ok && state.K_logical == fixture_cols &&
                              state.K_padded >= fixture_cols && state.stripe.size() > 1 &&
                              fixture_meta.theta.size() == state.stripe.size() &&
                              bounded_public_scratch;
        multi_stripe_check = check(run_ok && state.stripe.size() > 1, "multi-stripe ExSIA run was not observed");
        reference_check = check(state_ok, "valid ExSIA reference run failed or public scratch stayed live");

        const bool dequantized_ok = run_ok && dequantize_activation(
            dequantized_fixture.data(), fixture_cols, 1, fixture_rows, fixture_cols, fixture_args);
        bool values_ok = dequantized_ok;
        if (values_ok) {
            for (size_t i = 0; i < fixture_size; ++i) {
                const float tolerance = 1e-3f * std::max(1.0f, std::fabs(fixture_source[i]));
                if (std::fabs(dequantized_fixture[i] - fixture_source[i]) > tolerance) {
                    values_ok = false;
                    break;
                }
            }
        }
        dequantize_check = check(values_ok, "ExSIA dequantization mismatch");

        outlier_order_check = run_ok;
        for (size_t i = 0; outlier_order_check && i < fixture_meta.outliers.size(); ++i) {
            const auto &outlier = fixture_meta.outliers[i];
            const bool in_bounds = outlier.row >= 0 && outlier.col >= 0 &&
                                   static_cast<size_t>(outlier.row) < fixture_rows &&
                                   static_cast<size_t>(outlier.col) < fixture_cols &&
                                   outlier.residual != 0;
            const bool ordered = i == 0 ||
                                 fixture_meta.outliers[i - 1].row < outlier.row ||
                                 (fixture_meta.outliers[i - 1].row == outlier.row &&
                                  fixture_meta.outliers[i - 1].col < outlier.col);
            outlier_order_check = in_bounds && ordered;
        }
        outlier_order_check = check(outlier_order_check, "global outlier ordering check failed");
        mask_layout_check = run_ok;
        for (const auto &stripe : state.stripe) {
            const size_t expected_rows = stripe.row_end - stripe.row_start;
            mask_layout_check = mask_layout_check &&
                                stripe.outlier_mask.rows == expected_rows &&
                                stripe.outlier_mask.cols == state.K_padded;
        }
        mask_layout_check = check(mask_layout_check, "stripe mask layout was not stripe-local");

        mask_boundary_check = run_ok;
        for (const auto &outlier : fixture_meta.outliers) {
            if (outlier.row < 0 || outlier.col < 0) {
                mask_boundary_check = false;
                break;
            }

            const size_t row = static_cast<size_t>(outlier.row);
            const size_t col = static_cast<size_t>(outlier.col);
            if (row >= fixture_rows || col >= fixture_cols) {
                mask_boundary_check = false;
                break;
            }

            const size_t stripe_idx = row / stripe_rows;
            if (stripe_idx >= state.stripe.size()) {
                mask_boundary_check = false;
                break;
            }

            const auto &stripe = state.stripe[stripe_idx];
            if (row < stripe.row_start || row >= stripe.row_end) {
                mask_boundary_check = false;
                break;
            }

            const size_t local_row = row - stripe.row_start;
            if (local_row >= stripe.outlier_mask.rows || !stripe.outlier_mask.is_set(local_row, col)) {
                mask_boundary_check = false;
                break;
            }
        }
        mask_boundary_check = check(mask_boundary_check, "stripe-local outlier lookup failed");

        if (case_name == "stream-parity" || case_name == "all") {
            std::vector<elem_t> reference_quantized(fixture_size, 0);
            std::vector<float> reference_dequantized(fixture_size, 0.0f);
            ggml_gemmini_args_t reference_args;
            reference_args.I = fixture_rows;
            reference_args.K = fixture_cols;
            reference_args.A = reference_quantized.data();
            reference_args.sA = fixture_cols;
            reference_args.tile_I = 1;
            auto &reference_meta = reference_args.act_quant.storage().emplace<Meta>();

            ExSIAState reference_state;
            const bool reference_run = run_reference_exsia(reference_meta, reference_state, &fixture_tensor, reference_args);
            const bool reference_dequantize = reference_run && dequantize_activation(
                reference_dequantized.data(), fixture_cols, 1, fixture_rows, fixture_cols, reference_args);

            stream_parity_check = check(
                reference_run &&
                    reference_dequantize &&
                    quantized_fixture == reference_quantized &&
                    dequantized_fixture == reference_dequantized &&
                    fixture_meta.theta == reference_meta.theta &&
                    same_outliers(fixture_meta.outliers, reference_meta.outliers),
                "streamed EXSIA diverged from reference two-phase orchestration");
        }

        if (case_name == "scratch-reuse" || case_name == "all") {
            const auto same_masks = [](const ExSIAState &lhs, const ExSIAState &rhs) {
                if (lhs.stripe.size() != rhs.stripe.size()) {
                    return false;
                }
                for (size_t i = 0; i < lhs.stripe.size(); ++i) {
                    const auto &left = lhs.stripe[i].outlier_mask;
                    const auto &right = rhs.stripe[i].outlier_mask;
                    if (left.rows != right.rows || left.cols != right.cols ||
                        left.active_word_count() != right.active_word_count()) {
                        return false;
                    }
                    for (size_t word = 0; word < left.active_word_count(); ++word) {
                        if (left.words[word] != right.words[word]) {
                            return false;
                        }
                    }
                }
                return true;
            };

            std::vector<float> second_source(fixture_size, 0.0f);
            for (size_t row = 0; row < fixture_rows; ++row) {
                for (size_t col = 0; col < fixture_cols; ++col) {
                    second_source[row * fixture_cols + col] =
                        ((row + col) % (BLOCK_SIZE + 1) == 0) ? 2048.0f :
                        (col % 3 == 0 ? -3.0f : 0.25f);
                }
            }

            std::vector<elem_t> reused_quantized(fixture_size, 0);
            std::vector<float> reused_dequantized(fixture_size, 0.0f);
            ggml_tensor reused_tensor{};
            reused_tensor.type = GGML_TYPE_F32;
            reused_tensor.data = second_source.data();

            ggml_gemmini_args_t reused_args;
            reused_args.I = fixture_rows;
            reused_args.K = fixture_cols;
            reused_args.A = reused_quantized.data();
            reused_args.sA = fixture_cols;
            reused_args.tile_I = 1;
            auto &reused_meta = reused_args.act_quant.storage().emplace<Meta>();

            const bool reused_run = fixture_exsia.run(reused_meta, &reused_tensor, reused_args);
            const bool reused_dequantize = reused_run && dequantize_activation(
                reused_dequantized.data(), fixture_cols, 1, fixture_rows, fixture_cols, reused_args);

            std::vector<elem_t> fresh_quantized(fixture_size, 0);
            std::vector<float> fresh_dequantized(fixture_size, 0.0f);
            ggml_tensor fresh_tensor{};
            fresh_tensor.type = GGML_TYPE_F32;
            fresh_tensor.data = second_source.data();

            ggml_gemmini_args_t fresh_args;
            fresh_args.I = fixture_rows;
            fresh_args.K = fixture_cols;
            fresh_args.A = fresh_quantized.data();
            fresh_args.sA = fixture_cols;
            fresh_args.tile_I = 1;
            auto &fresh_meta = fresh_args.act_quant.storage().emplace<Meta>();

            ExSIA fresh_exsia;
            const bool fresh_run = fresh_exsia.run(fresh_meta, &fresh_tensor, fresh_args);
            const bool fresh_dequantize = fresh_run && dequantize_activation(
                fresh_dequantized.data(), fixture_cols, 1, fixture_rows, fixture_cols, fresh_args);

            scratch_reuse_check = check(
                reused_run && fresh_run &&
                    reused_dequantize && fresh_dequantize &&
                    reused_quantized == fresh_quantized &&
                    reused_dequantized == fresh_dequantized &&
                    reused_meta.theta == fresh_meta.theta &&
                    same_outliers(reused_meta.outliers, fresh_meta.outliers) &&
                    same_masks(fixture_exsia.state(), fresh_exsia.state()),
                "scratch reuse diverged from a fresh ExSIA run");
        }

        reference_check = reference_check && multi_stripe_check && dequantize_check &&
                          outlier_order_check && mask_layout_check && mask_boundary_check &&
                          ((case_name != "stream-parity" && case_name != "all") || stream_parity_check) &&
                          ((case_name != "scratch-reuse" && case_name != "all") || scratch_reuse_check);
        if (!reference_check && failure_reason.empty()) {
            failure_reason = "reference checks failed";
        }
    }

    if (!compiled_tau_check || !quantizer_check || !detector_check || !invalid_zero_dimension_check ||
        !meta_sigma_check ||
        !finite_input_policy_check ||
        !shift_extremes_check ||
        (case_name == "stream-invalid-shape" && !stream_invalid_shape_check) ||
        !initial_state_check || (reference_case_checked && !reference_check)) {
        return fail(failure_reason.empty() ? "harness checks failed" : failure_reason.c_str());
    }

    if (!write_artifact("pass")) {
        return 1;
    }
    return 0;
}
