#pragma once

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

namespace accuracy_fast {

struct LogitsRow {
    const float * logits;
    int target;
};

inline std::pair<double, bool> raw_log_probability(const float * logits, int size, int target) noexcept {
    const auto best = std::max_element(logits, logits + size);
    double sum = 0;
    for (int i = 0; i < size; ++i) sum += std::exp(double(logits[i]) - *best);
    const double result = double(logits[target]) - *best - std::log(sum);
    return {result, best - logits == target};
}

inline std::pair<double, bool> log_probability(const float * logits, int size, int target) {
    const auto result = raw_log_probability(logits, size, target);
    if (!std::isfinite(result.first)) throw std::runtime_error("Nonfinite log probability");
    return result;
}

inline std::vector<std::pair<double, bool>> ordered_log_probabilities(
        const std::vector<LogitsRow> & rows, int vocabulary) {
    std::vector<std::pair<double, bool>> results(rows.size());
    const char * serial = std::getenv("ACCURACY_FAST_LOGPROBS_SERIAL");
    const bool parallel = rows.size() >= 2 && vocabulary >= 4096 &&
        !(serial && std::strcmp(serial, "1") == 0);
    // Each vocabulary sum stays serial; only independent token rows share workers.
    #pragma omp parallel for if(parallel) num_threads(6) schedule(static)
    for (size_t i = 0; i < rows.size(); ++i) {
        results[i] = raw_log_probability(rows[i].logits, vocabulary, rows[i].target);
    }
    for (const auto & result : results) {
        if (!std::isfinite(result.first)) throw std::runtime_error("Nonfinite log probability");
    }
    return results;
}

} // namespace accuracy_fast
