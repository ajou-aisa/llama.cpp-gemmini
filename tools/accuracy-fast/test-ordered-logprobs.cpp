#include "ordered-logprobs.hpp"

#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>

using accuracy_fast::LogitsRow;
using Result = std::pair<double, bool>;

// Frozen scalar scorer formula is the independent numerical oracle.
static Result scalar_oracle(const float * logits, int size, int target) {
    const auto best = std::max_element(logits, logits + size);
    double sum = 0;
    for (int i = 0; i < size; ++i) sum += std::exp(double(logits[i]) - *best);
    const double result = double(logits[target]) - *best - std::log(sum);
    if (!std::isfinite(result)) throw std::runtime_error("Nonfinite log probability");
    return {result, best - logits == target};
}

static void require(bool value, const char * reason) {
    if (!value) throw std::runtime_error(reason);
}

static bool same_bits(double first, double second) {
    return std::memcmp(&first, &second, sizeof(double)) == 0;
}

struct Fixture {
    std::vector<float> logits;
    std::vector<LogitsRow> rows;

    Fixture(int vocabulary, int count) : logits(size_t(vocabulary) * count) {
        uint32_t state = 0x12345678;
        for (float & value : logits) {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            value = float(int(state % 64001) - 32000) / 512;
        }
        for (int i = 0; i < count; ++i) rows.push_back({logits.data() + size_t(i) * vocabulary,
                                                       (i * 7919) % vocabulary});
    }
};

static void verify_bitwise(int vocabulary, int count) {
    Fixture fixture(vocabulary, count);
    std::vector<Result> expected;
    double expected_total = -0.123456789;
    for (const auto & row : fixture.rows) {
        expected.push_back(scalar_oracle(row.logits, vocabulary, row.target));
        expected_total += expected.back().first;
    }
    for (const char * serial : {"0", "1"}) {
        setenv("ACCURACY_FAST_LOGPROBS_SERIAL", serial, 1);
        const auto actual = accuracy_fast::ordered_log_probabilities(fixture.rows, vocabulary);
        require(actual.size() == expected.size(), "Row count changed");
        double total = -0.123456789;
        for (size_t i = 0; i < actual.size(); ++i) {
            require(same_bits(actual[i].first, expected[i].first), "Log probability differs bitwise");
            require(actual[i].second == expected[i].second, "Greedy prediction differs");
            total += actual[i].first;
        }
        require(same_bits(total, expected_total), "Ordered continuation total differs bitwise");
    }
    std::cout << "PASS bitwise vocabulary=" << vocabulary << " rows=" << count << '\n';
}

static void verify_ties_and_nonfinite() {
    constexpr int vocabulary = 128256;
    std::vector<float> logits(vocabulary, 1000);
    for (const char * serial : {"0", "1"}) {
        setenv("ACCURACY_FAST_LOGPROBS_SERIAL", serial, 1);
        const auto ties = accuracy_fast::ordered_log_probabilities({{logits.data(), 0}, {logits.data(), 1}}, vocabulary);
        require(ties[0].second && !ties[1].second, "Greedy tie must select first vocabulary entry");
        require(same_bits(ties[0].first, -std::log(double(vocabulary))), "Equal logits probability changed");
        for (float invalid : {std::numeric_limits<float>::infinity(),
                              -std::numeric_limits<float>::infinity(),
                              std::numeric_limits<float>::quiet_NaN()}) {
            logits[1] = invalid;
            bool failed = false;
            try {
                accuracy_fast::ordered_log_probabilities({{logits.data(), 0}, {logits.data(), 1}}, vocabulary);
            } catch (const std::runtime_error & error) {
                failed = std::string(error.what()) == "Nonfinite log probability";
            }
            require(failed, "Nonfinite row must fail outside parallel region");
            logits[1] = 1000;
        }
    }
    std::cout << "PASS greedy ties and nonfinite rejection in both modes\n";
}

static void benchmark(int vocabulary, int count) {
    Fixture fixture(vocabulary, count);
    double timings[2]{};
    volatile double consumed = 0;
    for (int mode = 0; mode < 2; ++mode) {
        setenv("ACCURACY_FAST_LOGPROBS_SERIAL", mode == 0 ? "1" : "0", 1);
        accuracy_fast::ordered_log_probabilities(fixture.rows, vocabulary);
        std::vector<double> samples;
        for (int repetition = 0; repetition < 7; ++repetition) {
            const auto start = std::chrono::steady_clock::now();
            const auto result = accuracy_fast::ordered_log_probabilities(fixture.rows, vocabulary);
            samples.push_back(std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start).count());
            consumed = consumed + result.back().first;
        }
        std::sort(samples.begin(), samples.end());
        timings[mode] = samples[samples.size() / 2];
    }
    std::cout << "BENCH vocabulary=" << vocabulary << " rows=" << count
              << " serial_median_ms=" << timings[0] << " parallel_median_ms=" << timings[1]
              << " speedup=" << timings[0] / timings[1] << " checksum=" << consumed << '\n';
}

int main() {
    try {
        for (int vocabulary : {128256, 151936}) {
            for (int rows : {0, 1, 2, 7, 64, 512}) verify_bitwise(vocabulary, rows);
        }
        verify_ties_and_nonfinite();
        for (int vocabulary : {128256, 151936}) {
            for (int rows : {1, 2, 8, 64}) benchmark(vocabulary, rows);
        }
        std::cout << "PASS all numerical checks\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "FAIL " << error.what() << '\n';
        return 1;
    }
}
