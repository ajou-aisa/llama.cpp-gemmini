#!/usr/bin/env python3
"""Test actual scoring functions without building llama.cpp or loading a model.

Run: python3 tests/test-perplexity-deterministic.py [perplexity.cpp]
"""

from pathlib import Path
import subprocess
import sys
import tempfile


SOURCE = Path(__file__).resolve().parents[1] / "tools/perplexity/perplexity.cpp"
PREAMBLE = r"""
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
"""
HARNESS = r"""
template <typename T>
static bool exact(const T & a, const T & b) {
    return std::memcmp(&a, &b, sizeof(T)) == 0;
}

static void check(bool ok, const char * what, int count, int threads) {
    if (!ok) {
        std::fprintf(stderr, "%s: tokens=%d threads=%d\n", what, count, threads);
        std::exit(1);
    }
}

int main() {
    const int vocab = 17;
    const int nv = 2*((vocab + 1)/2) + 4;
    for (int count : {0, 1, 255, 4097}) {
        std::vector<int> tokens(count + 1);
        std::vector<float> logits(size_t(count)*vocab), expected_logits(count), expected_probs(count);
        std::vector<uint16_t> expected_dump(size_t(count)*nv, 0);
        for (int i = 0; i <= count; ++i) {
            tokens[i] = (i*7 + 3)%vocab;
        }
        for (int i = 0; i < count; ++i) {
            for (int j = 0; j < vocab; ++j) {
                logits[size_t(i)*vocab + j] = float((i*19 + j*23)%257 - 128)/7;
            }
            if (i%11 == 0) {
                // Equal logits exercise the zero-scale packed-output case.
                std::fill_n(logits.data() + size_t(i)*vocab, vocab, 3.0f);
            } else if (i%29 == 0) {
                // Mix magnitudes to expose changes to double reduction order.
                logits[size_t(i)*vocab + tokens[i+1]] = -1000000.0f;
            }
        }

        double expected_nll = 0, expected_nll2 = 0, expected_dump_nll = 0, expected_dump_nll2 = 0;
        for (int i = 0; i < count; ++i) {
            const auto result = log_softmax(vocab, logits.data() + size_t(i)*vocab, tokens[i+1]);
            const double v = -result.log_softmax;
            expected_nll += v;
            expected_nll2 += v*v;
            expected_logits[i] = result.logit;
            expected_probs[i] = result.prob;
            const double packed_v = log_softmax(vocab, logits.data() + size_t(i)*vocab,
                    expected_dump.data() + size_t(i)*nv, tokens[i+1]);
            expected_dump_nll += packed_v;
            expected_dump_nll2 += packed_v*packed_v;
        }
        // Nonzero prior totals also verify the existing per-pass accumulation boundary.
        expected_nll += 123.25; expected_nll2 += 456.5;
        expected_dump_nll += 123.25; expected_dump_nll2 += 456.5;

        for (int threads : {0, 1, 3, 7}) {
            for (int repeat = 0; repeat < 8; ++repeat) {
                std::vector<std::thread> workers(threads);
                std::vector<float> actual_logits(count), actual_probs(count);
                double nll = 123.25, nll2 = 456.5;
                process_logits(vocab, logits.data(), tokens.data(), count, workers, nll, nll2,
                        actual_logits.data(), actual_probs.data());
                check(exact(nll, expected_nll) && exact(nll2, expected_nll2), "plain NLL", count, threads);
                for (int i = 0; i < count; ++i) {
                    check(exact(actual_logits[i], expected_logits[i]) && exact(actual_probs[i], expected_probs[i]),
                            "token history", count, threads);
                }

                std::vector<uint16_t> actual_dump(size_t(count)*nv, 0);
                std::ostringstream out;
                nll = 123.25; nll2 = 456.5;
                process_logits(out, vocab, logits.data(), tokens.data(), count, workers, actual_dump, nll, nll2);
                check(exact(nll, expected_dump_nll) && exact(nll2, expected_dump_nll2), "packed NLL", count, threads);
                check(actual_dump == expected_dump, "packed probabilities", count, threads);
                check(out.str().size() == expected_dump.size()*sizeof(uint16_t), "packed length", count, threads);
                check(count == 0 || std::memcmp(out.str().data(), expected_dump.data(), out.str().size()) == 0,
                        "packed bytes", count, threads);
            }
        }
    }
    std::puts("PASS: both scorers, 0/1/255/4097 tokens, 1/2/4/8 threads, 8 repetitions, bitwise totals and payloads");
}
"""


def run_scoring_test(source: Path = SOURCE) -> None:
    text = source.read_text()
    # Compile the production definitions, not a copied implementation.
    begin = text.index("struct results_log_softmax {")
    end = text.index("struct kl_divergence_result {", begin)
    with tempfile.TemporaryDirectory(prefix="perplexity-deterministic-") as directory:
        root = Path(directory)
        cpp, binary = root / "scoring.cpp", root / "scoring"
        cpp.write_text(PREAMBLE + text[begin:end] + HARNESS)
        subprocess.run(["c++", "-std=c++17", "-O2", "-pthread", str(cpp), "-o", str(binary)], check=True)
        subprocess.run([str(binary)], check=True)


def test_perplexity_scoring_deterministic() -> None:
    run_scoring_test()


if __name__ == "__main__":
    run_scoring_test(Path(sys.argv[1]) if len(sys.argv) > 1 else SOURCE)
