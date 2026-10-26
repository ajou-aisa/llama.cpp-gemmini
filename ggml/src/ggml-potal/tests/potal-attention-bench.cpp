// Opt-in H3-7B5 kernel timing, not inference or FPGA-only timing.
// Compile this TU instead of matmul/attention.cpp so replay calls the exact
// checkpoint's checked reconstruction helper, not a copied implementation.
// POTAL_B5_SHARED_BASELINE is target-local and selects the legacy contract.
#ifndef POTAL_B5_ATTENTION_SOURCE
#define POTAL_B5_ATTENTION_SOURCE "../matmul/attention.cpp"
#endif
#include POTAL_B5_ATTENTION_SOURCE
#include "ggml-cpp.h"
// Point the detached baseline target at the current independent oracle. The
// baseline checkpoint's old boolean oracle argument had different semantics.
#ifndef POTAL_B5_GOLDEN_HEADER
#define POTAL_B5_GOLDEN_HEADER "support/attention-golden.hpp"
#endif
#include POTAL_B5_GOLDEN_HEADER

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {
namespace attention = ggml::potal::attention;
namespace golden = potal_test::attention_golden;
using namespace ggml::potal;
using clock_type = std::chrono::steady_clock;

#ifdef POTAL_B5_SHARED_BASELINE
constexpr bool shared_baseline = true;
constexpr const char * contract = "baseline_shared_acc";
#else
constexpr bool shared_baseline = false;
constexpr const char * contract = "independent_k_limb_acc";
#endif

enum class mode { none, bench, profile, reconstruct };

struct options {
    mode selected = mode::none;
    std::string name = "all";
    size_t repeat = 7;
};

bool parse(int argc, char ** argv, options & out) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--bench" || arg == "--profile" || arg == "--reconstruct") {
            if (out.selected != mode::none) return false;
            out.selected = arg == "--bench" ? mode::bench :
                           arg == "--profile" ? mode::profile : mode::reconstruct;
        } else if (arg == "--case" && i + 1 < argc) {
            out.name = argv[++i];
        } else if (arg == "--repeat" && i + 1 < argc) {
            const char * text = argv[++i];
            char * end = nullptr;
            errno = 0;
            const unsigned long long value = std::strtoull(text, &end, 10);
            if (errno || !*text || *text == '-' || *end || value == 0 || value > SIZE_MAX) return false;
            out.repeat = size_t(value);
        } else return false;
    }
    return out.selected != mode::none &&
        (out.name == "all" || out.name == "prefill" || out.name == "decode");
}

double milliseconds(clock_type::time_point start) {
    return std::chrono::duration<double, std::milli>(clock_type::now() - start).count();
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const size_t middle = values.size() / 2;
    return values.size() % 2 ? values[middle] : (values[middle - 1] + values[middle]) / 2;
}

void summary(const char * name, const char * boundary, const std::vector<double> & values) {
    if (values.size() < 2) {
        std::printf("POTAL_B5_SUMMARY case=%s contract=%s boundary=%s samples=%zu "
                    "median_mad=not_reported aggregation=external_required\n",
                    name, contract, boundary, values.size());
        return;
    }
    const double center = median(values);
    std::vector<double> deviations;
    deviations.reserve(values.size());
    for (double value : values) deviations.push_back(std::abs(value - center));
    std::printf("POTAL_B5_SUMMARY case=%s contract=%s boundary=%s samples=%zu median_ms=%.9f mad_ms=%.9f\n",
                name, contract, boundary, values.size(), center, median(std::move(deviations)));
}

// Reject diagnostic builds rather than presenting their overhead as baseline.
bool diagnostics_off() {
#if (defined(LOG_DEBUG) && LOG_DEBUG) || (defined(LOG_CYCLE) && LOG_CYCLE) || \
    (defined(CYCLE_DETAIL) && CYCLE_DETAIL) || (defined(CYCLE_SIM) && CYCLE_SIM) || \
    (defined(LOG_DUMP) && LOG_DUMP) || (defined(LOG_DUMP_SCALE) && LOG_DUMP_SCALE) || \
    defined(POTAL_PLANNING_DIAGNOSTICS) || \
    (defined(EXSIA_VALIDATION) && EXSIA_VALIDATION) || \
    (defined(EXSIA_BRANCH_COUNTS_ENABLED) && EXSIA_BRANCH_COUNTS_ENABLED)
    return false;
#else
    for (const char * key : {"POTAL_RAW_TRACE", "POTAL_PAIRED_PROFILE", "GEMMINI_EXSIA_PROFILE"}) {
        const char * value = std::getenv(key);
        if (value && *value && std::strcmp(value, "0") != 0) return false;
    }
    return true;
#endif
}

bool finished(const attention::result & result) {
    return result.status == POTAL_OK && !result.fatal && result.jobs > 0 && result.loops > 0 &&
           result.fences == result.loops && result.scale_writes > 0 &&
           result.scale_writes == result.scale_releases;
}

uint64_t qk_jobs(const attention::result & result) {
#ifdef POTAL_B5_SHARED_BASELINE
    return result.paired_k_jobs;
#else
    return result.qk_limb_jobs;
#endif
}

struct raw_job {
    size_t rows = 0, columns = 0;
    std::vector<int32_t> c;
    double elapsed_ms = 0;
    bool completed = false;
};

struct sample {
    attention::result result;
    double elapsed_ms = 0;
    bool exact = false;
};

void report_sample(const char * name, size_t index, const char * boundary, const sample & value) {
    const auto & r = value.result;
    std::printf("POTAL_B5_SAMPLE case=%s contract=%s sample=%zu boundary=%s outer_timer_measured=%d "
                "cpu_elapsed_ms=%.9f "
                "exact=%d status=%d fatal=%d jobs=%llu qk_jobs=%llu loop_ws=%llu fences=%llu "
                "scale_writes=%llu scale_releases=%llu tlb_flushes=%llu stage=%s\n",
                name, contract, index, boundary, std::strcmp(boundary, "attention_execute") == 0,
                value.elapsed_ms, value.exact, int(r.status), r.fatal,
                (unsigned long long)r.jobs, (unsigned long long)qk_jobs(r),
                (unsigned long long)r.loops, (unsigned long long)r.fences,
                (unsigned long long)r.scale_writes, (unsigned long long)r.scale_releases,
                (unsigned long long)r.tlb_flushes, r.stage ? r.stage : "none");
}

// Replay only completed hardware raw C, using this checkpoint's production
// helper and order. This is a separate CPU microbenchmark, never an execution
// substitute or a subtraction from end-to-end wall time.
bool replay(const std::vector<raw_job> & jobs, std::vector<int64_t> & out,
            std::vector<int64_t> & stationary) {
    try {
        std::fill(out.begin(), out.end(), 0);
#ifdef POTAL_B5_SHARED_BASELINE
        (void)stationary;
        for (size_t q = 0; q < jobs.size(); ++q)
            for (size_t x = 0; x < out.size(); ++x)
                attention::reconstruct(out[x], jobs[q].c[x], q);
#else
        for (size_t q = 0; q < jobs.size() / 2; ++q) {
            std::fill(stationary.begin(), stationary.end(), 0);
            for (size_t pass = 0; pass < 2; ++pass)
                for (size_t x = 0; x < out.size(); ++x)
                    attention::reconstruct(stationary[x], jobs[2 * q + pass].c[x], pass);
            for (size_t x = 0; x < out.size(); ++x)
                attention::reconstruct(out[x], stationary[x], q);
        }
#endif
        return true;
    } catch (potal_status) {
        return false;
    }
}

bool run(const char * name, size_t m, size_t n, const options & opts) {
    constexpr size_t k = 64;
    ggml_init_params params{8u * 1024 * 1024, nullptr, false};
    ggml_context_ptr context(ggml_init(params));
    if (!context) return false;
    auto * weight = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, k, n);
    auto * input = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, k, m);
    // Fixed, exactly representable tensors, identical in both checkpoint builds.
    // Mixed signs, several Q digits, negative K-high, and two scale blocks.
    for (size_t x = 0; x < k; ++x)
        for (size_t j = 0; j < n; ++j)
            static_cast<float *>(weight->data)[j * k + x] =
                std::ldexp(float(int((x * 17 + j * 11) % 63) - 31), x < 32 ? -5 : -8);
    for (size_t i = 0; i < m; ++i)
        for (size_t x = 0; x < k; ++x)
            static_cast<float *>(input->data)[i * k + x] =
                float(int((x * 13 + i * 7) % 2047) - 1023) / 1024;
    auto * output = ggml_mul_mat(context.get(), weight, input);
    ggml_gemmini_set_attention_role(output, GGML_GEMMINI_ATTENTION_QK);
    ggml_mul_mat_set_prec(output, GGML_PREC_F32);
    attention::stationary w;
    attention::stream a;
    potal_tile_plan stripes;
    if (potal_tile_select(m, n, k, 2, &stripes) != POTAL_OK ||
        stripes.tile_I * POTAL_HW_DIM < m ||
        attention::prepare_stationary(*weight, 0, 0, attention::role::qk, w) != POTAL_OK ||
        attention::prepare_stream(*input, 0, 0, 0, m, attention::role::qk, a) != POTAL_OK) return false;
    std::vector<int64_t> integer;
    if (!golden::evaluate(a, w, attention::role::qk, integer, shared_baseline)) return false;
    std::vector<float> expected(integer.size());
    for (size_t i = 0; i < m; ++i)
        for (size_t j = 0; j < n; ++j)
            expected[i * n + j] = float(std::ldexp(double(integer[i * n + j]), a.scale + int(w.base[j])));
    const size_t job_count = a.limb_count * (shared_baseline ? 1 : 2);
    std::printf("POTAL_B5_CASE case=%s M=%zu N=%zu K=%zu q_limbs=%zu expected_qk_jobs=%zu repeat=%zu\n",
                name, m, n, k, a.limb_count, job_count, opts.repeat);
    std::vector<std::vector<int32_t>> expected_raw;
    if (!shared_baseline && opts.selected != mode::bench)
        for (size_t q = 0; q < a.limb_count; ++q)
            for (size_t pass = 0; pass < 2; ++pass)
                expected_raw.push_back(golden::qk_pass(a, w, q, pass));
    std::vector<sample> samples(opts.repeat);
    // Primary mode has no intercepting runner, raw capture, or per-call timer.
    const bool instrumented = opts.selected != mode::bench;
    std::vector<std::vector<raw_job>> captures(instrumented ? opts.repeat : 0);
    for (auto & capture : captures) {
        capture.resize(job_count);
        for (auto & job : capture) {
            job.rows = m; job.columns = n;
            job.c.resize(m * n);
        }
    }
    attention::result warmup;
    issuer_owner owner;
    std::string why;
    potal_status status = POTAL_OK;
    size_t executed = 0;
    const bool started = owner.start([&](const owner_runners & runners) {
        // One untimed kernel warmup; no inference or separate owner startup
        // occurs within a measured sample.
        warmup = attention::execute(output, runners.planned);
        if (!finished(warmup) ||
            std::memcmp(output->data, expected.data(), expected.size() * sizeof(float)) != 0)
            return consume_outcome{warmup.fatal,
                warmup.status == POTAL_OK ? POTAL_ERR_BAD_RESPONSE : warmup.status, "benchmark_warmup"};
        for (size_t s = 0; s < samples.size(); ++s) {
            auto & measured = samples[s];
            if (!instrumented) {
                const auto start = clock_type::now();
                measured.result = attention::execute(output, runners.planned);
                measured.elapsed_ms = milliseconds(start);
            } else {
                size_t next = 0;
                measured.result = attention::execute(output, [&](const potal_stripe_gemm_desc & d) {
                    planned_result refused;
                    refused.status = POTAL_ERR_BAD_RESPONSE; refused.stage = "benchmark_capture";
                    if (next == captures[s].size() || d.m != m || d.n != n ||
                        d.k != k * (shared_baseline ? 2 : 1)) return refused;
                    auto & job = captures[s][next++];
                    // This optional interval contains Scale, issue, Fence and
                    // embedded final readback. It is CPU elapsed, not FPGA time.
                    clock_type::time_point start{};
                    if (opts.selected == mode::profile) start = clock_type::now();
                    auto completed = runners.planned(d);
                    if (opts.selected == mode::profile) job.elapsed_ms = milliseconds(start);
                    if (!completed.ok) return completed;
                    if (!completed.report.loop_ws ||
                        completed.report.fences != completed.report.loop_ws ||
                        completed.report.scale_writes != completed.report.scale_releases) return refused;
                    for (size_t i = 0; i < m; ++i)
                        for (size_t j = 0; j < n; ++j) job.c[i * n + j] = d.c[i * d.ldc + j];
                    job.completed = true;
                    return completed;
                });
            }
            ++executed;
            // Validation is outside the primary timer and performed every time.
            measured.exact = finished(measured.result) && qk_jobs(measured.result) == job_count &&
                std::memcmp(output->data, expected.data(), expected.size() * sizeof(float)) == 0;
            if (!measured.exact)
                return consume_outcome{measured.result.fatal,
                    measured.result.status == POTAL_OK ? POTAL_ERR_BAD_RESPONSE : measured.result.status,
                    measured.result.stage ? measured.result.stage : "benchmark_output"};
        }
        return consume_outcome{};
    }, why, status);
    owner.join();
    // No sample printing happens on the measured owner path.
    if (!started || !finished(warmup) || executed != opts.repeat) {
        std::printf("POTAL_B5_FAIL case=%s started=%d start_status=%d warmup_status=%d "
                    "warmup_fatal=%d executed=%zu reason=%s\n",
                    name, started, int(status), int(warmup.status), warmup.fatal, executed, why.c_str());
        for (size_t s = 0; s < executed; ++s) report_sample(name, s, "failed", samples[s]);
        return false;
    }
    std::vector<double> primary, low, high, combined;
    for (size_t s = 0; s < samples.size(); ++s) {
        const auto & measured = samples[s];
        report_sample(name, s, instrumented ? "instrumented_capture" : "attention_execute", measured);
        if (!measured.exact) return false;
        if (!instrumented) { primary.push_back(measured.elapsed_ms); continue; }
        double low_ms = 0, high_ms = 0, combined_ms = 0;
        for (size_t job = 0; job < captures[s].size(); ++job) {
            const auto & raw = captures[s][job];
            if (!raw.completed) return false;
            if (!shared_baseline && raw.c != expected_raw[job]) return false;
            if (opts.selected == mode::profile) {
                std::printf("POTAL_B5_PROFILE case=%s contract=%s sample=%zu q_limb=%zu "
                            "stationary=%s cpu_elapsed_ms=%.9f fpga_only=not_measured "
                            "isolated_readback_dma=unavailable_embedded_in_regular_loop_ws\n",
                            name, contract, s, shared_baseline ? job : job / 2,
                            shared_baseline ? "combined" : job % 2 ? "high" : "low", raw.elapsed_ms);
                if (shared_baseline) combined_ms += raw.elapsed_ms;
                else if (job % 2) high_ms += raw.elapsed_ms;
                else low_ms += raw.elapsed_ms;
            }
        }
        low.push_back(low_ms); high.push_back(high_ms); combined.push_back(combined_ms);
    }
    if (opts.selected == mode::bench) {
        summary(name, "attention_execute_cpu_wall_diagnostics_off", primary);
    } else if (opts.selected == mode::profile) {
        if (shared_baseline) {
            summary(name, "combined_runners_cpu_elapsed", combined);
            std::printf("POTAL_B5_PROFILE case=%s low_high_separate=unavailable_baseline_combined_job\n", name);
        } else {
            summary(name, "low_runners_cpu_elapsed", low);
            summary(name, "high_runners_cpu_elapsed", high);
        }
    } else {
        std::vector<int64_t> reconstructed(integer.size()), stationary(integer.size());
        std::vector<double> elapsed;
        constexpr size_t replays = 4096;
        volatile int64_t checksum = 0;
        for (size_t s = 0; s < captures.size(); ++s) {
            bool ok = true;
            const auto start = clock_type::now();
            for (size_t r = 0; r < replays; ++r) {
                // Compiler barrier only: no CPU or accelerator fence is issued.
                std::atomic_signal_fence(std::memory_order_seq_cst);
                ok = replay(captures[s], reconstructed, stationary) && ok;
                checksum = reconstructed[r % reconstructed.size()];
            }
            const double per_replay = milliseconds(start) / replays;
            if (!ok || reconstructed != integer) return false;
            elapsed.push_back(per_replay);
            std::printf("POTAL_B5_RECONSTRUCT case=%s contract=%s sample=%zu replays=%zu "
                        "cpu_ms_per_replay=%.9f exact=1 attribution=separate_cpu_microbenchmark "
                        "input=actual_completed_hardware_signed32_C\n",
                        name, contract, s, replays, per_replay);
        }
        std::printf("POTAL_B5_REPLAY_BOUNDARY includes=buffer_reset_checked_reconstruct_checksum "
                    "compiler_barrier=signal_fence_only cpu=calling_thread_not_issuer checksum=%lld\n",
                    (long long)checksum);
        summary(name, "production_checked_host_reconstruction_replay", elapsed);
    }
    return true;
}
}

int main(int argc, char ** argv) {
    options opts;
    if (!parse(argc, argv, opts)) {
        std::fprintf(stderr, "Usage: %s (--bench|--profile|--reconstruct) "
                     "[--case all|prefill|decode] [--repeat N>=1; default 7]\n", argv[0]);
        return 2;
    }
    if (!diagnostics_off()) {
        std::fprintf(stderr, "H3-7B5 requires diagnostic compile flags and runtime tracing OFF.\n");
        return 2;
    }
    std::printf("POTAL_B5_BEGIN contract=%s diagnostics=off timer=steady_clock_cpu_wall "
                "owner=existing_hart0_issuer startup=excluded warmup=one_kernel "
                "full_inference_warmup=none isolated_dma=not_measured fpga_cycles=not_measured\n", contract);
    if ((opts.name == "all" || opts.name == "prefill") && !run("prefill", 16, 16, opts)) return 1;
    if ((opts.name == "all" || opts.name == "decode") && !run("decode", 1, 23, opts)) return 1;
    std::printf("POTAL_B5_COMPLETE contract=%s\n", contract);
    return 0;
}
