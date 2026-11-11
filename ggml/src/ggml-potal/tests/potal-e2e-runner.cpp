// H3-8I-25V: model-scale E2E runner on the PoTal backend.
// Runs a deterministic N-token prompt (one llama_decode) and then M-1 single-token
// greedy decodes, so exactly M tokens are generated; EOS is never tested. Every step
// is timed with CLOCK_MONOTONIC and snapshots the PoTal counters; everything is kept in
// memory and written after the loop (one short stdout line per step for a killed run).
// Timeline marks and the recording gate use the opt-in hooks of ggml-potal.h; they are
// inert unless POTAL_TIMELINE is set.
#include "llama.h"
#include "ggml-potal.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <sched.h>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

enum { ST_INFERENCE = 81, ST_PREFILL = 82, ST_DECODE = 83, ST_TOKEN = 84 };
enum { EV_BEGIN = 1, EV_END = 2, EV_INSTANT = 7 };

uint64_t now_ns() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return uint64_t(ts.tv_sec) * 1000000000ull + uint64_t(ts.tv_nsec);
}

// I25X: target cycle counter (RISC-V rdcycle, user mode allowed on this target); 0 elsewhere.
uint64_t read_cycle() {
#if defined(__riscv)
    uint64_t c;
    asm volatile("rdcycle %0" : "=r"(c));
    return c;
#else
    return 0;
#endif
}

int cur_cpu() { return sched_getcpu(); }

struct Options {
    std::string model, out;
    int n_prompt = 1024, n_gen = 1024, n_ctx = -1, n_ubatch = -1, threads = 1;
    bool trace_prefill = false, check_logits = false, offload_kqv = true;
    std::set<int> trace_steps;
};

int to_int(const std::string & key, const std::string & value) {
    size_t used = 0;
    const int v = std::stoi(value, &used);
    if (used != value.size()) throw std::runtime_error("bad integer for " + key + ": " + value);
    return v;
}

bool to_bool(const std::string & key, const std::string & value) {
    if (value == "0") return false;
    if (value == "1") return true;
    throw std::runtime_error(key + " expects 0 or 1");
}

Options parse(int argc, char ** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--help" || key == "-h") {
            std::puts("potal-e2e-runner --model GGUF --out DIR [--prompt-tokens N=1024] [--gen-tokens M=1024]\n"
                      "  [--ctx C=N+M] [--ubatch U=N] [--threads T=1] [--trace-prefill 0|1]\n"
                      "  [--trace-steps i,j,...] [--check-logits 0|1] [--offload-kqv 0|1]");
            std::exit(0);
        }
        if (++i == argc) throw std::runtime_error("missing value for " + key);
        const std::string v = argv[i];
        if (key == "--model") o.model = v;
        else if (key == "--out") o.out = v;
        else if (key == "--prompt-tokens") o.n_prompt = to_int(key, v);
        else if (key == "--gen-tokens") o.n_gen = to_int(key, v);
        else if (key == "--ctx") o.n_ctx = to_int(key, v);
        else if (key == "--ubatch") o.n_ubatch = to_int(key, v);
        else if (key == "--threads") o.threads = to_int(key, v);
        else if (key == "--trace-prefill") o.trace_prefill = to_bool(key, v);
        else if (key == "--check-logits") o.check_logits = to_bool(key, v);
        else if (key == "--offload-kqv") o.offload_kqv = to_bool(key, v);
        else if (key == "--trace-steps") {
            std::stringstream ss(v);
            std::string item;
            while (std::getline(ss, item, ','))
                if (!item.empty()) o.trace_steps.insert(to_int(key, item));
        } else throw std::runtime_error("unknown option: " + key);
    }
    if (o.n_ctx < 0) o.n_ctx = o.n_prompt + o.n_gen;
    if (o.n_ubatch < 0) o.n_ubatch = o.n_prompt;
    if (o.model.empty() || o.out.empty() || o.n_prompt < 1 || o.n_gen < 1 || o.n_ubatch < 1 ||
        o.threads < 1 || o.n_ctx < o.n_prompt + o.n_gen - 1)
        throw std::runtime_error("invalid options (need --model, --out, N,M,U,T >= 1, C >= N+M-1); see --help");
    return o;
}

// Per-step counter deltas (fields of ggml_backend_potal_stats, plus graph_count).
#define E2E_COUNTERS(X) \
    X(mul_mat_nodes) X(rejected_nodes) X(hp1_nodes) X(hp1_completed) X(hp1_failed) \
    X(gemm_failed) X(stripe_pipeline_failed) X(compensation_failed) \
    X(attention_qk_nodes) X(attention_pv_nodes) X(attention_completed) X(attention_failed) \
    X(attention_strict_nodes) X(attention_issuer_cpu_work) \
    X(prepares_on_issuer_thread) X(prepares_on_issuer_cpu) X(paired_fallback_stripes) \
    X(weight_cache_hits) X(weight_cache_misses) X(weight_cache_bypasses) X(weight_cache_uncached)
#define E2E_TIMES(X) X(attention_wall_s) X(weight_pack_s)

struct Counters {
#define E2E_FIELD(f) uint64_t f = 0;
    E2E_COUNTERS(E2E_FIELD)
#undef E2E_FIELD
#define E2E_TFIELD(f) double f = 0;
    E2E_TIMES(E2E_TFIELD)
#undef E2E_TFIELD
    uint64_t graphs = 0;
};

Counters snapshot(bool & ok) {
    ggml_backend_potal_stats st{};
    Counters c;
    ok = ggml_backend_potal_get_last_stats(&st);
#define E2E_COPY(f) c.f = st.f;
    E2E_COUNTERS(E2E_COPY)
    E2E_TIMES(E2E_COPY)
#undef E2E_COPY
    c.graphs = ggml_backend_potal_graph_count();
    return c;
}

Counters delta(const Counters & a, const Counters & b) {
    Counters d;
#define E2E_SUB(f) d.f = b.f - a.f;
    E2E_COUNTERS(E2E_SUB)
    E2E_TIMES(E2E_SUB)
#undef E2E_SUB
    d.graphs = b.graphs - a.graphs;
    return d;
}

void add(Counters & total, const Counters & d) {
#define E2E_ADD(f) total.f += d.f;
    E2E_COUNTERS(E2E_ADD)
    E2E_TIMES(E2E_ADD)
#undef E2E_ADD
    total.graphs += d.graphs;
}

// Counters whose increase makes the run invalid.
bool failed(const Counters & d) {
    return d.rejected_nodes || d.hp1_failed || d.gemm_failed || d.stripe_pipeline_failed ||
           d.compensation_failed || d.attention_failed;
}

struct Step {
    uint64_t t_start = 0, t_end = 0, t_token = 0;
    int32_t token = -1;
    int context_len = 0;
    Counters d;
    int logits_finite = -1; // -1 not checked
    const char * status = "not_run";
    // I25X: cycle_end/cpu_end are taken where t_token is taken.
    uint64_t cycle_start = 0, cycle_end = 0;
    int cpu_start = -1, cpu_end = -1;
    int traced = 0;
    uint64_t cycles() const { return cycle_end >= cycle_start && cycle_end ? cycle_end - cycle_start : 0; }
};

// Median of a vector (empty -> caller checks).
uint64_t median_u64(std::vector<uint64_t> v) {
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return n % 2 ? v[n / 2] : (v[n / 2 - 1] + v[n / 2]) / 2;
}

std::ofstream open_out(const std::filesystem::path & path) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot write " + path.string());
    return f;
}

} // namespace

int main(int argc, char ** argv) try {
    const Options o = parse(argc, argv);
    std::filesystem::create_directories(o.out);
    const bool timeline = std::getenv("POTAL_TIMELINE") != nullptr;

    ggml_backend_load_all();
    llama_backend_init();

    llama_model * model = llama_model_load_from_file(o.model.c_str(), llama_model_default_params());
    if (!model) throw std::runtime_error("model loading failed: " + o.model);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    std::vector<llama_token> prompt(o.n_prompt);
    for (int i = 0; i < o.n_prompt; ++i) {
        prompt[i] = llama_token((uint64_t(i) * 7919u + 13u) % 150000u);
        if (prompt[i] >= n_vocab) throw std::runtime_error("prompt id exceeds vocab");
    }

    auto cp = llama_context_default_params();
    cp.n_ctx = uint32_t(o.n_ctx);
    cp.n_batch = uint32_t(std::max(o.n_prompt, o.n_ubatch));
    cp.n_ubatch = uint32_t(o.n_ubatch);
    cp.n_threads = cp.n_threads_batch = o.threads;
    cp.flash_attn = false;
    cp.offload_kqv = o.offload_kqv;
    cp.no_perf = true;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) throw std::runtime_error("context initialization failed");
    std::printf("E2E_MODEL_LOADED vocab=%d n_ctx=%u n_batch=%u n_ubatch=%u threads=%d\n", n_vocab,
                llama_n_ctx(ctx), llama_n_batch(ctx), llama_n_ubatch(ctx), o.threads);

    bool stats_ok = false;
    Counters prev = snapshot(stats_ok);
    if (!stats_ok) throw std::runtime_error("no live PoTal backend (ggml_backend_potal_get_last_stats false)");

    llama_sampler * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    std::vector<Step> steps(o.n_gen);
    std::vector<llama_token> tokens(o.n_gen, -1);
    const char * run_status = "ok";
    int done = 0;
    llama_token input = 0;

    std::puts("E2E_BEGIN");
    std::fflush(stdout);
    const uint64_t t_run_start = now_ns();
    ggml_backend_potal_timeline_set_recording(true);
    ggml_backend_potal_timeline_mark(ST_INFERENCE, EV_BEGIN, 0, uint64_t(o.n_prompt));
    for (int s = 0; s < o.n_gen; ++s) {
        Step & st = steps[s];
        const bool prefill = s == 0;
        st.traced = (prefill ? o.trace_prefill : o.trace_steps.count(s) != 0) ? 1 : 0;
        ggml_backend_potal_timeline_set_recording(st.traced != 0);
        llama_batch batch = prefill ? llama_batch_get_one(prompt.data(), o.n_prompt)
                                    : llama_batch_get_one(&input, 1);
        ggml_backend_potal_timeline_mark(prefill ? ST_PREFILL : ST_DECODE, EV_BEGIN, uint32_t(s), 0);
        st.t_start = now_ns();
        st.cycle_start = read_cycle();
        st.cpu_start = cur_cpu();
        const int rc = llama_decode(ctx, batch);
        llama_synchronize(ctx);
        st.t_end = now_ns();
        ggml_backend_potal_timeline_mark(prefill ? ST_PREFILL : ST_DECODE, EV_END, uint32_t(s), 0);
        st.context_len = o.n_prompt + s;
        const Counters cur = snapshot(stats_ok);
        st.d = delta(prev, cur);
        prev = cur;
        done = s + 1;
        if (rc != 0) {
            st.status = "decode_failed";
            run_status = "decode_failed";
            std::printf("E2E_STEP s=%d decode_rc=%d cycles=0 traced=%d\n", s, rc, st.traced);
            std::fflush(stdout);
            break;
        }
        const llama_token tok = llama_sampler_sample(smpl, ctx, -1);
        st.t_token = now_ns();
        st.cycle_end = read_cycle();
        st.cpu_end = cur_cpu();
        ggml_backend_potal_timeline_mark(ST_TOKEN, EV_INSTANT, uint32_t(s), uint64_t(tok));
        st.token = tok;
        tokens[s] = tok;
        input = tok;
        st.status = "ok";
        if (o.check_logits) {
            const float * row = llama_get_logits_ith(ctx, -1);
            st.logits_finite = row != nullptr;
            for (int i = 0; row && i < n_vocab; ++i)
                if (!std::isfinite(row[i])) { st.logits_finite = 0; break; }
            if (!st.logits_finite) { st.status = "nonfinite_logits"; run_status = "nonfinite_logits"; }
        }
        if (!stats_ok) { st.status = "stats_unavailable"; run_status = "stats_unavailable"; }
        else if (failed(st.d)) { st.status = "counters_failed"; run_status = "counters_failed"; }
        std::printf("E2E_STEP s=%d latency_ns=%llu token=%d cycles=%llu traced=%d\n", s,
                    static_cast<unsigned long long>(st.t_end - st.t_start), tok,
                    static_cast<unsigned long long>(st.cycles()), st.traced);
        if (prefill)
            std::printf("E2E_PREFILL_DONE ttft_ns=%llu\n",
                        static_cast<unsigned long long>(st.t_token - st.t_start));
        std::fflush(stdout);
        if (std::string(st.status) != "ok") break;
    }
    ggml_backend_potal_timeline_set_recording(true);
    ggml_backend_potal_timeline_mark(ST_INFERENCE, EV_END, 0, uint64_t(done));
    const uint64_t t_run_end = now_ns();

    const std::filesystem::path dir(o.out);
    Counters total;
    {
        auto f = open_out(dir / "steps.csv");
        f << "step,phase,context_len,t_start_ns,t_end_ns,t_token_ns,latency_ns,token_id";
#define E2E_HDR(x) f << "," #x;
        E2E_COUNTERS(E2E_HDR)
#undef E2E_HDR
#define E2E_THDR(x) f << "," #x "_ns";
        E2E_TIMES(E2E_THDR)
#undef E2E_THDR
        f << ",graph_count,logits_finite,status";
        f << ",cycle_start,cycle_end,cycles,cpu_start,cpu_end,traced\n";
        for (int s = 0; s < done; ++s) {
            const Step & st = steps[s];
            add(total, st.d);
            f << s << ',' << (s == 0 ? "prefill" : "decode") << ',' << st.context_len << ','
              << st.t_start << ',' << st.t_end << ',' << st.t_token << ',' << (st.t_end - st.t_start)
              << ',' << st.token;
#define E2E_VAL(x) f << ',' << st.d.x;
            E2E_COUNTERS(E2E_VAL)
#undef E2E_VAL
#define E2E_TVAL(x) f << ',' << std::llround(st.d.x * 1e9);
            E2E_TIMES(E2E_TVAL)
#undef E2E_TVAL
            f << ',' << st.d.graphs << ',' << st.logits_finite << ',' << st.status;
            f << ',' << st.cycle_start << ',' << st.cycle_end << ',' << st.cycles() << ',' << st.cpu_start
              << ',' << st.cpu_end << ',' << st.traced << '\n';
        }
    }
    int generated = 0;
    {
        auto f = open_out(dir / "tokens.txt");
        for (int s = 0; s < done; ++s)
            if (steps[s].token >= 0) { f << steps[s].token << '\n'; ++generated; }
    }

    bool tl_flushed = false, tl_stats = false;
    uint64_t tl_events = 0, tl_dropped = 0;
    uint32_t tl_overflow = 0, tl_threads = 0;
    if (timeline) {
        tl_flushed = ggml_backend_potal_timeline_flush();
        tl_stats = ggml_backend_potal_timeline_stats(&tl_events, &tl_dropped, &tl_overflow, &tl_threads);
        if (!tl_flushed || !tl_stats || tl_dropped || tl_overflow) {
            if (std::string(run_status) == "ok") run_status = "timeline_invalid";
        }
    }
    {
        uint64_t hw_windows = 0, hw_stripes = 0, hw_dropped = 0, hw_commands = 0;
        bool hw_disabled = false;
        const char * hw_reason = nullptr;
        if (ggml_backend_potal_hw_timeline_stats(&hw_windows, &hw_stripes, &hw_dropped, &hw_commands, &hw_disabled, &hw_reason)) {
            const bool hw_flushed = ggml_backend_potal_hw_timeline_flush();
            std::printf("E2E_HW_TIMELINE windows=%llu stripes=%llu dropped=%llu commands=%llu disabled=%d flushed=%d reason=%s\n",
                        (unsigned long long) hw_windows, (unsigned long long) hw_stripes, (unsigned long long) hw_dropped,
                        (unsigned long long) hw_commands, hw_disabled ? 1 : 0, hw_flushed ? 1 : 0, hw_reason ? hw_reason : "none");
            if ((hw_disabled || hw_dropped != 0 || hw_windows == 0 || !hw_flushed) && std::string(run_status) == "ok")
                run_status = "hw_timeline_invalid";
        }
    }

    const bool ok = std::string(run_status) == "ok" && generated == o.n_gen;
    if (std::string(run_status) == "ok" && !ok) run_status = "incomplete";
    {
        auto f = open_out(dir / "summary.txt");
        f << "model=" << o.model << "\nprompt_tokens=" << o.n_prompt << "\ngen_tokens=" << o.n_gen
          << "\nctx=" << o.n_ctx << "\nubatch=" << o.n_ubatch << "\nn_batch=" << cp.n_batch << "\nctx_effective=" << llama_n_ctx(ctx)
          << "\nn_batch_effective=" << llama_n_batch(ctx)
          << "\nthreads=" << o.threads << "\nprompt_formula=(i*7919+13)%150000"
          << "\ncheck_logits=" << o.check_logits << "\noffload_kqv=" << o.offload_kqv << "\ntrace_prefill=" << o.trace_prefill << "\ntrace_steps=";
        bool first = true;
        for (int s : o.trace_steps) { f << (first ? "" : ",") << s; first = false; }
        f << "\nsteps_run=" << done << "\ntokens_generated=" << generated;
        if (done > 0 && steps[0].t_token)
            f << "\nttft_ns=" << (steps[0].t_token - steps[0].t_start);
        else
            f << "\nttft_ns=NA";
        std::vector<uint64_t> lat;
        for (int s = 1; s <= 8 && s < done; ++s)
            if (steps[s].t_token) lat.push_back(steps[s].t_end - steps[s].t_start);
        if (!lat.empty()) {
            std::sort(lat.begin(), lat.end());
            const size_t n = lat.size();
            const uint64_t med = n % 2 ? lat[n / 2] : (lat[n / 2 - 1] + lat[n / 2]) / 2;
            f << "\ntpot_median8_ns=" << med << "\ntpot_median8_samples=" << n;
        } else {
            f << "\ntpot_median8_ns=NA\ntpot_median8_samples=0";
        }
        uint64_t decode_ns = 0;
        for (int s = 1; s < done; ++s) decode_ns += steps[s].t_end - steps[s].t_start;
        f << "\nprefill_latency_ns=" << (done > 0 ? steps[0].t_end - steps[0].t_start : 0)
          << "\ndecode_latency_total_ns=" << decode_ns << "\nrun_wall_ns=" << (t_run_end - t_run_start);
#define E2E_TOT(x) f << "\ntotal_" #x "=" << total.x;
        E2E_COUNTERS(E2E_TOT)
#undef E2E_TOT
        f << "\ntotal_graph_count=" << total.graphs << "\ntimeline_enabled=" << timeline;
        if (timeline)
            f << "\ntimeline_flushed=" << tl_flushed << "\ntimeline_events=" << tl_events
              << "\ntimeline_dropped=" << tl_dropped << "\ntimeline_overflowed_threads=" << tl_overflow
              << "\ntimeline_threads=" << tl_threads;
        // I25X: cycles/ns measured start-of-decode to token-sampled (the TTFT interval for step 0).
        if (done > 0 && steps[0].t_token) f << "\nttft_cycles=" << steps[0].cycles();
        else f << "\nttft_cycles=NA";
        std::vector<uint64_t> dec_cyc, dec_tr_ns, dec_untr_ns;
        for (int s = 1; s < done; ++s) {
            if (!steps[s].t_token) continue;
            dec_cyc.push_back(steps[s].cycles());
            (steps[s].traced ? dec_tr_ns : dec_untr_ns).push_back(steps[s].t_token - steps[s].t_start);
        }
        if (dec_cyc.empty()) f << "\ntpot_median_cycles=NA"; else f << "\ntpot_median_cycles=" << median_u64(dec_cyc);
        if (dec_tr_ns.empty()) f << "\ntpot_median_traced_ns=NA"; else f << "\ntpot_median_traced_ns=" << median_u64(dec_tr_ns);
        if (dec_untr_ns.empty()) f << "\ntpot_median_untraced_ns=NA"; else f << "\ntpot_median_untraced_ns=" << median_u64(dec_untr_ns);
        f << "\nstatus=" << run_status << "\nexit=" << (ok ? 0 : 1) << '\n';
    }

    std::printf("E2E_DONE status=%s steps=%d tokens=%d\n", run_status, done, generated);
    std::fflush(stdout);
    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return ok ? 0 : 1;
} catch (const std::exception & e) {
    std::fprintf(stderr, "E2E_ERROR %s\n", e.what());
    std::printf("E2E_DONE status=error\n");
    return 2;
}
