// Generates exactly M tokens (EOS is never tested) and stops at the first graph not run fully on PB.
// Default: the whole process on hart 0, one thread. With --issuer-cpu and --lane-cpus (five-core mode):
// a PB issuer thread alone on the CPU of hart 0 runs every Gemmini command, the main thread and the PB
// helpers split the PB CPU stages over the lane CPUs, and ggml's CPU operators use a threadpool on them.
#include "llama.h"
#include "ggml-cpu.h"
#include "ggml-gemmini.h"
#include "../matmul/attention.hpp"
#include "../pb/pb-exec.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <sched.h>
#include <unistd.h>

namespace {

// Gemmini is attached to hart 0 only, and hart numbers need not equal Linux CPU numbers.
void pin_to_hart0() {
#if defined(__riscv)
    std::ifstream info("/proc/cpuinfo");
    std::string line;
    int processor = -1, cpu = -1, value;
    while (cpu < 0 && std::getline(info, line)) {
        if (std::sscanf(line.c_str(), "processor : %d", &value) == 1) processor = value;
        else if (std::sscanf(line.c_str(), "hart : %d", &value) == 1 && value == 0) cpu = processor;
    }
    if (cpu < 0) throw std::runtime_error("cannot find the Linux CPU of hart 0 in /proc/cpuinfo");
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    if (sched_setaffinity(0, sizeof(set), &set) != 0) throw std::runtime_error("cannot pin to the CPU of hart 0");
    std::printf("E2E_PIN hart=0 cpu=%d\n", cpu);
#endif
}

// Linux CPU to hart id from /proc/cpuinfo; empty where the kernel prints no hart lines (x86 hosts).
std::map<int, int> cpu_harts() {
    std::ifstream info("/proc/cpuinfo");
    std::string line;
    std::map<int, int> harts;
    int processor = -1, value;
    while (std::getline(info, line)) {
        if (std::sscanf(line.c_str(), "processor : %d", &value) == 1) processor = value;
        else if (std::sscanf(line.c_str(), "hart : %d", &value) == 1 && processor >= 0) harts[processor] = value;
    }
    return harts;
}

void pin_main(int cpu) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    if (sched_setaffinity(0, sizeof(set), &set) != 0 || sched_getcpu() != cpu)
        throw std::runtime_error("cannot pin the main thread to CPU " + std::to_string(cpu));
}

std::string join(const std::vector<int> & v) {
    std::string s;
    for (const int x : v) s += (s.empty() ? "" : ",") + std::to_string(x);
    return s;
}

namespace pb = ggml::gemmini::pb;

uint64_t lane_busy_ns(const pb::ThreadReport & t) {
    return t.busy_prep_ns + t.busy_scale_ns + t.busy_quant_ns + t.busy_gemm_ns;
}

void write_thread_rows(std::ofstream & f, int step, const std::vector<pb::ThreadReport> & rep) {
    for (const auto & t : rep)
        f << step << ',' << t.role << ',' << t.lane << ',' << t.tid << ',' << t.cpu_assigned << ',' << t.cpu_first << ','
          << t.cpu_last << ',' << t.cpu_mismatch << ',' << t.sessions << ',' << t.phases << ',' << t.thread_cpu_ns << ','
          << t.busy_prep_ns << ',' << t.busy_scale_ns << ',' << t.busy_quant_ns << ',' << t.busy_gemm_ns << ','
          << t.wait_ns << ',' << t.gemm_calls << ',' << t.flushes << ',' << t.packed_a_bytes << ',' << t.packed_b_bytes << ','
          << t.scales_b << ',' << t.zeroed << ',' << t.scaled_elements << ',' << t.quantized_rows << '\n';
}

constexpr int N_TYPES = GGML_GEMMINI_PB_NODE_TYPE_COUNT;
const char * const TYPE_NAMES[N_TYPES] = {
    "attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down", "lm_head", "qk", "pv", "other",
};

using ull = unsigned long long;

uint64_t now_ns() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return uint64_t(ts.tv_sec) * 1000000000ull + uint64_t(ts.tv_nsec);
}

uint64_t read_cycle() {
#if defined(__riscv)
    uint64_t c;
    asm volatile("rdcycle %0" : "=r"(c));
    return c;
#else
    return 0;
#endif
}

int to_int(const std::string & key, const std::string & value) {
    size_t used = 0;
    const int v = std::stoi(value, &used);
    if (used != value.size()) throw std::runtime_error("bad integer for " + key + ": " + value);
    return v;
}

struct Options {
    std::string model, out, prompt_file, dump_logits;
    int n_prompt = 0, n_gen = 0, n_ctx = 2048, n_ubatch = -1, threads = 1;
    int issuer_cpu = -1;            // five-core mode when set
    std::vector<int> lane_cpus;     // ascending; the last one runs the main thread
};

std::vector<int> to_cpu_list(const std::string & key, const std::string & value) {
    std::vector<int> cpus;
    std::stringstream in(value);
    std::string item;
    while (std::getline(in, item, ',')) cpus.push_back(to_int(key, item));
    std::sort(cpus.begin(), cpus.end());
    if (cpus.empty() || std::adjacent_find(cpus.begin(), cpus.end()) != cpus.end() || cpus.front() < 0)
        throw std::runtime_error("bad CPU list for " + key + ": " + value);
    return cpus;
}


Options parse(int argc, char ** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--help" || key == "-h") {
            std::puts("gemmini-pb-e2e-runner --model GGUF --out DIR --prompt-file IDS --prompt-tokens N --gen-tokens M\n"
                      "  [--ctx C=2048] [--ubatch U=N] [--threads T=1]\n"
                      "  [--issuer-cpu CPU_OF_HART0 --lane-cpus A,B,C,D]  five-core mode; threads = number of lane CPUs");
            std::exit(0);
        }
        if (++i == argc) throw std::runtime_error("missing value for " + key);
        const std::string v = argv[i];
        if (key == "--model") o.model = v;
        else if (key == "--out") o.out = v;
        else if (key == "--prompt-file") o.prompt_file = v;
        else if (key == "--prompt-tokens") o.n_prompt = to_int(key, v);
        else if (key == "--gen-tokens") o.n_gen = to_int(key, v);
        else if (key == "--ctx") o.n_ctx = to_int(key, v);
        else if (key == "--ubatch") o.n_ubatch = to_int(key, v);
        else if (key == "--threads") o.threads = to_int(key, v);
        else if (key == "--dump-logits") o.dump_logits = v;
        else if (key == "--issuer-cpu") o.issuer_cpu = to_int(key, v);
        else if (key == "--lane-cpus") o.lane_cpus = to_cpu_list(key, v);
        else throw std::runtime_error("unknown option: " + key);
    }
    if ((o.issuer_cpu >= 0) != !o.lane_cpus.empty())
        throw std::runtime_error("--issuer-cpu and --lane-cpus go together");
    if (!o.lane_cpus.empty()) {
        if (std::find(o.lane_cpus.begin(), o.lane_cpus.end(), o.issuer_cpu) != o.lane_cpus.end())
            throw std::runtime_error("the issuer CPU cannot be a lane CPU");
        o.threads = int(o.lane_cpus.size());
    }
    if (o.n_ubatch < 0) o.n_ubatch = o.n_prompt;
    if (o.model.empty() || o.out.empty() || o.prompt_file.empty() || o.n_prompt < 1 || o.n_gen < 1 ||
        o.n_ubatch < 1 || o.threads < 1 || o.n_ctx < o.n_prompt + o.n_gen - 1)
        throw std::runtime_error("invalid options; see --help");
    return o;
}

std::vector<llama_token> read_prompt(const std::string & path, int n) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot read " + path);
    std::vector<llama_token> ids;
    int64_t id;
    while (int(ids.size()) < n && f >> id) ids.push_back(llama_token(id));
    if (int(ids.size()) != n) throw std::runtime_error("prompt file holds fewer than the requested ids");
    return ids;
}

struct Counters {
    ggml_gemmini_pb_counters by_type[N_TYPES] = {};
    uint64_t qk_tensors = 0, pv_tensors = 0;

    uint64_t sum(uint64_t ggml_gemmini_pb_counters::*field) const {
        uint64_t total = 0;
        for (const auto & c : by_type) total += c.*field;
        return total;
    }
    uint64_t linear_nodes() const {
        uint64_t total = 0;
        for (int t = GGML_GEMMINI_PB_Q; t <= GGML_GEMMINI_PB_LM_HEAD; ++t) total += by_type[t].nodes;
        return total;
    }
};

Counters snapshot() {
    Counters c;
    for (int t = 0; t < N_TYPES; ++t)
        if (!ggml_gemmini_pb_get_counters(ggml_gemmini_pb_node_type(t), &c.by_type[t]))
            throw std::runtime_error("PB counters unavailable");
    c.qk_tensors = ggml::gemmini::attention_completed_nodes(GGML_GEMMINI_ATTENTION_QK);
    c.pv_tensors = ggml::gemmini::attention_completed_nodes(GGML_GEMMINI_ATTENTION_PV);
    return c;
}

Counters delta(const Counters & a, const Counters & b) {
    Counters d;
    for (int t = 0; t < N_TYPES; ++t) {
        d.by_type[t].nodes = b.by_type[t].nodes - a.by_type[t].nodes;
        d.by_type[t].k_blocks = b.by_type[t].k_blocks - a.by_type[t].k_blocks;
        d.by_type[t].int32_elements = b.by_type[t].int32_elements - a.by_type[t].int32_elements;
        d.by_type[t].fallback_count = b.by_type[t].fallback_count - a.by_type[t].fallback_count;
    }
    d.qk_tensors = b.qk_tensors - a.qk_tensors;
    d.pv_tensors = b.pv_tensors - a.pv_tensors;
    return d;
}

void add(Counters & total, const Counters & d) {
    for (int t = 0; t < N_TYPES; ++t) {
        total.by_type[t].nodes += d.by_type[t].nodes;
        total.by_type[t].k_blocks += d.by_type[t].k_blocks;
        total.by_type[t].int32_elements += d.by_type[t].int32_elements;
        total.by_type[t].fallback_count += d.by_type[t].fallback_count;
    }
    total.qk_tensors += d.qk_tensors;
    total.pv_tensors += d.pv_tensors;
}

std::string routing_error(const Counters & d, uint64_t n_layer, uint64_t n_head) {
    const auto expect = [](const char * what, uint64_t got, uint64_t want) -> std::string {
        if (got == want) return "";
        return std::string(what) + "=" + std::to_string(got) + " expected=" + std::to_string(want);
    };
    for (int t = GGML_GEMMINI_PB_Q; t <= GGML_GEMMINI_PB_FFN_DOWN; ++t) {
        const std::string e = expect(TYPE_NAMES[t], d.by_type[t].nodes, n_layer);
        if (!e.empty()) return e;
    }
    for (const std::string & e : {
             expect("lm_head", d.by_type[GGML_GEMMINI_PB_LM_HEAD].nodes, 1),
             expect("qk_heads", d.by_type[GGML_GEMMINI_PB_QK].nodes, n_layer * n_head),
             expect("pv_heads", d.by_type[GGML_GEMMINI_PB_PV].nodes, n_layer * n_head),
             expect("qk_tensors", d.qk_tensors, n_layer),
             expect("pv_tensors", d.pv_tensors, n_layer),
             expect("other", d.by_type[GGML_GEMMINI_PB_OTHER].nodes, 0),
             expect("fallback", d.sum(&ggml_gemmini_pb_counters::fallback_count), 0),
         })
        if (!e.empty()) return e;
    return "";
}

struct Step {
    uint64_t t_start = 0, t_end = 0, t_token = 0, cycle_start = 0, cycle_end = 0;
    int32_t token = -1;
    int context_len = 0, logits_finite = -1;
    Counters d;
    const char * status = "not_run";
};

std::ofstream open_out(const std::filesystem::path & path) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot write " + path.string());
    return f;
}

} // namespace

int main(int argc, char ** argv) try {
    const Options o = parse(argc, argv);
    const bool five = o.issuer_cpu >= 0;
    ggml_threadpool * threadpool = nullptr;
    std::vector<int> helper_cpus;
    if (!five) {
        pin_to_hart0();
    } else {
        const auto harts = cpu_harts();
        const auto hart_of = [&](int cpu) { const auto it = harts.find(cpu); return it == harts.end() ? -1 : it->second; };
        if (!harts.empty()) {
            if (hart_of(o.issuer_cpu) != 0) throw std::runtime_error("--issuer-cpu is not the CPU of hart 0");
            for (const int cpu : o.lane_cpus)
                if (hart_of(cpu) < 1) throw std::runtime_error("lane CPU " + std::to_string(cpu) + " is not a non-zero hart");
        }
        // ggml's CPU operators: workers on the first lane CPUs, the calling thread on the last, sleeping when idle.
        ggml_threadpool_params tpp = ggml_threadpool_params_default(o.threads);
        std::fill(std::begin(tpp.cpumask), std::end(tpp.cpumask), false);
        for (const int cpu : o.lane_cpus) tpp.cpumask[cpu] = true;
        tpp.strict_cpu = true;
        tpp.poll = 0;
        threadpool = ggml_threadpool_new(&tpp);
        if (!threadpool) throw std::runtime_error("ggml threadpool creation failed");
        const int main_cpu = o.lane_cpus.back();
        pin_main(main_cpu);
        helper_cpus.assign(o.lane_cpus.begin(), o.lane_cpus.end() - 1);
        const auto status = pb::start_runtime({o.issuer_cpu, helper_cpus});
        if (status != pb::Status::success)
            throw std::runtime_error("PB five-core runtime failed to start, status " + std::to_string(int(status)));
        std::printf("E2E_FIVE_CORE issuer_cpu=%d issuer_hart=%d main_cpu=%d main_hart=%d helper_cpus=%s ggml_threads=%d\n",
                    o.issuer_cpu, hart_of(o.issuer_cpu), main_cpu, hart_of(main_cpu), join(helper_cpus).c_str(), o.threads);
    }
    std::filesystem::create_directories(o.out);
    std::vector<llama_token> prompt = read_prompt(o.prompt_file, o.n_prompt);

    ggml_backend_load_all();
    llama_backend_init();

    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    llama_model * model = llama_model_load_from_file(o.model.c_str(), mp);
    if (!model) throw std::runtime_error("model loading failed: " + o.model);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const uint64_t n_layer = uint64_t(llama_model_n_layer(model));
    const uint64_t n_head = uint64_t(llama_model_n_head(model));
    for (const llama_token id : prompt)
        if (id < 0 || id >= n_vocab) throw std::runtime_error("prompt id outside vocab");

    auto cp = llama_context_default_params();
    cp.n_ctx = uint32_t(o.n_ctx);
    cp.n_batch = uint32_t(std::max(o.n_prompt, o.n_ubatch));
    cp.n_ubatch = uint32_t(o.n_ubatch);
    cp.n_threads = cp.n_threads_batch = o.threads;
    cp.flash_attn = false;
    cp.offload_kqv = true;
    cp.no_perf = true;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) throw std::runtime_error("context initialization failed");
    if (threadpool) llama_attach_threadpool(ctx, threadpool, threadpool);
    std::printf("E2E_MODEL_LOADED vocab=%d n_ctx=%u n_batch=%u n_ubatch=%u threads=%d n_layer=%llu n_head=%llu\n",
                n_vocab, llama_n_ctx(ctx), llama_n_batch(ctx), llama_n_ubatch(ctx), o.threads, ull(n_layer), ull(n_head));

    llama_sampler * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    Counters prev = snapshot();
    std::vector<Step> steps(o.n_gen);
    std::vector<std::vector<pb::ThreadReport>> thread_steps;
    if (five) thread_steps.push_back(pb::runtime_report());
    const char * run_status = "ok";
    int done = 0;
    llama_token input = 0;

    std::puts("E2E_BEGIN");
    std::fflush(stdout);
    const uint64_t t_run_start = now_ns();
    for (int s = 0; s < o.n_gen; ++s) {
        Step & st = steps[s];
        const bool prefill = s == 0;
        llama_batch batch = prefill ? llama_batch_get_one(prompt.data(), o.n_prompt) : llama_batch_get_one(&input, 1);
        st.t_start = now_ns();
        st.cycle_start = read_cycle();
        const int rc = llama_decode(ctx, batch);
        llama_synchronize(ctx);
        st.t_end = now_ns();
        st.context_len = o.n_prompt + s;
        const Counters cur = snapshot();
        st.d = delta(prev, cur);
        prev = cur;
        done = s + 1;
        if (rc != 0) {
            st.status = "decode_failed";
            run_status = st.status;
            std::printf("E2E_ABORT s=%d reason=decode_rc rc=%d\n", s, rc);
            std::fflush(stdout);
            break;
        }
        const llama_token tok = llama_sampler_sample(smpl, ctx, -1);
        st.t_token = now_ns();
        st.cycle_end = read_cycle();
        st.token = tok;
        input = tok;
        st.status = "ok";

        const float * row = llama_get_logits_ith(ctx, -1);
        st.logits_finite = row != nullptr;
        for (int i = 0; row && i < n_vocab; ++i)
            if (!std::isfinite(row[i])) { st.logits_finite = 0; break; }
        if (s == 0 && row && !o.dump_logits.empty()) {
            std::ofstream dump(o.dump_logits, std::ios::binary);
            dump.write(reinterpret_cast<const char *>(row), std::streamsize(n_vocab) * std::streamsize(sizeof(float)));
            if (!dump) throw std::runtime_error("cannot write logits dump");
        }

        const std::string route_err = routing_error(st.d, n_layer, n_head);
        if (!route_err.empty()) st.status = "routing_failed";
        else if (!st.logits_finite) st.status = "nonfinite_logits";
        if (five) {
            thread_steps.push_back(pb::runtime_report());
            const auto & a = thread_steps[thread_steps.size() - 2];
            const auto & b = thread_steps.back();
            std::string lanes;
            for (size_t n = 0; n < b.size(); ++n)
                lanes += (n ? "," : "") + std::string(b[n].role).substr(0, 1) + std::to_string(b[n].lane) + "@" +
                         std::to_string(b[n].cpu_last) + ":" + std::to_string((lane_busy_ns(b[n]) - lane_busy_ns(a[n])) / 1000000);
            std::printf("E2E_LANES s=%d busy_ms=%s\n", s, lanes.c_str());
        }

        std::printf("E2E_STEP s=%d ctx=%d latency_ns=%llu token=%d cycles=%llu finite=%d linear=%llu qk=%llu pv=%llu k_blocks=%llu fallback=%llu\n",
                    s, st.context_len, ull(st.t_end - st.t_start), tok, ull(st.cycle_end - st.cycle_start), st.logits_finite,
                    ull(st.d.linear_nodes()), ull(st.d.qk_tensors), ull(st.d.pv_tensors),
                    ull(st.d.sum(&ggml_gemmini_pb_counters::k_blocks)),
                    ull(st.d.sum(&ggml_gemmini_pb_counters::fallback_count)));
        if (prefill) std::printf("E2E_PREFILL_DONE ttft_ns=%llu\n", ull(st.t_token - st.t_start));
        if (std::string(st.status) != "ok") {
            run_status = st.status;
            std::printf("E2E_ABORT s=%d reason=%s detail=%s\n", s, st.status, route_err.empty() ? "none" : route_err.c_str());
            std::fflush(stdout);
            break;
        }
        std::fflush(stdout);
    }
    const uint64_t t_run_end = now_ns();

    const std::filesystem::path dir(o.out);
    Counters total;
    int generated = 0;
    {
        auto f = open_out(dir / "steps.csv");
        f << "step,phase,context_len,t_start_ns,t_end_ns,t_token_ns,latency_ns,token_id,cycles,logits_finite,qk_tensors,pv_tensors";
        for (const char * name : TYPE_NAMES)
            f << ',' << name << "_nodes," << name << "_k_blocks," << name << "_int32_elements," << name << "_fallback";
        f << ",status\n";
        for (int s = 0; s < done; ++s) {
            const Step & st = steps[s];
            add(total, st.d);
            if (st.token >= 0) ++generated;
            f << s << ',' << (s == 0 ? "prefill" : "decode") << ',' << st.context_len << ',' << st.t_start << ','
              << st.t_end << ',' << st.t_token << ',' << (st.t_end - st.t_start) << ',' << st.token << ','
              << (st.cycle_end - st.cycle_start) << ',' << st.logits_finite << ',' << st.d.qk_tensors << ',' << st.d.pv_tensors;
            for (const auto & c : st.d.by_type)
                f << ',' << c.nodes << ',' << c.k_blocks << ',' << c.int32_elements << ',' << c.fallback_count;
            f << ',' << st.status << '\n';
        }
    }
    {
        auto f = open_out(dir / "tokens.txt");
        for (int s = 0; s < done; ++s)
            if (steps[s].token >= 0) f << steps[s].token << '\n';
    }

    const bool ok = std::string(run_status) == "ok" && generated == o.n_gen;
    if (std::string(run_status) == "ok" && !ok) run_status = "incomplete";
    const uint64_t fallback = total.sum(&ggml_gemmini_pb_counters::fallback_count);
    {
        auto f = open_out(dir / "summary.txt");
        f << "model=" << o.model << "\nprompt_file=" << o.prompt_file << "\nprompt_tokens=" << o.n_prompt
          << "\ngen_tokens=" << o.n_gen << "\nctx=" << llama_n_ctx(ctx) << "\nubatch=" << llama_n_ubatch(ctx)
          << "\nthreads=" << o.threads << "\nfive_core=" << (five ? 1 : 0) << "\nissuer_cpu=" << o.issuer_cpu
          << "\nlane_cpus=" << join(o.lane_cpus) << "\nsteps_run=" << done << "\ntokens_generated=" << generated
          << "\nrun_wall_ns=" << (t_run_end - t_run_start) << "\ntotal_qk_tensors=" << total.qk_tensors
          << "\ntotal_pv_tensors=" << total.pv_tensors << "\ntotal_linear_nodes=" << total.linear_nodes()
          << "\ntotal_fallback=" << fallback;
        for (int t = 0; t < N_TYPES; ++t)
            f << "\ntotal_" << TYPE_NAMES[t] << "_nodes=" << total.by_type[t].nodes
              << "\ntotal_" << TYPE_NAMES[t] << "_k_blocks=" << total.by_type[t].k_blocks
              << "\ntotal_" << TYPE_NAMES[t] << "_int32_elements=" << total.by_type[t].int32_elements;
        f << "\nstatus=" << run_status << "\nexit=" << (ok ? 0 : 1) << '\n';
    }

    if (five) {
        auto f = open_out(dir / "threads-steps.csv");
        f << "after_step,role,lane,tid,cpu_assigned,cpu_first,cpu_last,cpu_mismatch,sessions,phases,thread_cpu_ns,"
             "busy_prep_ns,busy_scale_ns,busy_quant_ns,busy_gemm_ns,wait_ns,gemm_calls,flushes,packed_a_bytes,"
             "packed_b_bytes,scales_b,zeroed,scaled_elements,quantized_rows\n";
        for (size_t n = 0; n < thread_steps.size(); ++n) write_thread_rows(f, int(n) - 1, thread_steps[n]);
        auto e = open_out(dir / "events.csv");
        e << "seq,session,lane,kind,block,begin_ns,end_ns\n";
        for (const auto & ev : pb::runtime_events())
            e << ev.seq << ',' << ev.session << ',' << ev.lane << ',' << ev.kind << ',' << ev.block << ','
              << ev.begin_ns << ',' << ev.end_ns << '\n';
        // Every thread of the process as the kernel saw it, ggml's workers included.
        std::map<long, std::string> roles;
        for (const auto & t : thread_steps.back()) roles[t.tid] = std::string(t.role) + std::to_string(t.lane);
        auto k = open_out(dir / "tasks.csv");
        k << "tid,role,state,utime_ticks,stime_ticks,last_cpu,cpus_allowed,clk_tck\n";
        for (const auto & entry : std::filesystem::directory_iterator("/proc/self/task")) {
            const long tid = std::stol(entry.path().filename().string());
            std::ifstream statf(entry.path() / "stat");
            std::string stat((std::istreambuf_iterator<char>(statf)), std::istreambuf_iterator<char>());
            std::istringstream rest(stat.substr(stat.rfind(')') + 2));
            std::vector<std::string> fld;
            for (std::string x; rest >> x;) fld.push_back(x);
            std::ifstream statusf(entry.path() / "status");
            std::string line, allowed;
            while (std::getline(statusf, line))
                if (line.rfind("Cpus_allowed_list:", 0) == 0) allowed = line.substr(line.find_first_not_of(" \t", 18));
            const auto it = roles.find(tid);
            k << tid << ',' << (it != roles.end() ? it->second : tid == long(getpid()) ? "main" : "ggml_or_other") << ','
              << (fld.size() > 36 ? fld[0] + ',' + fld[11] + ',' + fld[12] + ',' + fld[36] : std::string("?,?,?,?"))
              << ',' << '"' << allowed << '"' << ',' << sysconf(_SC_CLK_TCK) << '\n';
        }
        for (const auto & t : thread_steps.back())
            std::printf("E2E_THREAD role=%s lane=%d tid=%ld cpu=%d/%d/%d mismatch=%llu sessions=%llu phases=%llu gemm=%llu "
                        "flushes=%llu scaled=%llu packed_b=%llu quant_rows=%llu busy_ms=%llu wait_ms=%llu cpu_ms=%llu\n",
                        t.role, t.lane, t.tid, t.cpu_assigned, t.cpu_first, t.cpu_last, ull(t.cpu_mismatch), ull(t.sessions),
                        ull(t.phases), ull(t.gemm_calls), ull(t.flushes), ull(t.scaled_elements), ull(t.packed_b_bytes),
                        ull(t.quantized_rows), ull(lane_busy_ns(t) / 1000000), ull(t.wait_ns / 1000000), ull(t.thread_cpu_ns / 1000000));
    }
    for (int t = 0; t < N_TYPES; ++t)
        std::printf("E2E_TOTAL type=%s nodes=%llu k_blocks=%llu int32_elements=%llu fallback=%llu\n", TYPE_NAMES[t],
                    ull(total.by_type[t].nodes), ull(total.by_type[t].k_blocks), ull(total.by_type[t].int32_elements),
                    ull(total.by_type[t].fallback_count));
    std::printf("E2E_SUMMARY linear_nodes=%llu qk_tensors=%llu pv_tensors=%llu total_fallback=%llu\n",
                ull(total.linear_nodes()), ull(total.qk_tensors), ull(total.pv_tensors), ull(fallback));
    std::printf("E2E_DONE status=%s steps=%d tokens=%d fallback=%llu\n", run_status, done, generated, ull(fallback));
    std::fflush(stdout);
    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    if (threadpool) ggml_threadpool_free(threadpool);
    pb::stop_runtime();
    return ok ? 0 : 1;
} catch (const std::exception & e) {
    std::fprintf(stderr, "E2E_ERROR %s\n", e.what());
    std::printf("E2E_DONE status=error\n");
    return 2;
}
