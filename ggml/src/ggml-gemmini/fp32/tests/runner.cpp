#include "llama.h"
#include "ggml-cpu.h"
#if defined(GGML_GEMMINI_FP32)
#include "../exec.hpp"
#endif
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <sched.h>

namespace {
void check(bool ok, const std::string & why) { if (!ok) throw std::runtime_error(why); }
uint64_t now() {
    return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}
std::vector<int> cpus(const std::string & text) {
    std::istringstream in(text);
    std::vector<int> values;
    for (std::string s; std::getline(in, s, ',');) values.push_back(std::stoi(s));
    check(values.size() == 4, "exactly four --lane-cpus required");
    auto sorted = values; std::sort(sorted.begin(), sorted.end());
    check(sorted.front() >= 0 && sorted.back() < GGML_MAX_N_THREADS &&
          std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end(), "invalid lane CPU set");
    return values;
}
void check_harts(int issuer, const std::vector<int> & lanes) {
#if defined(__riscv)
    std::ifstream info("/proc/cpuinfo");
    std::map<int, int> harts;
    int cpu = -1, x;
    for (std::string line; std::getline(info, line);) {
        if (std::sscanf(line.c_str(), "processor : %d", &x) == 1) cpu = x;
        else if (std::sscanf(line.c_str(), "hart : %d", &x) == 1) harts[cpu] = x;
    }
    check(harts.count(issuer) && harts.at(issuer) == 0, "issuer must be Linux CPU of hart 0");
    for (const int c : lanes) check(harts.count(c) && harts.at(c) > 0, "lane must be nonzero hart");
#else
    (void)issuer; (void)lanes;
#endif
}
}
int main(int argc, char ** argv) try {
    std::map<std::string, std::string> args;
    for (int i = 1; i < argc; i += 2) {
        check(i + 1 < argc, "missing argument value");
        args[argv[i]] = argv[i + 1];
    }
    const auto get = [&](const char * key, const char * fallback = "") {
        const auto i = args.find(key);
        return i == args.end() ? std::string(fallback) : i->second;
    };
    const int prompt_n = std::stoi(get("--prompt-tokens", "512"));
    const int generated = std::stoi(get("--gen-tokens", "9"));
    const int ctx_n = std::stoi(get("--ctx", "2048"));
    const int ubatch = std::stoi(get("--ubatch", "512"));
    const int issuer = std::stoi(get("--issuer-cpu", "0"));
    const auto lanes = cpus(get("--lane-cpus", "1,2,3,4"));
    const std::string output = get("--out");
    check(!output.empty() && prompt_n > 0 && generated > 0 && ubatch >= prompt_n &&
          ctx_n >= prompt_n + generated - 1, "invalid run dimensions/output");
    check_harts(issuer, lanes);
    std::ifstream prompt_file(get("--prompt-file"));
    check(bool(prompt_file), "cannot read prompt file");
    std::vector<llama_token> prompt(prompt_n);
    for (int i = 0; i < prompt_n; ++i) {
        int token;
        check(bool(prompt_file >> token) && token == (i * 7919 + 13) % 150000, "prompt mismatch");
        prompt[i] = token;
    }
    std::filesystem::create_directories(output);
    auto tpp = ggml_threadpool_params_default(4);
    std::fill(std::begin(tpp.cpumask), std::end(tpp.cpumask), false);
    for (const int cpu : lanes) tpp.cpumask[cpu] = true;
    tpp.strict_cpu = true; tpp.poll = 0;
    auto * pool = ggml_threadpool_new(&tpp);
    check(pool, "cannot create four-lane CPU threadpool");
    cpu_set_t mask;
    CPU_ZERO(&mask); CPU_SET(lanes.back(), &mask);
    check(sched_setaffinity(0, sizeof(mask), &mask) == 0, "cannot pin coordinator");
#if defined(GGML_GEMMINI_FP32)
    namespace fp = ggml::gemmini::fp32;
    fp::set_order(fp::Order(std::stoul(get("--order", "0"))));
    fp::start_runtime(issuer, lanes);
    const char * route = "FP32_GEMMINI_MODEL";
#else
    const char * route = "CPU_F32_F16";
#endif
    ggml_backend_load_all();
    llama_backend_init();
    auto mp = llama_model_default_params();
#if defined(GGML_GEMMINI_FP32)
    mp.n_gpu_layers = 999;
#else
    mp.n_gpu_layers = 0;
#endif
    auto * model = llama_model_load_from_file(get("--model").c_str(), mp);
    check(model, "model load failed");
    check(llama_model_n_layer(model) == 24 && llama_model_n_embd(model) == 896, "not Qwen2-0.5B");
    auto cp = llama_context_default_params();
    cp.n_ctx = uint32_t(ctx_n); cp.n_batch = uint32_t(prompt_n); cp.n_ubatch = uint32_t(ubatch);
    cp.n_threads = cp.n_threads_batch = 4;
    cp.flash_attn = false; cp.no_perf = true;
    cp.type_k = cp.type_v = GGML_TYPE_F16;
#if defined(GGML_GEMMINI_FP32)
    cp.offload_kqv = true;
#else
    cp.offload_kqv = false;
#endif
    auto * context = llama_init_from_model(model, cp);
    check(context, "context creation failed");
    llama_attach_threadpool(context, pool, pool);
    const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    std::ofstream csv(output + "/steps.csv"), tokens(output + "/tokens.txt");
    csv << "graph,context,tokens_in,token,finite,linear,qk,pv,gemms,flushes,conversions,cached_bytes,wall_ns\n";
    check(bool(csv) && bool(tokens), "cannot create result files");
    std::printf("FP32_BEGIN route=%s prompt=%d gen=%d ctx=%d ubatch=%d issuer=%d lanes=%s\n",
                route, prompt_n, generated, ctx_n, ubatch, issuer, get("--lane-cpus","1,2,3,4").c_str());
    const uint64_t start = now();
    llama_token input = 0;
    for (int step = 0; step < generated; ++step) {
#if defined(GGML_GEMMINI_FP32)
        fp::begin_graph(uint64_t(step), output + "/dumps");
#endif
        const uint64_t begin = now();
        auto batch = step == 0 ? llama_batch_get_one(prompt.data(), prompt_n) : llama_batch_get_one(&input, 1);
        check(llama_decode(context, batch) == 0, "llama_decode failed");
        llama_synchronize(context);
        const uint64_t wall = now() - begin;
        const float * logits = llama_get_logits_ith(context, -1);
        check(logits, "no logits");
        for (int i = 0; i < vocab; ++i) check(std::isfinite(logits[i]), "nonfinite logits");
        input = llama_token(std::max_element(logits, logits + vocab) - logits);
        std::ofstream logit_file(output + "/logits-" + std::to_string(step) + ".bin", std::ios::binary);
        logit_file.write(reinterpret_cast<const char *>(logits), std::streamsize(vocab)*4);
        check(bool(logit_file), "logit dump failed");
        uint64_t linear = 0, qk = 0, pv = 0, gemms = 0, flushes = 0, conversions = 0, bytes = 0;
#if defined(GGML_GEMMINI_FP32)
        const auto c = fp::counters();
        linear = c.linear; qk = c.qk; pv = c.pv; gemms = c.gemms; flushes = c.flushes;
        conversions = c.conversions; bytes = c.cached_bytes;
        check(linear == 169 && qk == 24 && pv == 24, "strict routing count failed");
        check(gemms == 841 && flushes == gemms, "GEMM/flush count failed");
#endif
        csv << step << ',' << prompt_n + step << ',' << (step ? 1 : prompt_n) << ',' << input
            << ",1," << linear << ',' << qk << ',' << pv << ',' << gemms << ',' << flushes
            << ',' << conversions << ',' << bytes << ',' << wall << '\n';
        csv.flush();
        tokens << input << '\n'; tokens.flush();
        std::printf("FP32_STEP graph=%d token=%d finite=1 linear=%llu qk=%llu pv=%llu gemms=%llu flushes=%llu wall_ns=%llu\n",
                    step, input, (unsigned long long)linear, (unsigned long long)qk, (unsigned long long)pv,
                    (unsigned long long)gemms, (unsigned long long)flushes, (unsigned long long)wall);
        std::fflush(stdout);
    }
#if defined(GGML_GEMMINI_FP32)
    const auto report = fp::runtime_report();
    for (size_t i = 0; i < report.size(); ++i) {
        const auto & r = report[i];
        std::printf("FP32_THREAD lane=%zu cpu=%d jobs=%llu elements=%llu mismatch=%llu\n", i, r.cpu,
                    (unsigned long long)r.jobs, (unsigned long long)r.elements, (unsigned long long)r.mismatch);
        check(r.jobs > 0 && r.mismatch == 0, "worker work/affinity failed");
    }
#endif
    std::printf("FP32_DONE PASS route=%s graphs=%d generated=%d wall_ns=%llu\n",
                route, generated, generated, (unsigned long long)(now()-start));
    llama_free(context); llama_model_free(model);
#if defined(GGML_GEMMINI_FP32)
    fp::stop_runtime();
#endif
    ggml_threadpool_free(pool); llama_backend_free();
    return 0;
} catch (const std::exception & e) {
    std::fprintf(stderr, "FP32_RUN FAIL %s\n", e.what());
#if defined(GGML_GEMMINI_FP32)
    ggml::gemmini::fp32::stop_runtime();
#endif
    return 1;
}
