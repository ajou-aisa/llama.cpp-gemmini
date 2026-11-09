#include "llama.h"
#include "json.hpp"
#ifdef ACCURACY_POTAL
#include "../perplexity/perplexity-cuda-cpu-exact.hpp"
using Guard = perplexity_cuda_cpu_exact_guard;
#else
#include "../perplexity/perplexity-cuda-rtn.hpp"
using Guard = perplexity_cuda_rtn_guard;
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::json;

static std::pair<double, bool> log_probability(const float * logits, int size, int target) {
    const auto best = std::max_element(logits, logits + size);
    double sum = 0;
    for (int i = 0; i < size; ++i) sum += std::exp(double(logits[i]) - *best);
    const double result = double(logits[target]) - *best - std::log(sum);
    if (!std::isfinite(result)) throw std::runtime_error("Nonfinite log probability");
    return {result, best - logits == target};
}

static void check_softmax() {
    const float equal[] = {1000, 1000};
    const float unequal[] = {0, float(std::log(2.0)), float(std::log(3.0))};
    const auto first = log_probability(equal, 2, 0);
    const auto second = log_probability(unequal, 3, 1);
    if (std::abs(first.first + std::log(2.0)) > 1e-6 || !first.second ||
        std::abs(second.first + std::log(3.0)) > 1e-6 || second.second) {
        throw std::runtime_error("Analytic softmax checks failed");
    }
}

static std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text) {
    const int needed = -llama_tokenize(vocab, text.data(), int(text.size()), nullptr, 0, true, false);
    if (needed < 0) throw std::runtime_error("Invalid tokenizer capacity");
    std::vector<llama_token> result(needed);
    if (llama_tokenize(vocab, text.data(), int(text.size()), result.data(), needed, true, false) != needed) {
        throw std::runtime_error("Tokenizer count changed");
    }
    return result;
}

class Scorer {
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model{nullptr, llama_model_free};
    std::unique_ptr<llama_context, decltype(&llama_free)> context{nullptr, llama_free};
    llama_batch batch{};
    const llama_vocab * vocab = nullptr;
    Guard guard;
    int capacity;
    int layers = 0;
    int scored_total = 0;
    std::string profile;
    int bits = 0;
    uint64_t matmuls = 0;

    ggml_type weight_type() const {
        if (profile == "nvfp4-public") return GGML_TYPE_F16;
        if (profile == "rtn") return bits == 4 ? GGML_TYPE_Q4_0 : GGML_TYPE_Q8_0;
        return bits == 4 ? GGML_TYPE_MXINT4 : GGML_TYPE_MXINT8;
    }

    static bool observe_policy(ggml_tensor * tensor, bool ask, void * data) {
        auto & scorer = *static_cast<Scorer *>(data);
        if (ask && tensor->op == GGML_OP_MUL_MAT) {
            ++scorer.matmuls;
            const auto * weight = tensor->src[0];
            const auto * input = tensor->src[1];
            if (!weight || !input) {
                scorer.guard.failed = true;
            } else {
                const bool attention = input->ne[2] > 1 || input->ne[3] > 1;
                const bool q6_head = scorer.profile == "rtn" && weight->type == GGML_TYPE_Q6_K &&
                    std::strcmp(tensor->name, "result_output") == 0 &&
                    (std::strcmp(weight->name, "output.weight") == 0 ||
                     std::strcmp(weight->name, "token_embd.weight") == 0);
                const bool valid = attention
                    ? weight->type == GGML_TYPE_F16 || weight->type == GGML_TYPE_F32
                    : weight->type == scorer.weight_type() || q6_head;
                if (!valid) {
                    scorer.guard.failed = true;
                    std::fprintf(stderr, "Accuracy weight policy mismatch: %s (%s)\n",
                                 tensor->name, ggml_type_name(weight->type));
                }
            }
        }
        return !ask && !scorer.guard.failed;
    }

    json counters() const {
#ifdef ACCURACY_POTAL
        const auto scu = ggml_cuda_hp1_scu_stats();
        const auto attention = ggml_cuda_potal_attention_get_stats();
        return {{"dense_matmuls", guard.observed}, {"verified_dense_matmuls", guard.verified},
                {"qk_matmuls", guard.observed_qk}, {"pv_matmuls", guard.observed_pv},
                {"verified_attention_matmuls", guard.verified_quant},
                {"hp1_q4_matmuls", bits == 4 ? guard.observed : 0},
                {"hp1_q8_matmuls", bits == 8 ? guard.observed : 0},
                {"integer_gpu_launches", gemmini_gpu::int_launches()},
                {"residual_gpu_launches", gemmini_gpu::residual_launches()},
                {"attention_main_dot_calls", attention.main_dot_calls},
                {"attention_residual_dot_calls", attention.residual_dot_calls},
                {"rtn_dense_calls", ggml_cuda_rtn_dense_calls()},
                {"rtn_attention_calls", ggml_cuda_rtn_attention_calls()},
                {"scu_saturations", scu.main_shift_saturations + scu.main_acc_saturations +
                                     scu.residual_shift_saturations + scu.residual_acc_saturations},
                {"main_fragments", scu.main_fragments + attention.main_fragments},
                {"residual_fragments", scu.residual_fragments + attention.residual_fragments}};
#else
        const auto dense = ggml_cuda_rtn_dense_calls(), attention = ggml_cuda_rtn_attention_calls();
        return {{"dense_matmuls", dense}, {"verified_dense_matmuls", dense},
                {"qk_matmuls", 0}, {"pv_matmuls", 0}, {"verified_attention_matmuls", attention},
                {"hp1_q4_matmuls", guard.mxint4}, {"hp1_q8_matmuls", guard.mxint8},
                {"integer_gpu_launches", 0}, {"residual_gpu_launches", 0},
                {"attention_main_dot_calls", 0}, {"attention_residual_dot_calls", 0},
                {"rtn_dense_calls", dense}, {"rtn_attention_calls", attention},
                {"scu_saturations", 0}, {"main_fragments", 0}, {"residual_fragments", 0}};
#endif
    }

public:
    Scorer(const char * path, int threads, int batch_size, const std::string & selected)
        : capacity(batch_size), profile(selected) {
        if (threads != 6 || batch_size < 1 || batch_size > 512) {
            throw std::runtime_error("Require six CUDA host workers and batch 1..512");
        }
#ifdef ACCURACY_POTAL
        bits = EVALUATION_EXACT_BITS;
        const char * scope = std::getenv("GGML_GEMMINI_ATTENTION_SCOPE");
        if (profile != "potal" || !gemmini_gpu::enabled() || !ggml_hp1_scu_enabled() ||
            !ggml_cuda_potal_attention_enabled() || !scope || std::string(scope) != "both" ||
            (ggml::gemmini::sensitivity::reported_anchor() != 2 &&
             ggml::gemmini::sensitivity::reported_anchor() != 3)) {
            throw std::runtime_error("Require PoTal FULL CUDA, SCU and explicit TOP2/TOP3");
        }
#else
        bits = ggml_cuda_rtn_activation_bits();
        const char * nvfp4 = std::getenv("GGML_CUDA_NVFP4");
        const bool nvfp4_enabled = nvfp4 && std::string(nvfp4) == "1";
        if ((profile != "mxint" && profile != "rtn" && profile != "nvfp4-public") ||
            !ggml_cuda_rtn_enabled() || nvfp4_enabled != (profile == "nvfp4-public") ||
            (profile == "nvfp4-public" && bits != 4)) {
            throw std::runtime_error("Require matching RTN/MXINT-WA or public NVFP4 W4/A4 CUDA profile");
        }
#endif
        if (bits != 4 && bits != 8) throw std::runtime_error("Require 4-bit or 8-bit operands");
        common_params proof_params;
        proof_params.n_gpu_layers = 0;
        proof_params.warmup = false;
        proof_params.ppl_stride = 0;
        proof_params.cb_eval = observe_policy;
        proof_params.cb_eval_user_data = this;
        guard.install(proof_params);
        if (!guard.active) throw std::runtime_error("CUDA graph verification is inactive");
        auto model_params = llama_model_default_params();
        model_params.n_gpu_layers = 0;
        model.reset(llama_model_load_from_file(path, model_params));
        if (!model) throw std::runtime_error("Model load failed");
        char file_type[32];
        const int expected_file_type = profile == "nvfp4-public" ? LLAMA_FTYPE_MOSTLY_F16 :
            profile == "rtn" ? (bits == 4 ? LLAMA_FTYPE_MOSTLY_Q4_0 : LLAMA_FTYPE_MOSTLY_Q8_0) :
            (bits == 4 ? LLAMA_FTYPE_MOSTLY_MXINT4 : LLAMA_FTYPE_MOSTLY_MXINT8);
        if (llama_model_meta_val_str(model.get(), "general.file_type", file_type, sizeof(file_type)) <= 0 ||
            std::string(file_type) != std::to_string(expected_file_type)) {
            throw std::runtime_error("Model file type does not match accuracy profile/operand width");
        }
        vocab = llama_model_get_vocab(model.get());
        layers = llama_model_n_layer(model.get());
        auto params = llama_context_default_params();
        params.n_ctx = 4096;
        params.n_batch = batch_size;
        params.n_ubatch = batch_size;
        params.n_threads = threads;
        params.n_threads_batch = threads;
        params.offload_kqv = true;
        params.flash_attn = false;
        params.type_k = GGML_TYPE_F16;
        params.type_v = GGML_TYPE_F16;
        params.cb_eval = proof_params.cb_eval;
        params.cb_eval_user_data = proof_params.cb_eval_user_data;
        context.reset(llama_init_from_model(model.get(), params));
        if (!context) throw std::runtime_error("CUDA context creation failed");
        batch = llama_batch_init(batch_size, 0, 1);
    }

    ~Scorer() { if (batch.token) llama_batch_free(batch); }

    json score(const std::string & prefix, const std::string & continuation) {
        const auto prefix_tokens = tokenize(vocab, prefix);
        const auto tokens = tokenize(vocab, prefix + continuation);
        const size_t first = prefix_tokens.size();
        if (!first || first >= tokens.size() || tokens.size() > 4096) {
            throw std::runtime_error("Empty context/continuation or context overflow; no truncation allowed");
        }
        // Every choice starts from an empty cache; quantization can depend on batch geometry.
        llama_kv_self_clear(context.get());
        const json before = counters();
        const uint64_t matmuls_before = matmuls;
        double total = 0;
        bool greedy = true;
        size_t scored = 0;
        uint64_t decodes = 0;
        const int vocabulary = llama_vocab_n_tokens(vocab);
        for (size_t offset = 0; offset + 1 < tokens.size();) {
            batch.n_tokens = int(std::min(size_t(capacity), tokens.size() - 1 - offset));
            for (int j = 0; j < batch.n_tokens; ++j) {
                const size_t position = offset + j;
                batch.token[j] = tokens[position];
                batch.pos[j] = int(position);
                batch.n_seq_id[j] = 1;
                batch.seq_id[j][0] = 0;
                // This fork needs one materialized output even on a prefix-only batch.
                batch.logits[j] = position + 1 >= first || j + 1 == batch.n_tokens;
            }
            if (llama_decode(context.get(), batch) != 0) throw std::runtime_error("CUDA decode failed");
            ++decodes;
            for (int j = 0; j < batch.n_tokens; ++j) {
                if (offset + j + 1 < first) continue;
                const float * logits = llama_get_logits_ith(context.get(), j);
                if (!logits) throw std::runtime_error("Missing scored-token logits");
                const auto result = log_probability(logits, vocabulary, tokens[offset + j + 1]);
                total += result.first;
                greedy = greedy && result.second;
                ++scored;
            }
            offset += batch.n_tokens;
        }
        json proof = counters();
        for (auto & entry : proof.items()) {
            entry.value() = entry.value().get<uint64_t>() - before.at(entry.key()).get<uint64_t>();
        }
        const uint64_t dense = uint64_t(7 * layers + 1) * decodes, attention = uint64_t(2 * layers) * decodes;
        if (guard.failed || matmuls - matmuls_before != dense + attention ||
            proof.at("dense_matmuls") != dense || proof.at("verified_dense_matmuls") != dense ||
            proof.at("verified_attention_matmuls") != attention || proof.at("scu_saturations") != 0 ||
            scored != tokens.size() - first) {
            throw std::runtime_error("Incomplete CUDA graph/token coverage or SCU saturation");
        }
#ifdef ACCURACY_POTAL
        if (proof.at("qk_matmuls") != attention / 2 || proof.at("pv_matmuls") != attention / 2 ||
            proof.at("integer_gpu_launches") == 0 || proof.at("residual_gpu_launches") == 0 ||
            proof.at("rtn_dense_calls") != 0 || proof.at("rtn_attention_calls") != 0) {
            throw std::runtime_error("PoTal FULL dense/QK/PV path was not executed");
        }
#else
        if (guard.observed != guard.verified || proof.at("rtn_attention_calls") != attention ||
            proof.at("hp1_q4_matmuls") != (profile == "mxint" && bits == 4 ? dense : 0) ||
            proof.at("hp1_q8_matmuls") != (profile == "mxint" && bits == 8 ? dense : 0)) {
            throw std::runtime_error("Wrong RTN/MXINT/NVFP4 weight or CUDA attention path");
        }
#endif
        uint64_t gpu_calls = 0;
        for (const char * key : {"integer_gpu_launches", "residual_gpu_launches", "attention_main_dot_calls",
                                 "attention_residual_dot_calls", "rtn_dense_calls", "rtn_attention_calls"}) {
            gpu_calls += proof.at(key).get<uint64_t>();
        }
        proof.update({{"schema", "quantized-accuracy-v1"}, {"profile", profile}, {"complete", true},
                      {"bits", bits}, {"decode_calls", decodes}});
        scored_total += int(scored);
        return {{"loglikelihood", total}, {"greedy", greedy}, {"scored_tokens", scored},
                {"context_tokens", std::vector<llama_token>(tokens.begin(), tokens.begin() + first)},
                {"continuation_tokens", std::vector<llama_token>(tokens.begin() + first, tokens.end())},
                {"gpu_calls", gpu_calls}, {"backend_proof", proof}};
    }

    void finish() {
        if (!guard.finish(true, scored_total)) throw std::runtime_error("Final CUDA graph proof failed");
    }
};

int main(int argc, char ** argv) {
    try {
        if (argc != 5) throw std::runtime_error("Usage: native-score MODEL THREADS BATCH PROFILE");
        check_softmax();
        llama_backend_init();
        const int threads = std::stoi(argv[2]), batch_size = std::stoi(argv[3]);
        Scorer scorer(argv[1], threads, batch_size, argv[4]);
        std::cout << json({{"ready", true}, {"backend", "CUDA"}, {"threads", threads}, {"profile", argv[4]},
                           {"context_limit", 4096}, {"batch_size", batch_size}}).dump() << std::endl;
        std::string line;
        while (std::getline(std::cin, line)) {
            const auto request = json::parse(line);
            std::cout << scorer.score(request.at("context").get<std::string>(),
                                      request.at("continuation").get<std::string>()).dump() << std::endl;
        }
        scorer.finish();
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "Quantized accuracy scorer failed: " << error.what() << '\n';
        return 1;
    }
}
