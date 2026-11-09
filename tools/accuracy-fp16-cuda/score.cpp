#include "llama.h"
#include "json.hpp"
#include "../perplexity/perplexity-cuda-rtn.hpp"

#include <algorithm>
#include <cmath>
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

static bool require_fp16(ggml_tensor * tensor, bool ask, void *) {
    if (ask && tensor->op == GGML_OP_MUL_MAT) {
        const auto * weights = tensor->src[0];
        const auto * input = tensor->src[1];
        if (!weights || !input) throw std::runtime_error("Missing matrix multiplication input");
        const bool attention = input->ne[2] > 1 || input->ne[3] > 1;
        if ((!attention && weights->type != GGML_TYPE_F16) ||
            (attention && weights->type != GGML_TYPE_F16 && weights->type != GGML_TYPE_F32)) {
            throw std::runtime_error("FP16 CUDA rejects non-F16 dense or quantized attention weights");
        }
    }
    return !ask;
}

class Scorer {
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model{nullptr, llama_model_free};
    std::unique_ptr<llama_context, decltype(&llama_free)> context{nullptr, llama_free};
    llama_batch batch{};
    const llama_vocab * vocab = nullptr;
    perplexity_cuda_rtn_guard guard;
    int capacity;
    int layers = 0;
    int scored_total = 0;

public:
    Scorer(const char * path, int threads, int batch_size) : capacity(batch_size) {
        if (threads != 6 || batch_size < 1 || batch_size > 512) {
            throw std::runtime_error("Require six CUDA host workers and batch 1..512");
        }
        if (!ggml_cuda_rtn_enabled() || ggml_cuda_rtn_activation_bits() != 16) {
            throw std::runtime_error("Require GGML_CUDA_RTN_A_BITS=16 for unquantized FP16 CUDA");
        }
        common_params proof_params;
        proof_params.n_gpu_layers = 0;
        proof_params.warmup = false;
        proof_params.ppl_stride = 0;
        proof_params.cb_eval = require_fp16;
        guard.install(proof_params);
        if (!guard.active) throw std::runtime_error("CUDA graph verification is inactive");
        auto model_params = llama_model_default_params();
        model_params.n_gpu_layers = 0;
        model.reset(llama_model_load_from_file(path, model_params));
        if (!model) throw std::runtime_error("Model load failed");
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
        llama_kv_self_clear(context.get());
        const auto dense_before = ggml_cuda_rtn_dense_calls();
        const auto attention_before = ggml_cuda_rtn_attention_calls();
        const auto observed_before = guard.observed, verified_before = guard.verified;
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
        const auto dense = ggml_cuda_rtn_dense_calls() - dense_before;
        const auto attention = ggml_cuda_rtn_attention_calls() - attention_before;
        const auto observed = guard.observed - observed_before, verified = guard.verified - verified_before;
        if (guard.failed || guard.mxint4 || guard.mxint8 ||
            dense != uint64_t(7 * layers + 1) * decodes || attention != uint64_t(2 * layers) * decodes ||
            observed != dense + attention || verified != observed || scored != tokens.size() - first) {
            throw std::runtime_error("Incomplete FP16 CUDA graph/token coverage or quantized execution");
        }
        scored_total += int(scored);
        const json proof = {{"schema", "fp16-cuda-request-v1"}, {"complete", true}, {"activation_bits", 16},
                            {"decode_calls", decodes}, {"dense_gpu_calls", dense}, {"attention_gpu_calls", attention},
                            {"observed_matmuls", observed}, {"verified_matmuls", verified},
                            {"hp1_q4_matmuls", 0}, {"hp1_q8_matmuls", 0}, {"compensation", false}};
        return {{"loglikelihood", total}, {"greedy", greedy}, {"scored_tokens", scored},
                {"context_tokens", std::vector<llama_token>(tokens.begin(), tokens.begin() + first)},
                {"continuation_tokens", std::vector<llama_token>(tokens.begin() + first, tokens.end())},
                {"gpu_calls", dense + attention}, {"backend_proof", proof}};
    }

    void finish() {
        if (!guard.finish(true, scored_total)) throw std::runtime_error("Final CUDA graph proof failed");
    }
};

int main(int argc, char ** argv) {
    try {
        if (argc != 4) throw std::runtime_error("Usage: native-score-fp16-cuda MODEL THREADS BATCH");
        check_softmax();
        llama_backend_init();
        const int threads = std::stoi(argv[2]), batch_size = std::stoi(argv[3]);
        Scorer scorer(argv[1], threads, batch_size);
        std::cout << json({{"ready", true}, {"backend", "CUDA"}, {"threads", threads}, {"profile", "fp16-cuda"},
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
        std::cerr << "FP16 CUDA accuracy scorer failed: " << error.what() << '\n';
        return 1;
    }
}
