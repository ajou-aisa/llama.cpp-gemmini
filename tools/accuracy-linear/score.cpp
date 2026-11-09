#include "../accuracy-fast/ordered-logprobs.hpp"
#include "../perplexity/perplexity-cuda-cpu-exact.hpp"
#include "llama.h"
#include "json.hpp"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

using json = nlohmann::json;

static std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text) {
    const int count = -llama_tokenize(vocab, text.data(), int(text.size()), nullptr, 0, true, false);
    if (count < 0) throw std::runtime_error("Invalid tokenizer capacity");
    std::vector<llama_token> tokens(count);
    if (llama_tokenize(vocab, text.data(), int(text.size()), tokens.data(), count, true, false) != count)
        throw std::runtime_error("Tokenizer count changed");
    return tokens;
}

class Scorer {
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model{nullptr, llama_model_free};
    std::unique_ptr<llama_context, decltype(&llama_free)> context{nullptr, llama_free};
    perplexity_cuda_cpu_exact_guard guard;
    llama_batch batch{};
    const llama_vocab * vocab = nullptr;
    int layers = 0, scored_total = 0;
    uint64_t matmuls = 0;

    static bool observe(ggml_tensor * tensor, bool ask, void * data) {
        auto & scorer = *static_cast<Scorer *>(data);
        if (ask && tensor->op == GGML_OP_MUL_MAT) {
            ++scorer.matmuls;
            const auto * w = tensor->src[0];
            const auto * x = tensor->src[1];
            const bool attention = x && (x->ne[2] > 1 || x->ne[3] > 1);
            const auto expected = EVALUATION_EXACT_BITS == 4 ? GGML_TYPE_MXINT4 : GGML_TYPE_MXINT8;
            if (!w || !x || (attention ? w->type != GGML_TYPE_F16 && w->type != GGML_TYPE_F32
                                                     : w->type != expected))
                scorer.guard.failed = true;
        }
        return !ask && !scorer.guard.failed;
    }

    json counters() const {
        const auto scu = ggml_cuda_hp1_scu_stats();
        const auto attention = ggml_cuda_potal_attention_get_stats();
        return {{"dense_matmuls", guard.observed}, {"verified_dense_matmuls", guard.verified},
                {"attention_matmuls", guard.observed_attention},
                {"verified_attention_matmuls", guard.verified_attention},
                {"qk_quantized", guard.observed_qk}, {"pv_quantized", guard.observed_pv},
                {"integer_launches", gemmini_gpu::int_launches()},
                {"residual_launches", gemmini_gpu::residual_launches()},
                {"rtn_dense_calls", ggml_cuda_rtn_dense_calls()},
                {"rtn_attention_calls", ggml_cuda_rtn_attention_calls()},
                {"attention_main_calls", attention.main_dot_calls},
                {"attention_residual_calls", attention.residual_dot_calls},
                {"main_fragments", scu.main_fragments}, {"residual_fragments", scu.residual_fragments},
                {"saturations", scu.main_shift_saturations + scu.main_acc_saturations +
                                scu.residual_shift_saturations + scu.residual_acc_saturations}};
    }

public:
    explicit Scorer(const char * path) {
        const char * scope = std::getenv("GGML_GEMMINI_ATTENTION_SCOPE");
        if (!scope || std::strcmp(scope, "none") != 0 || !gemmini_gpu::enabled() ||
            !ggml_hp1_scu_enabled() || !ggml_cuda_rtn_enabled() || ggml_cuda_rtn_activation_bits() != 16 ||
            ggml::gemmini::sensitivity::reported_anchor() != 3)
            throw std::runtime_error("Require TOP3 PoTal LINEAR CUDA with unquantized FP16 attention");
        common_params proof_params;
        proof_params.n_gpu_layers = 0;
        proof_params.warmup = false;
        proof_params.ppl_stride = 0;
        proof_params.cb_eval = observe;
        proof_params.cb_eval_user_data = this;
        guard.install(proof_params);
        if (!guard.active || !guard.baseline_attention)
            throw std::runtime_error("Linear/FP16-attention CUDA graph verification is inactive");
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = 0;
        model.reset(llama_model_load_from_file(path, mp));
        if (!model) throw std::runtime_error("Model load failed");
        char file_type[32];
        const int expected = EVALUATION_EXACT_BITS == 4 ? LLAMA_FTYPE_MOSTLY_MXINT4 : LLAMA_FTYPE_MOSTLY_MXINT8;
        if (llama_model_meta_val_str(model.get(), "general.file_type", file_type, sizeof(file_type)) <= 0 ||
            std::string(file_type) != std::to_string(expected))
            throw std::runtime_error("Model width does not match PoTal LINEAR build");
        vocab = llama_model_get_vocab(model.get());
        layers = llama_model_n_layer(model.get());
        auto cp = llama_context_default_params();
        cp.n_ctx = 4096;
        cp.n_batch = cp.n_ubatch = 512;
        cp.n_threads = cp.n_threads_batch = 6;
        cp.offload_kqv = true;
        cp.flash_attn = false;
        cp.type_k = cp.type_v = GGML_TYPE_F16;
        cp.cb_eval = proof_params.cb_eval;
        cp.cb_eval_user_data = proof_params.cb_eval_user_data;
        context.reset(llama_init_from_model(model.get(), cp));
        if (!context) throw std::runtime_error("Context creation failed");
        batch = llama_batch_init(512, 0, 1);
    }

    ~Scorer() { if (batch.token) llama_batch_free(batch); }

    json score(const std::string & prefix, const std::string & continuation) {
        const auto prefix_tokens = tokenize(vocab, prefix), tokens = tokenize(vocab, prefix + continuation);
        const size_t first = prefix_tokens.size();
        if (!first || first >= tokens.size() || tokens.size() > 4096)
            throw std::runtime_error("Empty context/continuation or context overflow; no truncation allowed");
        llama_kv_self_clear(context.get());
        const json before = counters();
        const uint64_t matmuls_before = matmuls;
        uint64_t decodes = 0;
        double total = 0;
        bool greedy = true;
        size_t scored = 0;
        for (size_t offset = 0; offset + 1 < tokens.size();) {
            batch.n_tokens = int(std::min(size_t(512), tokens.size() - 1 - offset));
            for (int j = 0; j < batch.n_tokens; ++j) {
                const size_t position = offset + j;
                batch.token[j] = tokens[position];
                batch.pos[j] = int(position);
                batch.n_seq_id[j] = 1;
                batch.seq_id[j][0] = 0;
                batch.logits[j] = position + 1 >= first || j + 1 == batch.n_tokens;
            }
            if (llama_decode(context.get(), batch)) throw std::runtime_error("CUDA decode failed");
            ++decodes;
            std::vector<accuracy_fast::LogitsRow> rows;
            for (int j = 0; j < batch.n_tokens; ++j) {
                if (offset + j + 1 < first) continue;
                const auto * logits = llama_get_logits_ith(context.get(), j);
                if (!logits) throw std::runtime_error("Missing scored-token logits");
                rows.push_back({logits, tokens[offset + j + 1]});
            }
            for (const auto & result : accuracy_fast::ordered_log_probabilities(rows, llama_vocab_n_tokens(vocab))) {
                total += result.first;
                greedy = greedy && result.second;
                ++scored;
            }
            offset += batch.n_tokens;
        }
        json proof = counters();
        for (auto & entry : proof.items())
            entry.value() = entry.value().get<uint64_t>() - before.at(entry.key()).get<uint64_t>();
        const uint64_t dense = uint64_t(7 * layers + 1) * decodes, attention = uint64_t(2 * layers) * decodes;
        if (guard.failed || matmuls - matmuls_before != dense + attention ||
            proof.at("dense_matmuls") != dense || proof.at("verified_dense_matmuls") != dense ||
            proof.at("attention_matmuls") != attention || proof.at("verified_attention_matmuls") != attention ||
            proof.at("rtn_attention_calls") != attention || proof.at("integer_launches") == 0 ||
            proof.at("qk_quantized") != 0 || proof.at("pv_quantized") != 0 ||
            proof.at("attention_main_calls") != 0 || proof.at("attention_residual_calls") != 0 ||
            proof.at("rtn_dense_calls") != 0 || proof.at("saturations") != 0 || scored != tokens.size() - first)
            throw std::runtime_error("Incomplete LINEAR CUDA coverage or unexpected quantized attention");
        const uint64_t gpu_calls = proof.at("integer_launches").get<uint64_t>() +
                                  proof.at("residual_launches").get<uint64_t>() + attention;
        proof.update({{"schema", "potal-linear-request-v1"}, {"complete", true},
                      {"bits", EVALUATION_EXACT_BITS}, {"attention_bits", 16}, {"decode_calls", decodes}});
        scored_total += int(scored);
        return {{"loglikelihood", total}, {"greedy", greedy}, {"scored_tokens", scored},
                {"context_tokens", std::vector<llama_token>(tokens.begin(), tokens.begin() + first)},
                {"continuation_tokens", std::vector<llama_token>(tokens.begin() + first, tokens.end())},
                {"gpu_calls", gpu_calls}, {"backend_proof", proof}};
    }

    void finish() {
        if (!guard.finish(true, scored_total)) throw std::runtime_error("Final LINEAR CUDA proof failed");
    }
};

int main(int argc, char ** argv) {
    try {
        if (argc != 2) throw std::runtime_error("Usage: native-score-potal-linear MODEL");
        llama_backend_init();
        Scorer scorer(argv[1]);
        std::cout << json({{"ready", true}, {"backend", "CUDA"}, {"threads", 6}, {"profile", "potal-linear"},
                           {"bits", EVALUATION_EXACT_BITS}, {"context_limit", 4096}, {"batch_size", 512}}).dump() << std::endl;
        std::string line;
        while (std::getline(std::cin, line)) {
            const auto request = json::parse(line);
            std::cout << scorer.score(request.at("context"), request.at("continuation")).dump() << std::endl;
        }
        scorer.finish();
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "PoTal LINEAR accuracy failed: " << error.what() << '\n';
        return 1;
    }
}
