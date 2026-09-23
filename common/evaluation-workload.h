#pragma once

#include "llama.h"

#include <cstddef>
#include <string>
#include <vector>

enum class common_evaluation_mask { perplexity_half, generation_last };

struct common_evaluation_layout {
    int n_ctx;
    int n_batch;
    int n_chunks;
    int n_seq;
    int n_batches;
    size_t total_chunks;
    size_t tail_tokens;
    int first_chunk;
};

struct common_params_sampling;
inline constexpr const char * common_evaluation_e2e_recipe = "wikitext2-test-256x128-greedy-seed1234-v1";
std::vector<llama_token> common_evaluation_tokenize(const llama_context * ctx, const std::string & text);
common_evaluation_layout common_evaluation_plan(size_t tokens, int n_ctx, int n_batch, int max_chunks,
        int first_chunk = 0);
common_params_sampling common_evaluation_e2e_sampling();
void common_evaluation_validate_forced(const std::vector<llama_token> & tokens, int n_vocab, bool full_cpu_source);
bool common_evaluation_generation_complete(size_t samples, int decode_calls);
void common_evaluation_begin(llama_context * ctx);
int common_evaluation_batch(llama_batch & batch, const std::vector<llama_token> & tokens,
        const common_evaluation_layout & plan, int group, int part, bool add_bos, llama_token bos,
        common_evaluation_mask mask);
