#include "evaluation-workload.h"
#include "common.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

std::vector<llama_token> common_evaluation_tokenize(const llama_context * ctx, const std::string & text) {
    return common_tokenize(ctx, text, true, false);
}

common_evaluation_layout common_evaluation_plan(size_t tokens, int n_ctx, int n_batch, int max_chunks,
        int first_chunk) {
    if (n_ctx <= 0 || n_batch <= 0 || (n_batch >= n_ctx && n_batch % n_ctx != 0) ||
            tokens < 2 * size_t(n_ctx) || tokens / n_ctx > size_t(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("native workload requires two full contexts and a compatible batch size");
    }
    const size_t total = tokens / n_ctx;
    if (first_chunk < 0 || size_t(first_chunk) >= total) {
        throw std::invalid_argument("native chunk index must name a complete chunk");
    }
    const int available = int(total) - first_chunk;
    const int chunks = max_chunks < 0 ? available : std::min(max_chunks, available);
    return {n_ctx, n_batch, chunks, std::max(1, n_batch / n_ctx), 1 + (n_ctx - 1) / n_batch,
            total, tokens % n_ctx, first_chunk};
}

common_params_sampling common_evaluation_e2e_sampling() {
    common_params_sampling params;
    params.seed = 1234;
    params.temp = 0;
    params.top_k = 0;
    params.top_p = 1;
    params.min_p = 0;
    params.penalty_repeat = 1;
    params.penalty_last_n = 0;
    return params;
}

void common_evaluation_validate_forced(const std::vector<llama_token> & tokens, int n_vocab, bool full_cpu_source) {
    if (!full_cpu_source || tokens.size() != 128 || n_vocab <= 0 ||
            std::any_of(tokens.begin(), tokens.end(), [n_vocab](llama_token id) { return id < 0 || id >= n_vocab; })) {
        throw std::invalid_argument("forced trajectory requires FullCPU and exactly128 valid vocabulary token IDs");
    }
}

bool common_evaluation_generation_complete(size_t samples, int decode_calls) {
    return samples == 128 && decode_calls == 127;
}

void common_evaluation_begin(llama_context * ctx) {
    llama_kv_self_clear(ctx);
}

int common_evaluation_batch(llama_batch & batch, const std::vector<llama_token> & tokens,
        const common_evaluation_layout & plan, int group, int part, bool add_bos, llama_token bos,
        common_evaluation_mask mask) {
    if (group < 0 || group >= plan.n_chunks || part < 0 || part >= plan.n_batches) {
        throw std::out_of_range("native workload batch outside selected chunks");
    }
    const int n_seq = std::min(plan.n_seq, plan.n_chunks - group);
    const int batch_size = std::min(plan.n_ctx - part * plan.n_batch, plan.n_batch);
    int outputs = 0;
    batch.n_tokens = 0;
    for (int seq = 0; seq < n_seq; ++seq) {
        const size_t start = size_t(plan.first_chunk + group + seq) * plan.n_ctx + part * plan.n_batch;
        for (int k = 0; k < batch_size; ++k) {
            const int idx = seq * plan.n_ctx + k;
            batch.token[idx] = add_bos && part == 0 && k == 0 ? bos : tokens.at(start + k);
            batch.pos[idx] = part * plan.n_batch + k;
            batch.n_seq_id[idx] = 1;
            batch.seq_id[idx][0] = seq;
            batch.logits[idx] = mask == common_evaluation_mask::perplexity_half ?
                batch.pos[idx] >= plan.n_ctx / 2 : batch.pos[idx] == plan.n_ctx - 1;
            outputs += batch.logits[idx] != 0;
        }
        batch.n_tokens += batch_size;
    }
    return outputs;
}
