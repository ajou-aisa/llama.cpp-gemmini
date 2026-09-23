#include "evaluation-workload.h"
#include "common.h"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <numeric>
#include <stdexcept>
#include <vector>

// Frozen non-strided perplexity batch construction, independent of the helper.
static int original_batch(llama_batch & batch, std::vector<llama_token> tokens,
        int n_ctx, int n_batch, int n_chunks, int group, int part, bool add_bos) {
    const int n_seq = std::max(1, n_batch / n_ctx);
    const int size = std::min(n_ctx - part * n_batch, n_batch);
    int outputs = 0;
    batch.n_tokens = 0;
    for (int seq = 0; seq < std::min(n_seq, n_chunks - group); ++seq) {
        const int start = group * n_ctx + part * n_batch + seq * n_ctx;
        if (add_bos && part == 0) tokens[start] = 1;
        for (int k = 0; k < size; ++k) {
            const int idx = seq * n_ctx + k;
            batch.token[idx] = tokens[start + k];
            batch.pos[idx] = part * n_batch + k;
            batch.n_seq_id[idx] = 1;
            batch.seq_id[idx][0] = seq;
            batch.logits[idx] = batch.pos[idx] >= n_ctx / 2;
            outputs += batch.logits[idx] != 0;
        }
        batch.n_tokens += size;
    }
    return outputs;
}

int main() {
    std::vector<llama_token> tokens(256 * 5 + 17);
    std::iota(tokens.begin(), tokens.end(), 100);
    const auto unchanged = tokens;
    for (const int batch_size : {64, 100, 256, 512, 768}) {
        for (const bool bos : {false, true}) {
            const auto plan = common_evaluation_plan(tokens.size(), 256, batch_size, -1);
            assert(plan.n_chunks == 5 && plan.tail_tokens == 17);
            auto old = llama_batch_init(batch_size, 0, 1);
            auto actual = llama_batch_init(batch_size, 0, 1);
            for (int group = 0; group < plan.n_chunks; group += plan.n_seq) {
                for (int part = 0; part < plan.n_batches; ++part) {
                    const int expected = original_batch(old, tokens, 256, batch_size, 5, group, part, bos);
                    const int outputs = common_evaluation_batch(actual, tokens, plan, group, part, bos, 1,
                            common_evaluation_mask::perplexity_half);
                    assert(outputs == expected && old.n_tokens == actual.n_tokens);
                    for (int k = 0; k < old.n_tokens; ++k) {
                        assert(old.token[k] == actual.token[k]);
                        assert(old.pos[k] == actual.pos[k]);
                        assert(old.n_seq_id[k] == actual.n_seq_id[k]);
                        assert(old.seq_id[k][0] == actual.seq_id[k][0]);
                        assert(old.logits[k] == actual.logits[k]);
                    }
                }
            }
            llama_batch_free(old);
            llama_batch_free(actual);
        }
    }
    assert(tokens == unchanged);
    const auto plan = common_evaluation_plan(tokens.size(), 256, 256, 1);
    assert(plan.n_chunks == 1 && plan.total_chunks == 5);
    auto batch = llama_batch_init(256, 0, 1);
    assert(common_evaluation_batch(batch, tokens, plan, 0, 0, true, 1,
                common_evaluation_mask::perplexity_half) == 128);
    assert(batch.logits[128] == 1);
    assert(common_evaluation_batch(batch, tokens, plan, 0, 0, true, 1,
                common_evaluation_mask::generation_last) == 1);
    assert(batch.logits[128] == 0 && batch.logits[255] == 1);
    llama_batch_free(batch);
    for (const auto & invalid : {std::vector<int>{511, 256, 256}, {1024, 256, 300}, {1024, 0, 256}}) {
        bool rejected = false;
        try { common_evaluation_plan(invalid[0], invalid[1], invalid[2], -1); }
        catch (const std::invalid_argument &) { rejected = true; }
        assert(rejected);
    }
    const llama_token eos_token = 2;
    std::vector<llama_token> generated(127, 100);
    generated.back() = eos_token;
    assert(!common_evaluation_generation_complete(generated.size(), 126));
    generated.back() = 100;
    generated.push_back(eos_token);
    assert(common_evaluation_generation_complete(generated.size(), 127));
    assert(!common_evaluation_generation_complete(generated.size(), 128));
    assert(!common_evaluation_generation_complete(129, 127));
    const auto selected = common_evaluation_plan(tokens.size(), 256, 256, 1, 4);
    assert(selected.first_chunk == 4 && selected.n_chunks == 1 && selected.tail_tokens == 17);
    auto selected_batch = llama_batch_init(256, 0, 1);
    common_evaluation_batch(selected_batch, tokens, selected, 0, 0, true, 1,
            common_evaluation_mask::generation_last);
    assert(selected_batch.token[0] == 1 && selected_batch.token[1] == tokens[4 * 256 + 1]);
    assert(selected_batch.pos[0] == 0 && selected_batch.pos[255] == 255 && selected_batch.seq_id[0][0] == 0);
    std::vector<llama_token> ten_chunks(10 * 256 + 17);
    std::iota(ten_chunks.begin(), ten_chunks.end(), 100);
    for (int index = 0; index < 10; ++index) {
        const auto one = common_evaluation_plan(ten_chunks.size(), 256, 256, 1, index);
        common_evaluation_batch(selected_batch, ten_chunks, one, 0, 0, true, 1,
                common_evaluation_mask::generation_last);
        assert(one.first_chunk == index && one.n_chunks == 1 && one.tail_tokens == 17);
        assert(selected_batch.token[0] == 1);
        assert(selected_batch.token[1] == ten_chunks[index * 256 + 1]);
        assert(selected_batch.token[255] == ten_chunks[index * 256 + 255]);
    }
    llama_batch_free(selected_batch);
    for (const int index : {-1, 5}) {
        bool rejected = false;
        try { common_evaluation_plan(tokens.size(), 256, 256, 1, index); }
        catch (const std::invalid_argument &) { rejected = true; }
        assert(rejected);
    }
    const auto sampling = common_evaluation_e2e_sampling();
    assert(sampling.seed == 1234 && sampling.temp == 0 && sampling.top_k == 0);
    assert(sampling.top_p == 1 && sampling.min_p == 0 && sampling.penalty_repeat == 1);
    assert(sampling.penalty_last_n == 0 && sampling.grammar.empty() && sampling.logit_bias.empty());
    assert(!sampling.ignore_eos && sampling.typ_p == 1 && sampling.dry_multiplier == 0);
    std::vector<llama_token> forced(128, 100);
    common_evaluation_validate_forced(forced, 1000, true);
    for (int invalid = 0; invalid < 4; ++invalid) {
        auto ids = forced;
        if (invalid == 0) ids.resize(127);
        if (invalid == 1) ids[0] = -1;
        if (invalid == 2) ids[0] = 1000;
        bool rejected = false;
        try { common_evaluation_validate_forced(ids, 1000, invalid != 3); }
        catch (const std::invalid_argument &) { rejected = true; }
        assert(rejected);
    }
    std::puts("selected native chunk/BOS/tail; fixed greedy policy; forced count/range/CPU-only source PASS");
    std::puts("generation completion: short EOS incomplete; token128 EOS complete; exact decode count required PASS");
    std::puts("native workload parity: tokens/BOS/positions/seq IDs/logits/tail/batches PASS; generation mask differs PASS");
}
