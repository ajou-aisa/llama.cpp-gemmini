#include "llama.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

static void require(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

// Experimental replay seam: keep the native model, tokenizer, RoPE, GQA, mask,
// softmax and linear path. The callback replaces only completed QK/PV outputs.
// Original floating-point dots still execute, so decode wall time is NOT a
// measurement of an integrated optimized attention backend.
using replay_hook = int (*)(int, int, const float *, const float *, float *,
                            const int64_t *, const int64_t *, const int64_t *, void *);

static thread_local std::string replay_error;

static std::vector<float> tensor_values(const ggml_tensor * t) {
    require(t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_F16, "Replay requires floating operands");
    std::vector<uint8_t> bytes(ggml_nbytes(t));
    ggml_backend_tensor_get(t, bytes.data(), 0, bytes.size());
    std::vector<float> result(ggml_nelements(t));
    size_t next = 0;
    for (int64_t h3 = 0; h3 < t->ne[3]; ++h3)
        for (int64_t h = 0; h < t->ne[2]; ++h)
            for (int64_t r = 0; r < t->ne[1]; ++r)
                for (int64_t c = 0; c < t->ne[0]; ++c) {
                    const auto offset = h3*t->nb[3] + h*t->nb[2] + r*t->nb[1] + c*t->nb[0];
                    if (t->type == GGML_TYPE_F32) {
                        std::memcpy(&result[next], bytes.data() + offset, sizeof(float));
                    } else {
                        ggml_fp16_t half;
                        std::memcpy(&half, bytes.data() + offset, sizeof(half));
                        result[next] = ggml_fp16_to_fp32(half);
                    }
                    ++next;
                }
    return result;
}

struct replay_session {
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model{nullptr, llama_model_free};
    std::unique_ptr<llama_context, decltype(&llama_free)> context{nullptr, llama_free};
    replay_hook hook = nullptr;
    void * data = nullptr;
    int callbacks = 0;
    bool failed = false;
};

static bool replay_callback(ggml_tensor * t, bool ask, void * opaque) {
    auto & session = *static_cast<replay_session *>(opaque);
    const bool interested = t->op == GGML_OP_MUL_MAT && t->ne[2] > 1 &&
                            t->src[0] && !ggml_is_quantized(t->src[0]->type);
    if (ask) return interested;
    try {
        require(interested && t->type == GGML_TYPE_F32 && ggml_is_contiguous(t), "Unexpected replay output");
        const int role = t->src[1]->op == GGML_OP_SOFT_MAX ? 1 : 0;
        require(role == session.callbacks % 2, "Expected alternating QK and PV nodes");
        require(t->ne[3] == 1, "Replay currently requires one sequence");
        auto stationary = tensor_values(t->src[0]);
        auto streamed = tensor_values(t->src[1]);
        std::vector<float> output(ggml_nelements(t));
        require(session.hook(role, session.callbacks / 2, stationary.data(), streamed.data(), output.data(),
                             t->src[0]->ne, t->src[1]->ne, t->ne, session.data) == 0,
                "Attention replay hook failed");
        ggml_backend_tensor_set(t, output.data(), 0, output.size()*sizeof(float));
        ++session.callbacks;
        return true;
    } catch (const std::exception & error) {
        replay_error = error.what();
        session.failed = true;
        return false;
    }
}

extern "C" const char * potal_replay_error() { return replay_error.c_str(); }

extern "C" void * potal_replay_open(const char * path, int count, int threads, replay_hook hook, void * data) {
    try {
        require(count >= 32 && count <= 1024, "Context outside 32..1024");
        ggml_backend_load_all();
        llama_backend_init();
        auto session = std::make_unique<replay_session>();
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = 0;
        session->model.reset(llama_model_load_from_file(path, mp));
        require(bool(session->model), "Model load");
        session->hook = hook;
        session->data = data;
        auto cp = llama_context_default_params();
        cp.n_ctx = cp.n_batch = cp.n_ubatch = count;
        cp.n_threads = cp.n_threads_batch = threads;
        cp.type_k = cp.type_v = GGML_TYPE_F32;
        cp.flash_attn = false;
        cp.offload_kqv = false;
        if (hook) { cp.cb_eval = replay_callback; cp.cb_eval_user_data = session.get(); }
        session->context.reset(llama_init_from_model(session->model.get(), cp));
        require(bool(session->context), "Context creation");
        return session.release();
    } catch (const std::exception & error) { replay_error = error.what(); return nullptr; }
}

extern "C" void potal_replay_close(void * handle) { delete static_cast<replay_session *>(handle); }

extern "C" int potal_replay_tokenize(void * handle, const char * text, int length, int32_t * tokens, int capacity) {
    const auto * vocab = llama_model_get_vocab(static_cast<replay_session *>(handle)->model.get());
    return llama_tokenize(vocab, text, length, tokens, capacity, true, false);
}

extern "C" int potal_replay_vocabulary(void * handle) {
    return llama_vocab_n_tokens(llama_model_get_vocab(static_cast<replay_session *>(handle)->model.get()));
}

extern "C" const float * potal_replay_logits(void * handle) {
    return llama_get_logits(static_cast<replay_session *>(handle)->context.get());
}

extern "C" int potal_replay_eval(void * handle, const int32_t * tokens, int count, double * nll) {
    auto & session = *static_cast<replay_session *>(handle);
    auto batch = llama_batch_init(count, 0, 1);
    try {
        session.callbacks = 0;
        session.failed = false;
        llama_kv_self_clear(session.context.get());
        batch.n_tokens = count;
        for (int i = 0; i < count; ++i) {
            batch.token[i] = tokens[i]; batch.pos[i] = i; batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0; batch.logits[i] = i >= count/2 && i < count-1;
        }
        require(llama_decode(session.context.get(), batch) == 0 && !session.failed, "Replay decode failed");
        if (session.hook) require(session.callbacks == 2*llama_model_n_layer(session.model.get()),
                                  "Missing attention replay callbacks");
        const int vocabulary = potal_replay_vocabulary(handle);
        for (int row = count/2; row < count-1; ++row) {
            const auto * logits = llama_get_logits_ith(session.context.get(), row);
            require(logits != nullptr, "Missing logits");
            const double maximum = *std::max_element(logits, logits+vocabulary);
            double total = 0;
            for (int col = 0; col < vocabulary; ++col) total += std::exp(double(logits[col])-maximum);
            nll[row-count/2] = std::log(total)+maximum-logits[tokens[row+1]];
        }
        llama_batch_free(batch);
        return session.callbacks;
    } catch (const std::exception & error) {
        if (!session.failed) replay_error = error.what();
        llama_batch_free(batch);
        return -1;
    }
}

// CPU component comparison. Modes 1 and 2 use identical sparse metadata; only
// mode 2 specializes +/-1. Other digits retain exact multiplication. K<=32 and
// signed nibble inputs keep every raw partial well inside int32.
extern "C" int potal_replay_dot(const int8_t * a, const int8_t * b, int32_t * out,
                                int m, int k, int n, int mode, uint64_t * stats) {
    if (m < 0 || k < 0 || k > 32 || n < 0 || mode < 0 || mode > 2) return -1;
    using clock = std::chrono::steady_clock;
    auto begin = clock::now();
    struct entry { uint8_t column; int8_t value; };
    std::vector<entry> entries;
    std::vector<uint32_t> row_start;
    if (mode) {
        entries.reserve(size_t(m)*k);
        row_start.reserve(m+1);
        for (int i = 0; i < m; ++i) {
            row_start.push_back(entries.size());
            for (int r = 0; r < k; ++r) if (a[i*k+r]) entries.push_back({uint8_t(r), a[i*k+r]});
        }
        row_start.push_back(entries.size());
    }
    auto packed = clock::now();
    std::fill(out, out+size_t(m)*n, 0);
    for (int i = 0; i < m; ++i) {
        auto * target = out+size_t(i)*n;
        if (!mode) {
            for (int r = 0; r < k; ++r) {
                const int32_t value = a[i*k+r];
                const auto * source = b+size_t(r)*n;
                for (int j = 0; j < n; ++j) target[j] += value*source[j];
            }
        } else {
            for (uint32_t p = row_start[i]; p < row_start[i+1]; ++p) {
                const auto * source = b+size_t(entries[p].column)*n;
                const int32_t value = entries[p].value;
                if (mode == 2 && value == 1) {
                    for (int j = 0; j < n; ++j) target[j] += source[j];
                } else if (mode == 2 && value == -1) {
                    for (int j = 0; j < n; ++j) target[j] -= source[j];
                } else {
                    for (int j = 0; j < n; ++j) target[j] += value*source[j];
                }
            }
        }
    }
    auto end = clock::now();
    stats[0] = std::chrono::duration_cast<std::chrono::nanoseconds>(packed-begin).count();
    stats[1] = std::chrono::duration_cast<std::chrono::nanoseconds>(end-packed).count();
    stats[2] = entries.size()*sizeof(entry)+row_start.size()*sizeof(uint32_t);
    stats[3] = entries.capacity()*sizeof(entry)+row_start.capacity()*sizeof(uint32_t);
    return 0;
}

#ifndef POTAL_NATIVE_LIBRARY
int main(int argc, char ** argv) {
    try {
        require(argc == 4, "usage: native-model MODEL TOKENS CTX");
        const int count = std::stoi(argv[3]);
        require(count >= 32 && count <= 1024, "Context outside 32..1024");
        std::vector<llama_token> tokens(count);
        std::ifstream input(argv[2], std::ios::binary);
        input.read(reinterpret_cast<char *>(tokens.data()), count * sizeof(llama_token));
        require(input.good(), "Token input too short");
        ggml_backend_load_all();
        llama_backend_init();
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = 0;
        auto * model = llama_model_load_from_file(argv[1], mp);
        require(model != nullptr, "Model load");
        auto cp = llama_context_default_params();
        cp.n_ctx = cp.n_batch = cp.n_ubatch = count;
        cp.n_threads = cp.n_threads_batch = 2;
        cp.type_k = cp.type_v = GGML_TYPE_F32;
        cp.flash_attn = false;
        cp.offload_kqv = false;
        auto * context = llama_init_from_model(model, cp);
        require(context != nullptr, "Context creation");
        auto batch = llama_batch_init(count, 0, 1);
        batch.n_tokens = count;
        for (int i = 0; i < count; ++i) {
            batch.token[i] = tokens[i];
            batch.pos[i] = i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = i >= count / 2 && i < count - 1;
        }
        require(llama_decode(context, batch) == 0, "Decode failed");
        const int vocabulary = llama_vocab_n_tokens(llama_model_get_vocab(model));
        double nll = 0;
        for (int row = count / 2; row < count - 1; ++row) {
            const auto * logits = llama_get_logits_ith(context, row);
            require(logits != nullptr, "Missing logits");
            const double maximum = *std::max_element(logits, logits + vocabulary);
            double total = 0;
            for (int col = 0; col < vocabulary; ++col) total += std::exp(double(logits[col]) - maximum);
            nll += std::log(total) + maximum - logits[tokens[row + 1]];
        }
        std::cout << std::setprecision(17) << "{\"tokens\":" << count / 2 - 1
                  << ",\"nll\":" << nll << ",\"ppl\":" << std::exp(nll / (count / 2 - 1)) << "}\n";
        llama_batch_free(batch);
        llama_free(context);
        llama_model_free(model);
        llama_backend_free();
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
#endif
