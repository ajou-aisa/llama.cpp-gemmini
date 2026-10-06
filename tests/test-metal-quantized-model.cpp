#include <llama.h>
#include <ggml-backend.h>

#include "metal-quantized-fixtures.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

double seconds(Clock::time_point begin) {
    return std::chrono::duration<double>(Clock::now() - begin).count();
}

void require(bool condition, const std::string & message) {
    if (!condition) throw std::runtime_error(message);
}

uint32_t bits(float value) {
    uint32_t result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

uint64_t logits_hash(const float * values, size_t count) {
    uint64_t hash = UINT64_C(14695981039346656037);
    for (size_t i = 0; i < count; ++i) {
        const uint32_t word = bits(values[i]);
        for (unsigned byte = 0; byte < 4; ++byte) {
            hash ^= (word >> (8 * byte)) & 255u;
            hash *= UINT64_C(1099511628211);
        }
    }
    return hash;
}

struct HostTensor {
    ggml_tensor tensor;
    std::vector<uint8_t> storage;

    explicit HostTensor(const ggml_tensor * source) : tensor(*source), storage(ggml_nbytes(source)) {
        ggml_backend_tensor_get(source, storage.data(), 0, storage.size());
        tensor.data = storage.data();
        tensor.buffer = nullptr;
        tensor.view_src = nullptr;
        tensor.view_offs = 0;
    }
};

bool quantized_matmul(const ggml_tensor * tensor) {
    return tensor && (tensor->op == GGML_OP_MUL_MAT || tensor->op == GGML_OP_MUL_MAT_ID) &&
           tensor->src[0] && ggml_is_quantized(tensor->src[0]->type);
}

struct Validation {
    unsigned threads = 4;
    bool publish_reference = false;
    bool failed = false;
    std::string error;
    std::string stage;
    size_t matrices = 0;
    size_t values = 0;
    size_t prefill_matrices = 0;
    size_t decode_matrices = 0;
    uint64_t dense_before = 0;
    double reference_seconds = 0;
    double producer_seconds = 0;
};

bool validate_matmul(ggml_tensor * tensor, bool ask, void * opaque) {
    auto & state = *static_cast<Validation *>(opaque);
    if (!quantized_matmul(tensor)) return ask ? false : true;
    if (state.failed) return false;
    if (ask) {
        state.dense_before = ggml_metal_quantized_get_stats().dense_launches;
        return true;
    }
    try {
        require(tensor->op == GGML_OP_MUL_MAT && ggml_metal_quantized_supports_op(tensor),
                "Unsupported quantized model matmul: " + std::string(tensor->name));
        const auto stats = ggml_metal_quantized_get_stats();
        require(stats.dense_launches == state.dense_before + 1,
                "Model matmul did not execute exactly one custom Metal dense kernel: " + std::string(tensor->name));
        require(stats.failed_calls == 0, "A custom Metal kernel reported failure");
        require(tensor->buffer != nullptr, "Model output has no backend buffer");
        const auto device = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(tensor->buffer));
        require(device && std::strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(device)), "Metal") == 0,
                "Quantized model output is not placed on Metal: " + std::string(tensor->name));

        HostTensor weights(tensor->src[0]);
        HostTensor activation(tensor->src[1]);
        HostTensor actual(tensor);
        ggml_metal_quantized_payload * raw = nullptr;
        char error[512]{};
        const auto producer_begin = Clock::now();
        require(ggml_metal_quantized_prepare(&weights.tensor, &activation.tensor, &raw, error, sizeof(error)),
                "Reference CPU producer failed for " + std::string(tensor->name) + ": " + error);
        state.producer_seconds += seconds(producer_begin);
        std::unique_ptr<ggml_metal_quantized_payload, decltype(&ggml_metal_quantized_free)> payload(raw, ggml_metal_quantized_free);
        const auto * view = ggml_metal_quantized_get_view(payload.get());
        require(view && view->m == size_t(tensor->ne[1]) && view->n == size_t(tensor->ne[0]),
                "Model reference payload shape mismatch");
        std::vector<const ggml_metal_quantized_request *> requests(view->request_count);
        for (size_t i = 0; i < requests.size(); ++i) requests[i] = ggml_metal_quantized_get_request(payload.get(), i);
        const auto reference_begin = Clock::now();
        const auto expected = metal_quantized_fixtures::reference(*view, requests, false, state.threads);
        state.reference_seconds += seconds(reference_begin);
        require(expected.valid, "Independent CPU arithmetic overflow for " + std::string(tensor->name));

        for (size_t row = 0; row < view->m; ++row) {
            for (size_t column = 0; column < view->n; ++column) {
                float observed;
                std::memcpy(&observed, actual.storage.data() + row * tensor->nb[1] + column * tensor->nb[0], sizeof(observed));
                const float wanted = expected.output[row * view->n + column];
                if (bits(observed) != bits(wanted)) {
                    std::fprintf(stderr,
                        "METAL_MODEL_MISMATCH stage=%s tensor=%s weight=%s type=%s m=%zu n=%zu k=%zu row=%zu column=%zu actual=%a actual_bits=%08x expected=%a expected_bits=%08x\n",
                        state.stage.c_str(), tensor->name, tensor->src[0]->name, ggml_type_name(tensor->src[0]->type),
                        view->m, view->n, view->k, row, column, observed, bits(observed), wanted, bits(wanted));
                    throw std::runtime_error("Model quantized matmul differs from independent ordered CPU oracle");
                }
            }
        }
        if (state.publish_reference) {
            // Test-only replay propagates independently calculated outputs after
            // proving the production Metal kernel produced identical bytes.
            for (size_t row = 0; row < view->m; ++row)
                ggml_backend_tensor_set(tensor, expected.output.data() + row * view->n,
                                        row * tensor->nb[1], view->n * sizeof(float));
        }
        ++state.matrices;
        state.values += expected.output.size();
        if (view->m == 1) ++state.decode_matrices;
        else ++state.prefill_matrices;
        std::printf("METAL_MODEL_MATMUL pass=%s stage=%s tensor=%s weight=%s m=%zu n=%zu k=%zu values=%zu gpu_dense_launch=%llu f32=bitwise PASS\n",
                    state.publish_reference ? "reference_replay" : "gpu", state.stage.c_str(), tensor->name,
                    tensor->src[0]->name, view->m, view->n, view->k, expected.output.size(),
                    static_cast<unsigned long long>(stats.dense_launches));
        std::fflush(stdout);
        return true;
    } catch (const std::exception & error) {
        state.failed = true;
        state.error = error.what();
        std::fprintf(stderr, "METAL_MODEL_CALLBACK_FAIL %s\n", error.what());
        return false;
    }
}

struct Logits {
    std::vector<float> values;
    llama_token greedy = -1;
};

struct Run {
    std::vector<Logits> logits;
    Validation validation;
    ggml_metal_quantized_stats stats{};
};

Run trajectory(llama_model * model, const std::vector<llama_token> & tokens,
               unsigned prefill, unsigned threads, bool publish_reference) {
    Run run;
    run.validation.threads = threads;
    run.validation.publish_reference = publish_reference;
    auto params = llama_context_default_params();
    params.n_ctx = 64;
    params.n_batch = prefill;
    params.n_ubatch = prefill;
    params.n_threads = threads;
    params.n_threads_batch = threads;
    params.type_k = params.type_v = GGML_TYPE_F16;
    params.offload_kqv = false;
    params.flash_attn = false;
    params.no_perf = true;
    params.cb_eval = validate_matmul;
    params.cb_eval_user_data = &run.validation;
    std::unique_ptr<llama_context, decltype(&llama_free)> context(llama_init_from_model(model, params), llama_free);
    require(context != nullptr, "Could not create model validation context");
    const size_t vocabulary = static_cast<size_t>(llama_vocab_n_tokens(llama_model_get_vocab(model)));
    ggml_metal_quantized_reset_stats();
    llama_batch batch = llama_batch_init(static_cast<int32_t>(prefill), 0, 1);
    require(batch.token && batch.logits && batch.seq_id && batch.pos, "Could not allocate token batch");
    try {
        for (unsigned step = 0; step < 3; ++step) {
            const unsigned begin = step == 0 ? 0 : prefill + step - 1;
            const unsigned count = step == 0 ? prefill : 1;
            run.validation.stage = step == 0 ? "prefill" : "decode_" + std::to_string(step);
            batch.n_tokens = static_cast<int32_t>(count);
            for (unsigned i = 0; i < count; ++i) {
                batch.token[i] = tokens[begin + i];
                batch.pos[i] = static_cast<llama_pos>(begin + i);
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i] = 1;
            }
            const int status = llama_decode(context.get(), batch);
            llama_synchronize(context.get());
            require(!run.validation.failed, run.validation.error);
            require(status == 0, "llama_decode failed with status " + std::to_string(status));
            for (unsigned i = 0; i < count; ++i) {
                const float * data = llama_get_logits_ith(context.get(), static_cast<int32_t>(i));
                require(data != nullptr, "Missing requested token logits");
                Logits snapshot;
                snapshot.values.assign(data, data + vocabulary);
                for (float value : snapshot.values) require(std::isfinite(value), "Nonfinite model logit");
                snapshot.greedy = static_cast<llama_token>(std::max_element(snapshot.values.begin(), snapshot.values.end()) - snapshot.values.begin());
                std::printf("METAL_MODEL_LOGITS pass=%s token_position=%u forced_token=%d greedy=%d vocabulary=%zu fnv1a64=%016llx\n",
                            publish_reference ? "reference_replay" : "gpu", begin + i, tokens[begin + i],
                            snapshot.greedy, vocabulary, static_cast<unsigned long long>(logits_hash(data, vocabulary)));
                run.logits.push_back(std::move(snapshot));
            }
        }
        llama_batch_free(batch);
    } catch (...) {
        llama_batch_free(batch);
        throw;
    }
    run.stats = ggml_metal_quantized_get_stats();
    require(run.validation.prefill_matrices > 0 && run.validation.decode_matrices > 0,
            "Both prefill and decode must execute custom quantized Metal matmuls");
    require(run.stats.failed_calls == 0 && run.stats.dense_launches == run.validation.matrices &&
            run.stats.block_calls + run.stats.hp1_calls == run.validation.matrices,
            "Model quantized operation count differs from verified actual Metal dispatch count");
    std::printf("METAL_MODEL_PASS pass=%s matrices=%zu values=%zu prefill_matrices=%zu decode_matrices=%zu dense_launches=%llu residual_launches=%llu producer_reference_seconds=%.6f cpu_oracle_seconds=%.6f PASS\n",
                publish_reference ? "reference_replay" : "gpu", run.validation.matrices, run.validation.values,
                run.validation.prefill_matrices, run.validation.decode_matrices,
                static_cast<unsigned long long>(run.stats.dense_launches),
                static_cast<unsigned long long>(run.stats.residual_launches),
                run.validation.producer_seconds, run.validation.reference_seconds);
    return run;
}

unsigned small_positive(const std::string & text) {
    size_t consumed = 0;
    const unsigned long result = std::stoul(text, &consumed);
    require(consumed == text.size() && result > 0 && result <= 64, "Expected a positive integer no greater than 64");
    return static_cast<unsigned>(result);
}

} // namespace

int main(int argc, char ** argv) {
    try {
        std::string path;
        unsigned prefill = 8, threads = 4;
        for (int i = 1; i < argc; ++i) {
            const std::string flag = argv[i];
            if (flag == "--help") {
                std::printf("Usage: %s --model original.gguf [--prefill 8|16] [--threads 4]\n", argv[0]);
                return 0;
            }
            require(i + 1 < argc, "Missing value for " + flag);
            const std::string value = argv[++i];
            if (flag == "--model") path = value;
            else if (flag == "--prefill") prefill = small_positive(value);
            else if (flag == "--threads") threads = small_positive(value);
            else throw std::runtime_error("Unknown argument " + flag);
        }
        require(!path.empty() && (prefill == 8 || prefill == 16), "Specify --model and --prefill 8 or 16");
        require(ggml_metal_quantized_enabled(), "Custom Metal arithmetic must be enabled");
        llama_backend_init();
        ggml_backend_load_all();
        const auto metal = ggml_backend_reg_by_name("Metal");
        require(metal && ggml_backend_reg_dev_count(metal) > 0, "Actual Metal device is required; CPU substitution is forbidden");
        ggml_backend_dev_t devices[] = {ggml_backend_reg_dev_get(metal, 0), nullptr};
        auto model_params = llama_model_default_params();
        model_params.devices = devices;
        model_params.n_gpu_layers = 999;
        model_params.split_mode = LLAMA_SPLIT_MODE_NONE;
        model_params.use_mmap = true;
        model_params.check_tensors = true;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(llama_model_load_from_file(path.c_str(), model_params), llama_model_free);
        require(model != nullptr, "Could not load original model GGUF");
        const auto profile = ggml_metal_quantized_get_profile();
        std::printf("METAL_MODEL_PROFILE model=%s device=%s mode=%u bits=%u dim=%u residual=%u cpu_producer=1 cpu_kv_kqv=1 prefill=%u decode=2 oracle_threads=%u\n",
                    path.c_str(), ggml_backend_dev_name(devices[0]), static_cast<unsigned>(profile.mode),
                    profile.bits, profile.dim, profile.residual_enabled, prefill, threads);
        const std::string prompt = "Metal quantized arithmetic keeps the original weight codes and scales. The same token sequence is evaluated twice so every matrix output and every next token logit can be checked exactly.";
        std::vector<llama_token> tokens(prompt.size() + 8);
        const int count = llama_tokenize(llama_model_get_vocab(model.get()), prompt.c_str(), static_cast<int32_t>(prompt.size()),
                                         tokens.data(), static_cast<int32_t>(tokens.size()), true, false);
        require(count >= static_cast<int>(prefill + 2), "Fixed prompt must provide the full forced token trajectory");
        tokens.resize(prefill + 2);
        std::printf("METAL_MODEL_TOKENS ids=");
        for (size_t i = 0; i < tokens.size(); ++i) std::printf("%s%d", i ? "," : "", tokens[i]);
        std::printf("\n");
        const auto gpu = trajectory(model.get(), tokens, prefill, threads, false);
        const auto replay = trajectory(model.get(), tokens, prefill, threads, true);
        require(gpu.logits.size() == replay.logits.size(), "Logit snapshot counts differ");
        size_t compared = 0;
        for (size_t position = 0; position < gpu.logits.size(); ++position) {
            const auto & original = gpu.logits[position];
            const auto & reference = replay.logits[position];
            require(original.values.size() == reference.values.size(), "Vocabulary sizes differ");
            for (size_t token = 0; token < original.values.size(); ++token) {
                if (bits(original.values[token]) != bits(reference.values[token])) {
                    std::fprintf(stderr, "METAL_MODEL_LOGIT_MISMATCH position=%zu token=%zu gpu=%a replay=%a gpu_bits=%08x replay_bits=%08x\n",
                                 position, token, original.values[token], reference.values[token],
                                 bits(original.values[token]), bits(reference.values[token]));
                    throw std::runtime_error("Same-trajectory logits differ bitwise after independent reference replay");
                }
            }
            require(original.greedy == reference.greedy, "Greedy token diverges on the fixed trajectory");
            compared += original.values.size();
        }
        std::printf("METAL_MODEL_RESULT snapshots=%zu logits_compared=%zu verified_matrices=%zu logits=bitwise greedy_divergences=0 actual_gpu=1 reference_replay=1 PASS\n",
                    gpu.logits.size(), compared, gpu.validation.matrices + replay.validation.matrices);
        model.reset();
        llama_backend_free();
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "METAL_MODEL_RESULT FAIL: %s\n", error.what());
        return 1;
    }
}
