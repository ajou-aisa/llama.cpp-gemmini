// H3-7A: real-model acceptance runner using the existing llama and backend APIs.
// This standalone test does not replace any graph, quantizer or hardware path.
// Capture runs observe scheduler assignments and save every sampled logit row.
// Timing runs leave the evaluation callback disabled and write files afterwards.
#include "llama.h"
#include "llama-context.h"
#include "ggml-cpu.h"
#include "ggml-gemmini.h"
#include "ggml-potal.h"
#include "json.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using json = nlohmann::ordered_json;
using Clock = std::chrono::steady_clock;

double milliseconds(Clock::duration duration) {
    return std::chrono::duration<double, std::milli>(duration).count();
}

struct Options {
    std::string model, prompt, output, forced;
    std::string backend = "potal", route = "serial";
    int prompt_tokens = 16, generated = 8, repeats = 1;
    bool capture = false, warmup = false, help = false, attention = false;
};

Options parse(int argc, char ** argv) {
    Options out;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--help") out.help = true;
        else if (key == "--capture") out.capture = true;
        else if (key == "--warmup") out.warmup = true;
        else if (key == "--no-warmup") out.warmup = false;
        else if (key == "--attention") out.attention = true;
        else {
            if (++i == argc) throw std::runtime_error("missing option value: " + key);
            const std::string value = argv[i];
            if (key == "--model") out.model = value;
            else if (key == "--prompt-file") out.prompt = value;
            else if (key == "--output") out.output = value;
            else if (key == "--backend") out.backend = value;
            else if (key == "--route") out.route = value;
            else if (key == "--force-tokens") out.forced = value;
            else if (key == "--prompt-tokens") out.prompt_tokens = std::stoi(value);
            else if (key == "--generate") out.generated = std::stoi(value);
            else if (key == "--repeat") out.repeats = std::stoi(value);
            else throw std::runtime_error("unknown option: " + key);
        }
    }
    if (out.help) return out;
    if (out.model.empty() || out.prompt.empty() || out.output.empty() ||
        out.prompt_tokens < 1 || out.prompt_tokens > 256 ||
        out.generated < 2 || out.generated > 32 || out.repeats < 1 || out.repeats > 3 ||
        (out.backend != "cpu" && out.backend != "potal") ||
        (out.route != "serial" && out.route != "paired" && out.route != "conservative")) {
        throw std::runtime_error("invalid H3-7A workload options; see --help");
    }
    return out;
}

std::ifstream input(const std::string & path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("cannot read " + path);
    stream.exceptions(std::ios::badbit);
    return stream;
}

std::ofstream output(const std::string & path) {
    std::ofstream stream;
    stream.exceptions(std::ios::failbit | std::ios::badbit);
    stream.open(path, std::ios::binary);
    return stream;
}

struct Observation {
    llama_context * context = nullptr;
    std::array<std::array<uint64_t, GGML_OP_COUNT>, 3> operations{};
    uint64_t cpu_hp1 = 0, unknown_matmul = 0, cpu_attention = 0, potal_attention = 0;
    json matmuls = json::array();
};

bool observe(ggml_tensor * tensor, bool ask, void * opaque) {
    if (!ask) return true;
    auto & data = *static_cast<Observation *>(opaque);
    const ggml_backend_t backend =
        ggml_backend_sched_get_tensor_backend(data.context->get_sched(), tensor);
    const int owner = backend && ggml_backend_is_potal(backend) ? 0 :
                      backend && std::strcmp(ggml_backend_name(backend), "CPU") == 0 ? 1 : 2;
    ++data.operations[owner][tensor->op];
    if (tensor->op == GGML_OP_MUL_MAT || tensor->op == GGML_OP_MUL_MAT_ID) {
        if (!backend) ++data.unknown_matmul;
        const bool hp1 = tensor->src[0] && tensor->src[0]->type == GGML_TYPE_Q8_HP1;
        if (owner == 1 && hp1) ++data.cpu_hp1;
        const auto role = ggml_gemmini_get_attention_role(tensor);
        if (role != GGML_GEMMINI_ATTENTION_NONE) {
            if (owner == 0) ++data.potal_attention;
            else if (owner == 1) ++data.cpu_attention;
        }
        data.matmuls.push_back({
            {"name", tensor->name}, {"backend", backend ? ggml_backend_name(backend) : "UNKNOWN"},
            {"weight_type", tensor->src[0] ? ggml_type_name(tensor->src[0]->type) : "NONE"},
            {"attention_role", role == GGML_GEMMINI_ATTENTION_QK ? "QK" :
                               role == GGML_GEMMINI_ATTENTION_PV ? "PV" : "NONE"},
            {"shape", {tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]}}
        });
    }
    // Count the assigned graph nodes without forcing a synchronization per node.
    return false;
}

json counters(const ggml_backend_potal_stats & before, const ggml_backend_potal_stats & after) {
    json data;
#define COUNTER(field) data[#field] = after.field - before.field
    COUNTER(hp1_nodes); COUNTER(hp1_completed); COUNTER(hp1_failed);
    COUNTER(prefill_shaped_nodes); COUNTER(decode_shaped_nodes);
    COUNTER(weight_cache_hits); COUNTER(weight_cache_misses);
    COUNTER(weight_cache_bypasses); COUNTER(weight_cache_uncached);
    COUNTER(weight_validations); COUNTER(weight_packs); COUNTER(weight_pack_bytes);
    COUNTER(stripe_pipeline_failed); COUNTER(stripes_semantic_completed);
    COUNTER(main_loop_ws); COUNTER(residual_loop_ws);
    COUNTER(main_fences); COUNTER(residual_fences);
    COUNTER(paired_stripes); COUNTER(paired_fallback_stripes);
    COUNTER(paired_main_loop_ws); COUNTER(paired_residual_loop_ws);
    COUNTER(paired_index_configs); COUNTER(paired_index_writes);
    COUNTER(paired_readbacks); COUNTER(paired_readback_tiles);
    COUNTER(paired_fences); COUNTER(paired_drains);
    COUNTER(scale_writes); COUNTER(scale_releases);
    COUNTER(radix_reconstructions); COUNTER(merges_completed); COUNTER(compensation_failed);
    COUNTER(gemm_failed);
    COUNTER(attention_qk_nodes); COUNTER(attention_pv_nodes);
    COUNTER(attention_completed); COUNTER(attention_failed);
    COUNTER(attention_limb_jobs); COUNTER(attention_residual_jobs); COUNTER(attention_loop_ws);
    COUNTER(attention_qk_limb_jobs); COUNTER(attention_pv_corrections); COUNTER(attention_bias_mvins);
#undef COUNTER
    data["weight_prepare_ms"] = (after.weight_pack_s - before.weight_pack_s) * 1000.0;
    data["weight_allocate_ms"] = (after.weight_alloc_s - before.weight_alloc_s) * 1000.0;
    data["weight_lookup_ms"] = (after.weight_lookup_s - before.weight_lookup_s) * 1000.0;
    data["scale_api_ms"] = (after.scale_s - before.scale_s) * 1000.0;
    data["issue_api_ms"] = (after.issue_s - before.issue_s) * 1000.0;
    data["fence_api_ms"] = (after.fence_s - before.fence_s) * 1000.0;
    data["activation_quant_ms"] = (after.act_quant_s - before.act_quant_s) * 1000.0;
    data["pipeline_wall_ms"] = (after.pipeline_wall_s - before.pipeline_wall_s) * 1000.0;
    data["attention_wall_ms"] = (after.attention_wall_s - before.attention_wall_s) * 1000.0;
    data["last_status"] = after.last_status;
    data["issuer_ready"] = after.issuer_ready;
    data["issuer_cpu"] = after.issuer_cpu;
    return data;
}

struct Batch {
    llama_batch value;
    explicit Batch(int tokens) : value(llama_batch_init(tokens, 0, 1)) {}
    ~Batch() { llama_batch_free(value); }
    Batch(const Batch &) = delete;
    Batch & operator=(const Batch &) = delete;
};

struct BackendLifetime {
    BackendLifetime() { ggml_backend_load_all(); llama_backend_init(); }
    ~BackendLifetime() { llama_backend_free(); }
};

class Request {
public:
    const Options & options;
    llama_context * context;
    const llama_vocab * vocabulary;
    ggml_backend_t potal = nullptr;
    std::vector<llama_token> prompt, forced;
    std::vector<bool> end_token;
    Observation observation;

    Request(const Options & settings, llama_context * ctx, const llama_vocab * vocab) :
        options(settings), context(ctx), vocabulary(vocab) {
        observation.context = context;
        auto stream = input(options.prompt);
        const std::string text((std::istreambuf_iterator<char>(stream)), {});
        const int count = -llama_tokenize(vocabulary, text.data(), int(text.size()), nullptr, 0, true, false);
        prompt.resize(count);
        if (llama_tokenize(vocabulary, text.data(), int(text.size()), prompt.data(), count, true, false) != count ||
            count < options.prompt_tokens) throw std::runtime_error("prompt tokenization/count failed");
        prompt.resize(options.prompt_tokens);
        const int vocab_size = llama_vocab_n_tokens(vocabulary);
        end_token.resize(vocab_size);
        for (int i = 0; i < vocab_size; ++i) end_token[i] = llama_vocab_is_eog(vocabulary, i);
        if (!options.forced.empty()) {
            auto tokens = input(options.forced);
            int token;
            while (tokens >> token) {
                if (token < 0 || token >= vocab_size) throw std::runtime_error("forced token out of range");
                forced.push_back(token);
            }
            if (!tokens.eof() || forced.size() < size_t(options.generated - 1))
                throw std::runtime_error("incomplete forced token trajectory");
        }
        auto * scheduler = context->get_sched();
        for (int i = 0; i < ggml_backend_sched_get_n_backends(scheduler); ++i) {
            auto backend = ggml_backend_sched_get_backend(scheduler, i);
            std::printf("H7_BACKEND name=%s\n", ggml_backend_name(backend));
            if (ggml_backend_is_potal(backend)) potal = backend;
        }
        if (options.backend == "potal" && !potal) throw std::runtime_error("POTAL backend absent");
        json ids = prompt;
        std::printf("H7_PROMPT count=%zu source_count=%d ids=%s\n", prompt.size(), count, ids.dump().c_str());
    }

    ggml_backend_potal_stats stats() const {
        ggml_backend_potal_stats data{};
        if (potal && !ggml_backend_potal_get_stats(potal, &data))
            throw std::runtime_error("POTAL statistics unavailable");
        return data;
    }

    json run(int generated, const std::string & label) {
        const int vocab_size = llama_vocab_n_tokens(vocabulary);
        const bool capture = options.capture && (options.warmup ? label == "warmup" : label != "warmup");
        std::vector<float> logits(capture ? size_t(generated) * vocab_size : 0);
        std::vector<llama_token> selected(generated);
        std::vector<double> emitted(generated);
        std::vector<json> steps(capture ? generated : 0);
        Batch batch(options.prompt_tokens);
        // decode resets the scheduler callback from cparams on every graph.
        // This version-bound test owns a non-const context and changes only its
        // callback fields between synchronized requests, leaving timing runs off.
        auto & callback_params = const_cast<llama_cparams &>(context->get_cparams());
        callback_params.cb_eval = capture ? observe : nullptr;
        callback_params.cb_eval_user_data = capture ? &observation : nullptr;
        const auto before = stats();
        const auto start = Clock::now();
        llama_kv_self_clear(context);
        double prefill_ms = 0;
        Clock::time_point prefill_end{};
        for (int step = 0; step < generated; ++step) {
            observation = Observation{};
            observation.context = context;
            batch.value.n_tokens = step == 0 ? options.prompt_tokens : 1;
            for (int i = 0; i < batch.value.n_tokens; ++i) {
                batch.value.token[i] = step == 0 ? prompt[i] :
                    forced.empty() ? selected[step - 1] : forced[step - 1];
                batch.value.pos[i] = step == 0 ? i : options.prompt_tokens + step - 1;
                batch.value.n_seq_id[i] = 1;
                batch.value.seq_id[i][0] = 0;
                batch.value.logits[i] = i + 1 == batch.value.n_tokens;
            }
            const auto compute_start = Clock::now();
            const int status = llama_decode(context, batch.value);
            if (status != 0) throw std::runtime_error("llama_decode step " + std::to_string(step) +
                                                     " returned " + std::to_string(status));
            float * row = llama_get_logits_ith(context, -1); // Synchronizes the real graph.
            const auto computed = Clock::now();
            if (!row) throw std::runtime_error("missing logits");
            if (step == 0) { prefill_ms = milliseconds(computed - compute_start); prefill_end = computed; }
            if (capture) std::copy(row, row + vocab_size, logits.begin() + size_t(step) * vocab_size);
            int best = -1;
            for (int i = 0; i < vocab_size; ++i) {
                if (!std::isfinite(row[i])) throw std::runtime_error("nonfinite logits");
                if (!end_token[i] && (best < 0 || row[i] > row[best])) best = i;
            }
            if (best < 0) throw std::runtime_error("no greedy token");
            selected[step] = best;
            std::printf("H7_TOKEN request=%s step=%d token_id=%d\n", label.c_str(), step, best);
            if (std::fflush(stdout) != 0) throw std::runtime_error("token output failed");
            emitted[step] = milliseconds(Clock::now() - start);
            if (capture) {
                const int expected = options.prompt_tokens + step;
                const int cached = llama_kv_self_n_tokens(context);
                const int position = llama_kv_self_seq_pos_max(context, 0);
                if (cached != expected || position != expected - 1)
                    throw std::runtime_error("KV cache count/position mismatch at step " + std::to_string(step));
                if (observation.matmuls.empty())
                    throw std::runtime_error("scheduler callback observed no MatMul nodes");
                if (observation.unknown_matmul) throw std::runtime_error("unknown MatMul backend");
                if (options.attention && options.backend == "potal" &&
                    (observation.cpu_attention != 0 || observation.potal_attention != 24))
                    throw std::runtime_error("Attention backend assignments are incomplete or used CPU");
                steps[step] = {{"step", step}, {"input_position", batch.value.pos[0]},
                    {"kv_tokens", cached}, {"kv_position_max", position}, {"selected", best},
                    {"emitted_ms", emitted[step]}, {"cpu_hp1_matmul", observation.cpu_hp1},
                    {"cpu_attention_matmul", observation.cpu_attention},
                    {"potal_attention_matmul", observation.potal_attention},
                    {"matmuls", observation.matmuls}, {"operations", json::array()}};
                for (int backend = 0; backend < 3; ++backend)
                    for (int op = 0; op < GGML_OP_COUNT; ++op)
                        if (observation.operations[backend][op])
                            steps[step]["operations"].push_back({
                                {"backend", backend == 0 ? "POTAL" : backend == 1 ? "CPU" : "OTHER"},
                                {"op", ggml_op_name(ggml_op(op))}, {"count", observation.operations[backend][op]}});
            }
        }
        const auto finished = Clock::now();
        const auto after = stats();
        const json hardware = counters(before, after);
        const int cached = llama_kv_self_n_tokens(context);
        const int position = llama_kv_self_seq_pos_max(context, 0);
        if (cached != options.prompt_tokens + generated - 1 || position != cached - 1)
            throw std::runtime_error("final KV cache mismatch");
        json result = {
            {"schema", "h3-7a.request.v1"}, {"backend", options.backend}, {"route", options.route},
            {"label", label}, {"capture", capture}, {"forced_inputs", !forced.empty()},
            {"forced_token_ids", forced}, {"prompt_ids", prompt},
            {"generated_ids", selected}, {"vocab_size", vocab_size},
            {"ignore_eos", true}, {"sampling", "greedy_lowest_id_tie"}, {"kv_tokens", cached},
            {"kv_position_max", position}, {"prefill_ms", prefill_ms}, {"ttft_ms", emitted.front()},
            {"first_generation_ms", emitted.front() - milliseconds(prefill_end - start)},
            {"tpot_ms", (emitted.back() - emitted.front()) / (generated - 1)},
            {"emitted_ms", emitted}, {"total_ms", emitted.back()},
            {"request_return_ms", milliseconds(finished - start)},
            {"hardware", hardware}, {"steps", steps}
        };
        const std::string prefix = options.output + "." + label;
        {
            auto file = output(prefix + ".json");
            file << result.dump(2) << '\n';
        }
        {
            auto file = output(prefix + ".tokens");
            for (auto token : selected) file << token << '\n';
        }
        if (capture) {
            static_assert(sizeof(float) == 4, "logit capture requires binary32");
            auto file = output(prefix + ".logits.f32");
            file.write(reinterpret_cast<const char *>(logits.data()), std::streamsize(logits.size() * sizeof(float)));
        }
        if (options.backend == "potal") {
            if (after.hp1_completed <= before.hp1_completed ||
                after.hp1_failed != before.hp1_failed || after.gemm_failed != before.gemm_failed ||
                after.stripe_pipeline_failed != before.stripe_pipeline_failed ||
                hardware["scale_writes"] != hardware["scale_releases"] ||
                hardware["hp1_completed"].get<uint64_t>() != uint64_t(49 * generated) ||
                hardware["main_loop_ws"].get<uint64_t>() + hardware["paired_main_loop_ws"].get<uint64_t>() == 0 ||
                hardware["residual_loop_ws"].get<uint64_t>() + hardware["paired_residual_loop_ws"].get<uint64_t>() == 0)
                throw std::runtime_error("incomplete or failed GPT2 hardware execution");
            if (options.route != "serial" && hardware["paired_stripes"].get<uint64_t>() == 0)
                throw std::runtime_error("requested Paired route executed no Paired stripe");
            if (options.attention &&
                (hardware["attention_qk_nodes"].get<uint64_t>() != uint64_t(12 * generated) ||
                 hardware["attention_pv_nodes"].get<uint64_t>() != uint64_t(12 * generated) ||
                 hardware["attention_completed"].get<uint64_t>() != uint64_t(24 * generated) ||
                 hardware["attention_failed"].get<uint64_t>() != 0 ||
                 hardware["attention_qk_limb_jobs"].get<uint64_t>() == 0 ||
                 hardware["attention_bias_mvins"].get<uint64_t>() == 0))
                throw std::runtime_error("incomplete or failed PoTal Attention hardware execution");
        } else if (after.hp1_completed != before.hp1_completed) {
            throw std::runtime_error("CPU reference unexpectedly used POTAL");
        }
        std::printf("H7_HARDWARE request=%s counters=%s\n", label.c_str(), hardware.dump().c_str());
        std::printf("H7_REQUEST_EXECUTED label=%s prompt=%zu generated=%d capture=%d ttft_ms=%.6f tpot_ms=%.6f total_ms=%.6f\n",
                    label.c_str(), prompt.size(), generated, capture, emitted.front(),
                    (emitted.back() - emitted.front()) / (generated - 1), emitted.back());
        return result;
    }
};
} // namespace

int main(int argc, char ** argv) try {
    const Options options = parse(argc, argv);
    if (options.help) {
        std::puts("potal-full-inference --model GGUF --prompt-file TEXT --output PREFIX\n"
                  "  [--backend potal|cpu] [--route serial|paired|conservative]\n"
                  "  [--prompt-tokens 1..256] [--generate 2..32] [--repeat 1..3]\n"
                  "  [--capture] [--no-warmup|--warmup] [--attention] [--force-tokens FILE]\n"
                  "Warmup is a separate complete two-token request.\n"
                  "With --capture --warmup, capture one full warmup request; measure uncaptured repeats.\n"
                  "Capture timings are diagnostic, not primary performance samples.");
        return 0;
    }
    if (std::getenv("POTAL_PLANNING_PROFILE") || std::getenv("POTAL_PAIRED_PROFILE") ||
        std::getenv("POTAL_RAW_TRACE")) throw std::runtime_error("disable diagnostic environment variables");
    if (setenv("GGML_POTAL_PAIRED_AUTO", options.route == "serial" ? "OFF" : "ON", 1) != 0 ||
        setenv("GGML_POTAL_PAIRED_FENCES", options.route == "conservative" ? "CONSERVATIVE" : "RTL", 1) != 0)
        throw std::runtime_error("route environment setup failed");
    BackendLifetime lifetime;
    auto params = llama_model_default_params();
    params.n_gpu_layers = 0;
    ggml_backend_dev_t cpu_devices[] = {nullptr};
    llama_model_tensor_buft_override cpu_buffers[] = {{".*", ggml_backend_cpu_buffer_type()}, {nullptr, nullptr}};
    if (options.backend == "cpu") { params.devices = cpu_devices; params.tensor_buft_overrides = cpu_buffers; }
    const auto loading = Clock::now();
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
        llama_model_load_from_file(options.model.c_str(), params), llama_model_free);
    if (!model) throw std::runtime_error("model loading failed");
    const double load_ms = milliseconds(Clock::now() - loading);
    char architecture[64];
    if (llama_model_meta_val_str(model.get(), "general.architecture", architecture, sizeof(architecture)) < 0 ||
        std::strcmp(architecture, "gpt2") != 0 || llama_model_n_params(model.get()) != 124439808)
        throw std::runtime_error("this acceptance runner requires the audited GPT2 checkpoint");
    auto context_params = llama_context_default_params();
    context_params.n_ctx = 512;
    context_params.n_batch = 256;
    context_params.n_ubatch = 256;
    context_params.n_threads = context_params.n_threads_batch = 3;
    context_params.offload_kqv = options.attention && options.backend == "potal";
    context_params.flash_attn = false;
    context_params.op_offload = options.backend == "potal";
    context_params.no_perf = false;
    const auto initializing = Clock::now();
    std::unique_ptr<llama_context, decltype(&llama_free)> context(
        llama_init_from_model(model.get(), context_params), llama_free);
    if (!context) throw std::runtime_error("context initialization failed");
    const double context_ms = milliseconds(Clock::now() - initializing);
    std::printf("H7_MODEL_LOADED parameters=%llu model_load_ms=%.6f context_init_ms=%.6f\n",
                static_cast<unsigned long long>(llama_model_n_params(model.get())), load_ms, context_ms);
    Request request(options, context.get(), llama_model_get_vocab(model.get()));
    json manifest = {{"model", options.model}, {"parameters", llama_model_n_params(model.get())},
        {"model_load_ms", load_ms}, {"context_init_ms", context_ms}, {"prompt_ids", request.prompt},
        {"n_ctx", 512}, {"n_batch", 256}, {"n_ubatch", 256}, {"threads", 3}, {"kv_type", "F16"},
        {"attention_offload", context_params.offload_kqv}, {"warmup", options.warmup}};
    { auto file = output(options.output + ".model.json"); file << manifest.dump(2) << '\n'; }
    if (options.warmup) request.run(options.capture ? options.generated : 2, "warmup");
    for (int i = 0; i < options.repeats; ++i) request.run(options.generated, "request" + std::to_string(i));
    std::puts("H7_RUN_COMPLETE");
    return 0;
} catch (const std::exception & error) {
    std::fprintf(stderr, "H7_INFERENCE_ERROR %s\n", error.what());
    return 1;
}
