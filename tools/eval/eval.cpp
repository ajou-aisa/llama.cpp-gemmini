#include "common.h"
#include "evaluation-workload.h"
#include "evaluation-trace.h"
#include "json.hpp"
#include "log.h"
#include "sampling.h"
#include <gemmini/evaluation_metrics.hpp>
#include <gemmini/host-timing.hpp>
#include <gemmini/semantic.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>

using json = nlohmann::ordered_json;
namespace metrics = ggml::gemmini::evaluation;

static json build_info() {
    return {{"schema", "potal-evaluation-build"}, {"version", 1},
        {"activation_metrics", GGML_GEMMINI_ACT_QUANT_METRICS},
        {"residual_metrics", GGML_GEMMINI_RESIDUAL_METRICS}, {"cycle_sim", CYCLE_SIM},
        {"activation_bits", GGML_GEMMINI_ACTIVATION_BITS}, {"weight_bits", GGML_GEMMINI_WEIGHT_BITS},
        {"dim", GGML_GEMMINI_DIM}, {"backend", EVALUATION_BACKEND},
        {"gemmini_option", EVALUATION_GEMMINI_OPTION}, {"cuda", EVALUATION_CUDA},
        {"gemmini", EVALUATION_GEMMINI}, {"cuda_evaluation_supported", EVALUATION_CUDA != 0},
        {"activation_mode", EVALUATION_ACTIVATION_MODE}, {"block_size", EVALUATION_BLOCK_SIZE},
        {"rmd_enabled", GGML_GEMMINI_ENABLE_RMD}, {"rmd_backend", EVALUATION_RMD_BACKEND},
        {"hp1", EVALUATION_HP1 != 0}, {"matmul_mode", EVALUATION_MATMUL_MODE},
        {"cpu_only", ggml::gemmini::semantic::compiled_cpu_only_build()},
        {"log_cycle", LOG_CYCLE}, {"cycle_detail", CYCLE_DETAIL},
        {"ggml_cpu_cycle_log", EVALUATION_CPU_CYCLE_LOG}};
}

static uint64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
}

static int positive(const std::string & value) {
    size_t used = 0;
    const long long result = std::stoll(value, &used);
    if (used != value.size() || result <= 0 || result > std::numeric_limits<int>::max()) {
        throw std::invalid_argument("expected a positive integer: " + value);
    }
    return int(result);
}

static void fingerprint_logits(uint64_t & hash, const float * logits, size_t count) {
    const auto * bytes = reinterpret_cast<const unsigned char *>(logits);
    for (size_t i = 0; i < count * sizeof(float); ++i) {
        hash = (hash ^ bytes[i]) * UINT64_C(1099511628211);
    }
}

static json batch_record(const llama_batch & batch, int chunk, int part) {
    json result = {{"chunk_id", chunk}, {"part", part}, {"tokens", json::array()},
        {"positions", json::array()}, {"n_seq_id", json::array()},
        {"seq_ids", json::array()}, {"logits", json::array()}};
    for (int i = 0; i < batch.n_tokens; ++i) {
        result["tokens"].push_back(batch.token[i]);
        result["positions"].push_back(batch.pos[i]);
        result["n_seq_id"].push_back(batch.n_seq_id[i]);
        result["seq_ids"].push_back(batch.seq_id[i][0]);
        result["logits"].push_back(int(batch.logits[i]));
    }
    return result;
}

static int run(int argc, char ** argv) {
    common_params params;
    params.n_batch = params.n_ubatch = 256;
    params.cpuparams.n_threads = params.cpuparams_batch.n_threads = 1;
    params.n_gpu_layers = 0;
    params.warmup = false;
    params.sampling.seed = 0;
    params.sampling.temp = 0;
#if GGML_GEMMINI_ACT_QUANT_METRICS || GGML_GEMMINI_RESIDUAL_METRICS
    metrics::Config metric_config;
#endif
    std::string file, output, forced_file, workload = "METRIC_PREFILL_256";
    int max_chunks = 1, first_chunk = 0, smoke_generated_tokens = 0;
    bool seed_set = false, temp_set = false, chunk_index_set = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            std::puts("llama-eval-workload --model MODEL --file WIKITEXT --output-dir DIR\n"
                "  --workload METRIC_PREFILL_256|E2E_GENERATION_256_128 --max-chunks N (0 = all)\n"
                "  --batch-size N --ubatch-size N --threads N --threads-batch N\n"
                "  --seed N --temp F --gpu-layers N --activation-output PATH --residual-output PATH\n"
                "  --chunk-index N --forced-token-ids JSON --run-id ID --build-info\n"
                "  --smoke-generated-tokens 1 (CYCLE_SIM diagnostic; never an E2E campaign)\n"
                "Native non-strided WikiText chunks, no warmup. Defaults: one chunk, batch/ubatch 256,\n"
                "one thread. E2E requires chunk-index0..9; fixed greedy seed1234/temp0, EOS stopping disabled.\n"
                "Forced tokens are FullCPU cost-only; they are never reported as sampled output.");
            return 0;
        }
        if (arg == "--build-info") { std::puts(build_info().dump().c_str()); return 0; }
        if (arg == "--activation-reference-candidate") {
            throw std::invalid_argument("obsolete ACT candidate policy; confirmed row-by-BK32 policy is mandatory");
        }
#if !GGML_GEMMINI_ACT_QUANT_METRICS
        if (arg == "--activation-output") throw std::invalid_argument("activation metrics are compiled out");
#endif
#if !GGML_GEMMINI_RESIDUAL_METRICS
        if (arg == "--residual-output") throw std::invalid_argument("residual metrics are compiled out");
#endif
        if (++i == argc) throw std::invalid_argument("missing value for " + arg);
        const std::string value = argv[i];
        if (arg == "--model" || arg == "-m") params.model.path = value;
        else if (arg == "--file" || arg == "-f") file = value;
        else if (arg == "--output-dir") output = value;
        else if (arg == "--forced-token-ids") forced_file = value;
        else if (arg == "--workload") workload = value;
        else if (arg == "--chunk-index") {
            first_chunk = value == "0" ? 0 : positive(value);
            chunk_index_set = true;
        }
        else if (arg == "--max-chunks") max_chunks = value == "0" ? -1 : positive(value);
        else if (arg == "--smoke-generated-tokens") smoke_generated_tokens = positive(value);
        else if (arg == "--batch-size") params.n_batch = positive(value);
        else if (arg == "--ubatch-size") params.n_ubatch = positive(value);
        else if (arg == "--threads") params.cpuparams.n_threads = positive(value);
        else if (arg == "--threads-batch") params.cpuparams_batch.n_threads = positive(value);
        else if (arg == "--gpu-layers") params.n_gpu_layers = value == "-1" ? -1 : value == "0" ? 0 : positive(value);
#if GGML_GEMMINI_ACT_QUANT_METRICS
        else if (arg == "--activation-output") metric_config.activation_path = value;
#endif
#if GGML_GEMMINI_RESIDUAL_METRICS
        else if (arg == "--residual-output") metric_config.residual_path = value;
#endif
        else if (arg == "--run-id") {
#if GGML_GEMMINI_ACT_QUANT_METRICS || GGML_GEMMINI_RESIDUAL_METRICS
            metric_config.run_id = value;
#endif
        }
        else if (arg == "--seed") {
            size_t used = 0;
            const auto seed = std::stoull(value, &used);
            if (used != value.size() || seed >= LLAMA_DEFAULT_SEED || value[0] == '-') {
                throw std::invalid_argument("seed must be explicit and below LLAMA_DEFAULT_SEED");
            }
            params.sampling.seed = uint32_t(seed);
            seed_set = true;
        } else if (arg == "--temp") {
            size_t used = 0;
            params.sampling.temp = std::stof(value, &used);
            if (used != value.size() || !std::isfinite(params.sampling.temp) || params.sampling.temp < 0) {
                throw std::invalid_argument("temperature must be finite and nonnegative");
            }
            temp_set = true;
        } else throw std::invalid_argument("unknown option " + arg);
    }
    const bool generation = workload == "E2E_GENERATION_256_128";
    const bool forced_cost_only = !forced_file.empty();
    if (smoke_generated_tokens &&
        (!CYCLE_SIM || !generation || forced_cost_only || smoke_generated_tokens != 1))
        throw std::invalid_argument("smoke-generated-tokens requires CYCLE_SIM free generation and exactly 1 token");
    const int generation_target = smoke_generated_tokens ? smoke_generated_tokens : 128;
    if (!generation && workload != "METRIC_PREFILL_256") throw std::invalid_argument("unknown workload");
    if (generation) {
        if (!chunk_index_set || first_chunk > 9 || max_chunks != 1) {
            throw std::invalid_argument("E2E requires --chunk-index0..9 and --max-chunks1");
        }
        if ((seed_set && params.sampling.seed != 1234) || (temp_set && params.sampling.temp != 0)) {
            throw std::invalid_argument("E2E recipe requires seed1234 and temperature0");
        }
        params.sampling = common_evaluation_e2e_sampling();
    } else if (first_chunk != 0 || forced_cost_only) {
        throw std::invalid_argument("chunk selection and forced tokens require E2E workload");
    }
    if (forced_cost_only && (CYCLE_SIM || !LOG_CYCLE || params.n_gpu_layers != 0 ||
                !ggml::gemmini::semantic::compiled_cpu_only_build())) {
        throw std::invalid_argument("forced trajectory requires verified FullCPU cost-only build with LOG_CYCLE=1");
    }
    if (generation && (GGML_GEMMINI_ACT_QUANT_METRICS || GGML_GEMMINI_RESIDUAL_METRICS)) {
        throw std::invalid_argument("E2E requires ACT_QUANT_METRICS=0 and RESIDUAL_METRICS=0");
    }
    if (params.n_gpu_layers != 0 && !EVALUATION_CUDA) throw std::invalid_argument("CUDA is not compiled in");
    if (params.model.path.empty() || file.empty() || output.empty()) {
        throw std::invalid_argument("--model, --file and --output-dir are required");
    }
    if (params.n_batch > 256 || params.n_ubatch > params.n_batch) {
        throw std::invalid_argument("runner requires 1 <= ubatch <= batch <= 256 (one sequence per chunk)");
    }
#if GGML_GEMMINI_ACT_QUANT_METRICS || GGML_GEMMINI_RESIDUAL_METRICS
    if (generation && (!metric_config.activation_path.empty() || !metric_config.residual_path.empty())) {
        throw std::invalid_argument("metric collection requires METRIC_PREFILL_256");
    }
#endif
    std::ifstream input(file, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open dataset " + file);
    std::string text((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    if (!input.eof() && input.fail()) throw std::runtime_error("dataset read failed");
    const bool trimmed_lf = !text.empty() && text.back() == '\n';
    if (trimmed_lf) text.pop_back();
    const auto parent = std::filesystem::path(output).parent_path();
    if (!parent.empty()) std::filesystem::create_directories(parent);
    if (!std::filesystem::create_directory(output)) throw std::runtime_error("output directory already exists");
    std::ofstream manifest(std::filesystem::path(output) / "workload.json");
    std::ofstream batches(std::filesystem::path(output) / "batches.jsonl");
    std::ofstream application(std::filesystem::path(output) / "application.jsonl");
    std::ofstream application_cpu(std::filesystem::path(output) / "application-cpu.jsonl");
    if (!manifest || !batches || !application || !application_cpu) throw std::runtime_error("cannot open output files");
    params.n_ctx = generation ? 384 : 256;
    params.n_parallel = 1;
    common_init();
    llama_backend_init();
    auto initialized = common_init_from_params(params);
    auto * ctx = initialized.context.get();
    auto * model = initialized.model.get();
    if (!ctx || !model) throw std::runtime_error("model initialization failed");
    char file_type_text[64] = {}, description[256] = {};
    json model_file_type = nullptr;
    const int file_type_length = llama_model_meta_val_str(model, "general.file_type", file_type_text, sizeof(file_type_text));
    if (file_type_length > 0 && file_type_length < int(sizeof(file_type_text))) {
        size_t used = 0;
        const int value = std::stoi(file_type_text, &used);
        if (used != size_t(file_type_length)) throw std::runtime_error("invalid model general.file_type metadata");
        model_file_type = value;
    }
    llama_model_desc(model, description, sizeof(description));
    if (params.n_gpu_layers != 0 && model_file_type != LLAMA_FTYPE_MOSTLY_Q4_0 &&
            model_file_type != LLAMA_FTYPE_MOSTLY_Q8_0) {
        throw std::invalid_argument("CUDA reference requires actual Q4_0 or Q8_0 general.file_type metadata");
    }
    const auto * vocab = llama_model_get_vocab(model);
    if (llama_vocab_get_add_eos(vocab)) throw std::runtime_error("native PPL requires add_eos=false");
    std::vector<llama_token> forced_tokens;
    if (forced_cost_only) {
        std::ifstream forced_input(forced_file);
        if (!forced_input) throw std::runtime_error("cannot open forced token file");
        const auto ids = json::parse(forced_input);
        if (!ids.is_array() || ids.size() != 128) throw std::invalid_argument("forced token JSON must be an array of128 IDs");
        for (const auto & entry : ids) {
            if (!entry.is_number_integer()) throw std::invalid_argument("forced token IDs must be integers");
            const int64_t id = entry.get<int64_t>();
            if (id < 0 || id > std::numeric_limits<llama_token>::max()) throw std::invalid_argument("forced token ID out of range");
            forced_tokens.push_back(llama_token(id));
        }
        common_evaluation_validate_forced(forced_tokens, llama_vocab_n_tokens(vocab),
                !CYCLE_SIM && params.n_gpu_layers == 0 && ggml::gemmini::semantic::compiled_cpu_only_build());
    }
    const auto tokens = common_evaluation_tokenize(ctx, text);
    const auto plan = common_evaluation_plan(tokens.size(), 256, params.n_batch, max_chunks, first_chunk);
    const bool add_bos = llama_vocab_get_add_bos(vocab);
    const auto mask = generation ? common_evaluation_mask::generation_last : common_evaluation_mask::perplexity_half;
    const std::string source_role = CYCLE_SIM ? "potal_collection" :
        EVALUATION_CUDA && params.n_gpu_layers != 0 ? "cuda" :
        ggml::gemmini::semantic::compiled_cpu_only_build() ? "full_cpu" : "unsupported";
#if GGML_GEMMINI_ACT_QUANT_METRICS || GGML_GEMMINI_RESIDUAL_METRICS
    metric_config.workload_id = workload;
    auto metric_session = metrics::Session::start(metric_config);
#endif
    std::unique_ptr<common_sampler, decltype(&common_sampler_free)> sampler(
            generation && !forced_cost_only ? common_sampler_init(model, params.sampling) : nullptr, common_sampler_free);
    if (generation && !forced_cost_only && !sampler) throw std::runtime_error("sampler initialization failed");
    auto batch = llama_batch_init(params.n_batch, 0, 1);
    const std::string execution_kind = forced_cost_only ? "FORCED_CPU_COST_ONLY" :
        generation ? "FREE_GENERATION" : "METRIC_PREFILL";
    json result = {{"schema", "potal-native-workload"}, {"version", 1}, {"workload", workload},
        {"model_path", params.model.path}, {"dataset_path", file}, {"tokens", tokens.size()},
        {"model_file_type", model_file_type}, {"model_description", description},
        {"model_parameters", llama_model_n_params(model)},
        {"complete_chunks", plan.total_chunks}, {"selected_chunks", plan.n_chunks},
        {"first_chunk", plan.first_chunk}, {"chunk_index", generation ? json(plan.first_chunk) : json()},
        {"dropped_tail_tokens", plan.tail_tokens}, {"context_tokens", 256},
        {"batch", llama_n_batch(ctx)}, {"ubatch", llama_n_ubatch(ctx)}, {"runtime_context", llama_n_ctx(ctx)},
        {"threads", llama_n_threads(ctx)}, {"threads_batch", llama_n_threads_batch(ctx)},
        {"add_special", true}, {"parse_special", false}, {"trailing_lf_removed", trimmed_lf},
        {"add_bos", add_bos}, {"bos_token", llama_vocab_bos(vocab)}, {"bos_policy", "replace_chunk_first"},
        {"output_mask", generation ? "last_token" : "second_half"}, {"kv_policy", "clear_per_chunk"},
        {"warmup", 0}, {"seed", params.sampling.seed}, {"temperature", params.sampling.temp},
        {"sampler_policy", generation ? "user-confirmed-greedy-v3" : "common_default_chain"},
        {"recipe_id", generation ? json(common_evaluation_e2e_recipe) : json()},
        {"top_k", params.sampling.top_k},
        {"top_p", params.sampling.top_p}, {"min_p", params.sampling.min_p},
        {"typical_p", params.sampling.typ_p}, {"penalty_repeat", params.sampling.penalty_repeat},
        {"penalty_last_n", params.sampling.penalty_last_n}, {"grammar", params.sampling.grammar},
        {"eos_stopping", false}, {"eos_stop", false}, {"eos_logit_suppression", false},
        {"ignore_eos", params.sampling.ignore_eos}, {"vocab_size", llama_vocab_n_tokens(vocab)},
        {"execution_kind", execution_kind}, {"cost_only", forced_cost_only},
        {"diagnostic_smoke", smoke_generated_tokens != 0},
        {"requested_generated_tokens", generation ? generation_target : 0},
        {"trajectory_source", forced_cost_only ? json("POTAL") : json()},
        {"sampling_executed", generation && !forced_cost_only}, {"forced_token_ids_path", forced_file},
        {"gpu_layers_requested", params.n_gpu_layers}, {"placement_proof", "model_load_log"},
        {"placement_verified", false}, {"source_role", source_role},
        {"target_trace_collection_enabled", generation && CYCLE_SIM != 0},
        {"target_trace_collection_reason", !generation ? "metric_statistics_only" :
            CYCLE_SIM ? "generation_target_collection" : "cycle_sim_disabled"},
        {"build", build_info()}, {"chunks", json::array()}, {"complete", false}};
    manifest << result.dump(2) << '\n';
    manifest.flush();
    bool success = true;
    try {
        for (int selected_chunk = 0; selected_chunk < plan.n_chunks; ++selected_chunk) {
            const int chunk = plan.first_chunk + selected_chunk;
            common_evaluation_begin(ctx);
            if (sampler) common_sampler_reset(sampler.get());
#if GGML_GEMMINI_ACT_QUANT_METRICS || GGML_GEMMINI_RESIDUAL_METRICS
            if (metric_session) metric_session->chunk(chunk);
#endif
            const auto trace_dir = std::filesystem::path(output) / ("chunk-" + std::to_string(chunk));
            std::filesystem::create_directories(trace_dir);
#ifdef _WIN32
            _putenv_s("GEMMINI_LOG_DIR", trace_dir.string().c_str());
#else
            if (setenv("GEMMINI_LOG_DIR", trace_dir.string().c_str(), 1) != 0) throw std::runtime_error("setenv failed");
#endif
            evaluation_trace trace(params, params.model.path, 256,
                                   generation ? generation_target : 0, forced_cost_only);
            std::vector<llama_token> prompt(tokens.begin() + chunk * 256, tokens.begin() + (chunk + 1) * 256);
            if (add_bos) prompt[0] = llama_vocab_bos(vocab);
            if (sampler) for (const auto token : prompt) common_sampler_accept(sampler.get(), token, false);
            trace.phase("prefill", prompt);
            std::vector<json> records;
            std::vector<std::pair<gemmini_cpu_sample, gemmini_cpu_sample>> prefill_times;
            prefill_times.reserve(plan.n_batches);
            std::vector<uint64_t> endpoints;
            std::vector<llama_token> generated;
            std::vector<std::pair<gemmini_cpu_sample, gemmini_cpu_sample>> sampling_times;
            endpoints.reserve(generation_target);
            generated.reserve(generation_target);
            sampling_times.reserve(generation_target);
            int decode_calls = 0;
            bool eos = false;
            uint64_t logits_hash = UINT64_C(14695981039346656037);
            size_t logits_values = 0;
            common_log_pause(common_log_main());
            const uint64_t t0 = now_ns();
            trace.request_start();
            for (int part = 0; part < plan.n_batches; ++part) {
                const auto preparation_start = trace.pipeline_collection()
                    ? gemmini_cpu_timing_read() : gemmini_cpu_sample{};
                const int outputs = common_evaluation_batch(batch, tokens, plan, selected_chunk, part,
                        add_bos, llama_vocab_bos(vocab), mask);
                if (trace.pipeline_collection()) {
                    prefill_times.emplace_back(preparation_start, gemmini_cpu_timing_read());
                }
                records.push_back(batch_record(batch, chunk, part));
                trace.prefill_batch_ready(part);
                if (trace.decode(ctx, batch)) throw std::runtime_error("prefill decode failed");
                if (!generation && outputs != 0) {
                    const size_t count = size_t(outputs) * llama_vocab_n_tokens(vocab);
                    fingerprint_logits(logits_hash, llama_get_logits(ctx), count);
                    logits_values += count;
                }
            }
            for (int sample = 0; generation && sample < generation_target; ++sample) {
                llama_token token;
                if (forced_cost_only) {
                    token = forced_tokens[sample];
                    trace.forced_complete(sample, token);
                } else {
                    const auto sample_start = gemmini_cpu_timing_read();
                    token = common_sampler_sample(sampler.get(), ctx, -1);
                    common_sampler_accept(sampler.get(), token, true);
                    const auto sample_end = gemmini_cpu_timing_read();
                    endpoints.push_back(now_ns());
                    trace.sample_complete(sample, token);
                    sampling_times.emplace_back(sample_start, sample_end);
                }
                generated.push_back(token);
                eos = eos || llama_vocab_is_eog(vocab, token);
                if (sample + 1 == generation_target) break;
                common_batch_clear(batch);
                common_batch_add(batch, token, 256 + sample, {0}, true);
                trace.phase("decode", {token}, sample);
                ++decode_calls;
                if (trace.decode(ctx, batch)) throw std::runtime_error("generation decode failed");
            }
            const bool complete = !generation || (smoke_generated_tokens
                ? generated.size() == size_t(generation_target) && decode_calls + 1 == generation_target
                : common_evaluation_generation_complete(generated.size(), decode_calls));
            common_log_resume(common_log_main());
            if (generation) {
                logits_values = llama_vocab_n_tokens(vocab);
                fingerprint_logits(logits_hash, llama_get_logits_ith(ctx, -1), logits_values);
            }
            char fingerprint[17];
            std::snprintf(fingerprint, sizeof(fingerprint), "%016llx", static_cast<unsigned long long>(logits_hash));
            success = success && complete;
            trace.finish(complete);
            for (const auto & record : records) batches << record.dump() << '\n';
            result["chunks"].push_back({{"chunk_id", chunk}, {"token_offset", chunk * 256},
                {"input_tokens", prompt}, {"trace_dir", trace_dir.string()}, {"complete", complete},
                {"logits_fingerprint", fingerprint}, {"logits_fingerprint_algorithm", "fnv1a64_native_float_bytes_diagnostic"},
                {"logits_values", logits_values}, {"logits_scope", forced_cost_only ? "last_decode_output" :
                    generation ? "last_sample_input" : "all_requested_prefill"}});
            application << json({{"schema", "potal-application-endpoints"}, {"version", 1},
                {"chunk_id", chunk}, {"workload", workload}, {"t0_ns", t0}, {"sample_accept_ns", endpoints},
                {"generated_tokens", generated}, {"samples", endpoints.size()}, {"actual_samples", endpoints.size()},
                {"actual_sampler_calls", endpoints.size()}, {"forced_steps", forced_cost_only ? generated.size() : 0},
                {"decode_calls", decode_calls}, {"execution_kind", execution_kind}, {"cost_only", forced_cost_only},
                {"diagnostic_smoke", smoke_generated_tokens != 0},
                {"requested_generated_tokens", generation ? generation_target : 0},
                {"trajectory_source", forced_cost_only ? json("POTAL") : json()},
                {"recipe_id", generation ? json(common_evaluation_e2e_recipe) : json()},
                {"complete", complete}, {"eos", eos}, {"eos_seen", eos}, {"eos_stopping", false},
                {"warmup", 0}, {"timing_source", "steady_clock"},
                {"timing_unit", "ns"}, {"excludes_terminal_io", true},
                {"source_role", source_role}}).dump() << '\n';
            for (size_t part = 0; part < prefill_times.size(); ++part) {
                const auto &timing = prefill_times[part];
                json service = json::parse("{\"schema\":\"potal-application-cpu\"" +
                    ggml::gemmini::cycle::serialize_cpu_timing_contract(timing.first, timing.second) + "}");
                service.update({{"version", 2}, {"chunk_id", chunk}, {"stage", "prefill_batch_prepare"},
                    {"batch_index", part}, {"dispatch_id", part},
                    {"sample_index", nullptr}, {"token_id", nullptr},
                    {"source_role", source_role}, {"phase", "prefill"}, {"decode_index", nullptr}});
                application_cpu << service.dump() << '\n';
            }
            for (size_t sample = 0; sample < sampling_times.size(); ++sample) {
                const auto & timing = sampling_times[sample];
                json service = json::parse("{\"schema\":\"potal-application-cpu\"" +
                    ggml::gemmini::cycle::serialize_cpu_timing_contract(timing.first, timing.second) + "}");
                service.update({{"version", trace.pipeline_collection() ? 2 : 1},
                    {"chunk_id", chunk}, {"stage", "sample_accept"},
                    {"sample_index", sample}, {"token_id", generated[sample]}, {"source_role", source_role},
                    {"phase", sample == 0 ? "prefill" : "decode"},
                    {"decode_index", sample == 0 ? json() : json(sample - 1)}});
                application_cpu << service.dump() << '\n';
            }
#if GGML_GEMMINI_ACT_QUANT_METRICS || GGML_GEMMINI_RESIDUAL_METRICS
            if (metric_session) metric_session->ensure_healthy();
#endif
        }
#if GGML_GEMMINI_ACT_QUANT_METRICS || GGML_GEMMINI_RESIDUAL_METRICS
        if (metric_session) metric_session->finish(success);
#endif
    } catch (...) {
        common_log_resume(common_log_main());
        llama_batch_free(batch);
#if GGML_GEMMINI_ACT_QUANT_METRICS || GGML_GEMMINI_RESIDUAL_METRICS
        if (metric_session) metric_session->finish(false);
#endif
        throw;
    }
    llama_batch_free(batch);
    result["complete"] = success;
    manifest.close();
    manifest.open(std::filesystem::path(output) / "workload.json", std::ios::trunc);
    manifest << result.dump(2) << '\n';
    manifest.close(); batches.close(); application.close(); application_cpu.close();
    if (!manifest || !batches || !application || !application_cpu) throw std::runtime_error("output write failed");
    if (!common_fpga_execution_check()) throw std::runtime_error("backend execution failed");
    return success ? 0 : 2;
}

int main(int argc, char ** argv) {
    try { return run(argc, argv); }
    catch (const std::exception & error) { std::fprintf(stderr, "llama-eval-workload: %s\n", error.what()); return 1; }
}
