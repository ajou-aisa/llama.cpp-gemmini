#include "llama.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

struct capture_state {
    std::filesystem::path directory;
    std::ofstream channels;
    size_t ordinal = 0;
};

static bool selected(const char * name) {
    for (const char * prefix : {"inp_embd", "inpL", "attn_norm-", "ffn_inp-", "ffn_norm-",
                                "ffn_out-", "l_out-", "norm-", "norm", "result_norm"}) {
        if (std::strcmp(name, prefix) == 0 ||
            (prefix[std::strlen(prefix) - 1] == '-' && std::strncmp(name, prefix, std::strlen(prefix)) == 0)) {
            return true;
        }
    }
    return false;
}

static bool record(ggml_tensor * tensor, bool ask, void * userdata) {
    if (ask) {
        return selected(tensor->name);
    }
    auto & state = *static_cast<capture_state *>(userdata);
    GGML_ASSERT(tensor->type == GGML_TYPE_F32 && ggml_is_contiguous(tensor));
    GGML_ASSERT(tensor->ne[2] == 1 && tensor->ne[3] == 1);
    const size_t width = tensor->ne[0];
    const size_t rows = tensor->ne[1];
    std::vector<float> values(width * rows);
    ggml_backend_tensor_get(tensor, values.data(), 0, values.size() * sizeof(float));
    for (size_t channel = 0; channel < width; ++channel) {
        double sum = 0, squares = 0, max_abs = 0;
        for (size_t row = 0; row < rows; ++row) {
            const double value = values[row * width + channel];
            GGML_ASSERT(std::isfinite(value));
            sum += value;
            squares += value * value;
            max_abs = std::max(max_abs, std::abs(value));
        }
        state.channels << state.ordinal << ',' << tensor->name << ',' << ggml_op_name(tensor->op) << ','
                       << rows << ',' << width << ',' << channel << ',' << sum / rows << ','
                       << squares / rows << ',' << max_abs << '\n';
    }
    if (std::strcmp(tensor->name, "result_norm") == 0) {
        std::ofstream out(state.directory / "head-input.f32", std::ios::binary);
        out.exceptions(std::ios::badbit | std::ios::failbit);
        out.write(reinterpret_cast<const char *>(values.data()), values.size() * sizeof(float));
    }
    ++state.ordinal;
    return true;
}

int main(int argc, char ** argv) {
    if (argc != 4) {
        std::cerr << "usage: capture-layer-channels MODEL.gguf TEXT OUTPUT_DIRECTORY\n";
        return 2;
    }
    constexpr int count = 512;
    const std::filesystem::path directory(argv[3]);
    std::filesystem::create_directories(directory);
    capture_state state{directory, std::ofstream(directory / "channels.csv")};
    state.channels.exceptions(std::ios::badbit | std::ios::failbit);
    state.channels << std::setprecision(17) << "ordinal,node,op,rows,width,channel,mean,mean_square,max_abs\n";
    std::ifstream source(argv[2], std::ios::binary);
    GGML_ASSERT(source.is_open());
    const std::string text((std::istreambuf_iterator<char>(source)), std::istreambuf_iterator<char>());
    ggml_backend_load_all();
    llama_backend_init();
    auto model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], model_params);
    GGML_ASSERT(model != nullptr);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int required = -llama_tokenize(vocab, text.data(), text.size(), nullptr, 0, true, true);
    GGML_ASSERT(required >= count);
    std::vector<llama_token> tokens(required);
    GGML_ASSERT(llama_tokenize(vocab, text.data(), text.size(), tokens.data(), tokens.size(), true, true) == required);
    {
        std::ofstream out(directory / "tokens.i32", std::ios::binary);
        out.exceptions(std::ios::badbit | std::ios::failbit);
        out.write(reinterpret_cast<const char *>(tokens.data()), tokens.size() * sizeof(llama_token));
    }
    auto params = llama_context_default_params();
    params.n_ctx = count;
    params.n_batch = count;
    params.n_ubatch = count;
    params.n_threads = 1;
    params.n_threads_batch = 1;
    params.offload_kqv = false;
    params.flash_attn = false;
    params.type_k = GGML_TYPE_F16;
    params.type_v = GGML_TYPE_F16;
    params.cb_eval = record;
    params.cb_eval_user_data = &state;
    llama_context * context = llama_init_from_model(model, params);
    GGML_ASSERT(context != nullptr);
    auto batch = llama_batch_init(count, 0, 1);
    batch.n_tokens = count;
    for (int i = 0; i < count; ++i) {
        batch.token[i] = tokens[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = true;
    }
    const int status = llama_decode(context, batch);
    GGML_ASSERT(status == 0);
    state.channels.flush();
    std::cout << "captured " << state.ordinal << " nodes; " << count << " tokens; CPU, 1 thread\n";
    llama_batch_free(batch);
    llama_free(context);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
