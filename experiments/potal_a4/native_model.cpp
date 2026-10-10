#include "llama.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static void require(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

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
