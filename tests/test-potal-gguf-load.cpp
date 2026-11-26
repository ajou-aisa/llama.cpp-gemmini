// Loads a real Q8_HP1 GGUF through the production llama.cpp model loader with
// the POTAL backend registered, and checks the buffer placement the loader
// chose. No graph is built or computed: nothing reaches the PoTal hardware.
//
//   usage: potal-gguf-load-test <model.gguf> [--mmap=1|0|both]
//   exit:  0 pass, 1 fail, 77 skip (model file absent)
//
// For a model whose output projection shares token_embd.weight (GPT-2):
//   input  token_embd.weight  GET_ROWS on CPU, in a CPU-accessible buffer
//   output token_embd.weight  MUL_MAT, in the POTAL buffer type
// and every weight buffer is marked GGML_BACKEND_BUFFER_USAGE_WEIGHTS (the
// POTAL weight cache only caches those).

#include "../src/llama-model.h"

#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml.h"
#include "llama.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

void check(const std::string & name, bool ok) {
    std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", name.c_str());
    if (!ok) {
        g_failures++;
    }
}

// Loader lines that show the buffer selection; everything else is dropped.
std::vector<std::string> g_log;

void log_cb(ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) user_data;
    if (std::strstr(text, "buffer size") || std::strstr(text, "compatible buffer type") ||
        std::strstr(text, "[POTAL_BACKEND] registered") || std::strstr(text, "error")) {
        g_log.emplace_back(text);
    }
}

const char * buft_name_of(const ggml_tensor * t) {
    return t && t->buffer ? ggml_backend_buft_name(ggml_backend_buffer_get_type(t->buffer)) : "(none)";
}

bool supports(ggml_backend_dev_t dev, const ggml_tensor * op) {
    return dev != nullptr && ggml_backend_dev_supports_op(dev, op);
}

void run(const char * path, bool use_mmap, ggml_backend_dev_t potal, ggml_backend_dev_t cpu) {
    const std::string tag = use_mmap ? "mmap=1 " : "mmap=0 ";
    g_log.clear();

    llama_model_params mp = llama_model_default_params();
    mp.use_mmap           = use_mmap;
    mp.n_gpu_layers       = 0;
    llama_model * model   = llama_model_load_from_file(path, mp);
    for (const auto & line : g_log) {
        std::printf("  log: %s", line.c_str());
    }
    check(tag + "llama_model_load_from_file", model != nullptr);
    if (model == nullptr) {
        return;
    }

    const ggml_tensor * embd = model->tok_embd;
    const ggml_tensor * out  = model->output;
    check(tag + "token_embd.weight and output loaded", embd != nullptr && out != nullptr);
    if (embd == nullptr || out == nullptr) {
        llama_model_free(model);
        return;
    }
    std::printf("  input  %s type=%s buft=%s\n", embd->name, ggml_type_name(embd->type), buft_name_of(embd));
    std::printf("  output %s type=%s buft=%s\n", out->name, ggml_type_name(out->type), buft_name_of(out));

    ggml_backend_buffer_type_t potal_buft = ggml_backend_dev_buffer_type(potal);
    check(tag + "weights are Q8_HP1", embd->type == GGML_TYPE_Q8_HP1 && out->type == GGML_TYPE_Q8_HP1);
    check(tag + "output shares token_embd.weight", std::strcmp(out->name, "token_embd.weight") == 0);
    check(tag + "output in the POTAL buffer type", ggml_backend_buffer_get_type(out->buffer) == potal_buft);
    check(tag + "input not in the POTAL buffer type", ggml_backend_buffer_get_type(embd->buffer) != potal_buft);
    check(tag + "both buffers host-accessible",
          ggml_backend_buffer_is_host(embd->buffer) && ggml_backend_buffer_is_host(out->buffer));
    check(tag + "both buffers usage WEIGHTS",
          ggml_backend_buffer_get_usage(embd->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
              ggml_backend_buffer_get_usage(out->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    if (use_mmap) {
        // Both copies alias the one file mapping: no weight bytes were copied.
        check(tag + "input and output alias the same mapped bytes", embd->data == out->data);
    } else {
        check(tag + "input and output are separate loaded copies with equal bytes",
              embd->data != out->data && ggml_nbytes(embd) == ggml_nbytes(out) &&
                  std::memcmp(embd->data, out->data, ggml_nbytes(out)) == 0);
    }

    // The ops the graph will run on these tensors, as the loader probes them.
    ggml_init_params ip = { ggml_tensor_overhead() * 8, nullptr, true };
    ggml_context_ptr ctx(ggml_init(ip));
    ggml_tensor *    ids      = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 8);
    ggml_tensor *    get_rows = ggml_get_rows(ctx.get(), const_cast<ggml_tensor *>(embd), ids);
    ggml_tensor *    act      = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, out->ne[0], 8);
    ggml_tensor *    mul_mat  = ggml_mul_mat(ctx.get(), const_cast<ggml_tensor *>(out), act);
    check(tag + "CPU supports Q8_HP1 GET_ROWS (input)", supports(cpu, get_rows));
    check(tag + "POTAL refuses GET_ROWS", !supports(potal, get_rows));
    check(tag + "POTAL supports F32 x Q8_HP1 MUL_MAT (output)", supports(potal, mul_mat));
    check(tag + "CPU refuses Q8_HP1 MUL_MAT", !supports(cpu, mul_mat));

    llama_model_free(model);
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <model.gguf> [--mmap=1|0|both]\n", argv[0]);
        return 1;
    }
    const char * path = argv[1];
    std::string  mmap = "both";
    for (int i = 2; i < argc; ++i) {
        if (std::strncmp(argv[i], "--mmap=", 7) == 0) {
            mmap = argv[i] + 7;
        }
    }
    if (mmap != "1" && mmap != "0" && mmap != "both") {
        std::fprintf(stderr, "bad --mmap=%s\n", mmap.c_str());
        return 1;
    }
    if (FILE * f = std::fopen(path, "rb")) {
        std::fclose(f);
    } else {
        std::printf("POTAL_GGUF_LOAD_TEST SKIP: model not found: %s\n", path);
        return 77;
    }

    llama_log_set(log_cb, nullptr);
    llama_backend_init();

    ggml_backend_dev_t potal = ggml_backend_dev_by_name("POTAL");
    ggml_backend_dev_t cpu   = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    check("POTAL device registered", potal != nullptr);
    check("CPU device registered", cpu != nullptr);
    if (potal == nullptr || cpu == nullptr) {
        std::printf("POTAL_GGUF_LOAD_TEST FAIL %d\n", g_failures);
        return 1;
    }
    check("POTAL device type ACCEL", ggml_backend_dev_type(potal) == GGML_BACKEND_DEVICE_TYPE_ACCEL);

    if (mmap != "0") {
        run(path, true, potal, cpu);
    }
    if (mmap != "1") {
        run(path, false, potal, cpu);
    }

    llama_backend_free();
    std::printf("POTAL_GGUF_LOAD_TEST %s failures=%d\n", g_failures == 0 ? "PASS" : "FAIL", g_failures);
    return g_failures == 0 ? 0 : 1;
}
