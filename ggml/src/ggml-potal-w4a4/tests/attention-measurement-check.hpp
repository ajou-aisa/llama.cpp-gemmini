// Exercise the current paired attention API; the historical attention test
// still uses the old planned-runner API. Included by the pipeline test.
#include "matmul/attention.hpp"
#include "ggml-gemmini.h"

static bool attention_measurement_check() {
    namespace attention = ggml::potal::attention;
    using namespace ggml::potal;
    bool ok = true;
    ggml_context * ctx = ggml_init({4u << 20, nullptr, false});
    if (ctx == nullptr) return false;
    for (auto role : {attention::role::qk, attention::role::pv}) {
        for (int rows : {1, 33}) {
            ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 40);
            ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, rows);
            ggml_tensor * out = ggml_mul_mat(ctx, w, a);
            ggml_gemmini_set_attention_role(out, role == attention::role::qk
                ? GGML_GEMMINI_ATTENTION_QK : GGML_GEMMINI_ATTENTION_PV);
            for (int i = 0; i < 64 * 40; ++i)
                static_cast<float *>(w->data)[i] = float(i % 17 - 8) / 16;
            for (int i = 0; i < 64 * rows; ++i)
                static_cast<float *>(a->data)[i] = role == attention::role::qk
                    ? std::ldexp(float(i % 19 - 9), i % 5 - 3) : float(i % 11 + 1) / 512;
            potal_test::h2_emulator emulator;
            uint32_t generation = 0;
            paired_runner runner = [&](const potal_paired_request & request, potal_paired_report & report) {
                auto q = request;
                q.generation = &generation;
                const auto sink = emulator.sink();
                return potal_paired_run_session(q, &sink, report);
            };
            const auto serial = attention::execute(out, runner);
            std::vector<float> expected(static_cast<float *>(out->data),
                                        static_cast<float *>(out->data) + 40 * rows);
            std::fill_n(static_cast<float *>(out->data), 40 * rows, 12345.0f);
            potal_test::thread_owner owner({}, runner);
            const auto strict = attention::execute_strict(out, owner, {{-1,-1,-1,-1,-1}}, false);
            owner.join();
            const bool same = serial.status == POTAL_OK && strict.status == POTAL_OK &&
                std::memcmp(out->data, expected.data(), expected.size() * sizeof(float)) == 0 &&
                std::any_of(expected.begin(), expected.end(), [](float x) { return x != 0; });
            std::printf("%s attention measurement %s rows=%d serial=%d strict=%d\n",
                        same ? "PASS" : "FAIL", role == attention::role::qk ? "QK" : "PV",
                        rows, int(serial.status), int(strict.status));
            ok = same && ok;
        }
    }
    ggml_free(ctx);
    return ok;
}
