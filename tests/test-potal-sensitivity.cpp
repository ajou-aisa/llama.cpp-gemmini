#include "../ggml/src/ggml-gemmini/potal-sensitivity.hpp"
#include "ggml-gemmini.h"
#include "../ggml/src/ggml-gemmini/quants/act/exsia/local.hpp"
#include "../common/json.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cfenv>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unistd.h>

namespace sensitivity = ggml::gemmini::sensitivity;
static void check(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

int main(int argc, char ** argv) {
    try {
        // Given exact half-integers, including negative ties, under a non-default FP mode.
        const int previous_rounding = std::fegetround();
        std::fesetround(FE_UPWARD);
        const std::vector<float> ties{.5f,1.5f,2.5f,-.5f,-1.5f,-2.5f,1.49f,-1.49f,4,-4};
        const std::vector<int32_t> expected{1,2,3,-1,-2,-3,1,-1,4,-4};
        ggml::gemmini::quants::act::exsia::WideQuantizer quantizer;
        for (int16_t theta : {-8,0,4}) {
            auto values = ties;
            for (auto & value : values) value = std::ldexp(value,theta);
            // When the real activation quantizer runs, ties move away regardless of FP mode.
            const auto actual = quantizer.quantize_block(values,theta);
            check(actual == expected,"activation quantization must use ties-away independently of FP mode");
        }
        std::fesetround(previous_rounding);
        if (argc == 2) {
            setenv("GGML_GEMMINI_ANCHOR_DEFAULT",argv[1],1);
            ggml_tensor w{}, x{}, out{};
            out.op = GGML_OP_MUL_MAT; out.src[0] = &w; out.src[1] = &x;
            std::strcpy(w.name,"blk.1.attn_q.weight");
            sensitivity::anchor(&out);
            return 2; // Invalid anchors must throw rather than silently using the build default.
        }
        char policy[] = "/tmp/potal-policy-XXXXXX", trace[] = "/tmp/potal-trace-XXXXXX";
        const int policy_fd = mkstemp(policy), trace_fd = mkstemp(trace);
        check(policy_fd >= 0 && trace_fd >= 0,"temporary files"); close(policy_fd); close(trace_fd);
        struct Cleanup { const char * a; const char * b; ~Cleanup() { std::remove(a); std::remove(b); } } cleanup{policy,trace};
        { std::ofstream file(policy); file << "blk.1.attn_q.weight\t3\nqk:kq-1\t3\n"; }
        setenv("GGML_GEMMINI_ANCHOR_DEFAULT","2",1);
        setenv("GGML_GEMMINI_ANCHOR_POLICY",policy,1);
        setenv("GGML_GEMMINI_SENSITIVITY_LOG",trace,1);
        // Given distinct defaults and overrides, plus a two-element exact dot product.
        ggml_tensor w{}, x{}, out{};
        float weights[]{1,2}, input[]{3,4}, output[]{10};
        w.type = x.type = out.type = GGML_TYPE_F32;
        w.data = weights; x.data = input; out.data = output;
        for (auto * t : {&w,&x,&out}) {
            for (int d = 0; d < 4; ++d) { t->ne[d] = 1; t->nb[d] = sizeof(float); }
        }
        w.ne[0] = x.ne[0] = 2; w.nb[1] = x.nb[1] = 2*sizeof(float);
        out.op = GGML_OP_MUL_MAT; out.src[0] = &w; out.src[1] = &x;
        std::strcpy(w.name,"blk.1.attn_q.weight"); std::strcpy(out.name,"kq-1");
        // When the policy is resolved, the exact keys override only their own nodes.
        check(sensitivity::anchor(&out) == 3,"linear override ignored");
        std::strcpy(w.name,"blk.2.attn_q.weight");
        check(sensitivity::anchor(&out) == 2,"default replaced by unrelated override");
        ggml_gemmini_set_attention_role(&out,GGML_GEMMINI_ATTENTION_QK);
        check(sensitivity::anchor(&out) == 3,"QK override ignored");
        // When shadow replay writes a different answer, the live output must be restored.
        sensitivity::observe(&out,sensitivity::snapshot(),[&] {
            check(sensitivity::shadow() && sensitivity::anchor(&out) == 2,"shadow override ignored");
            output[0] = 9;
        });
        check(output[0] == 10 && !sensitivity::shadow(),"shadow contaminated live output");
        std::ifstream file(trace); nlohmann::json row; file >> row;
        check(row["reference_energy"] == 121 && row["3"]["sse"] == 1 && row["2"]["sse"] == 4,
              "double reference or paired errors incorrect");
        check(sensitivity::finish(),"used keys reported unused");
        std::puts("PASS: policy precedence, shadow isolation, exact local error reference");
        return 0;
    } catch (const std::exception & error) {
        if (argc == 2) { std::printf("PASS rejected invalid anchor: %s\n",error.what()); return 0; }
        std::fprintf(stderr,"FAIL: %s\n",error.what()); return 1;
    }
}
