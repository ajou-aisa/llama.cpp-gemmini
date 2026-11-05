#include "potal-sensitivity.hpp"
#include "ggml-gemmini.h"
#include "ggml-gemmini-matmul-config.hpp"
#include "../../../common/json.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>
#include <vector>
#if defined(GGML_GEMMINI_CUDA_CPU_EXACT)
#include "../ggml-cuda/cpu-exact/ggml-cuda-hp1-scu.h"
#include "../ggml-cuda/cpu-exact/ggml-cuda-potal-attention.h"
#endif

namespace ggml::gemmini::sensitivity {
namespace {
thread_local int override_anchor = 0;
struct Settings {
    int fallback = GGML_GEMMINI_EXSIA_STRIPE_ANCHOR;
    std::map<std::string, int> policy;
    std::set<std::string> used;
    std::map<std::string, size_t> calls;
    std::ofstream log;
    std::mutex mutex;
    Settings() {
        auto parse = [](const std::string & value) {
            if (value == "2") return 2;
            if (value == "3") return 3;
            throw std::runtime_error("PoTal anchor must be exactly 2 or 3");
        };
        if (const char * value = std::getenv("GGML_GEMMINI_ANCHOR_DEFAULT")) fallback = parse(value);
        if (const char * path = std::getenv("GGML_GEMMINI_ANCHOR_POLICY")) {
            std::ifstream input(path);
            if (!input) throw std::runtime_error("Cannot open PoTal anchor policy");
            std::string line;
            while (std::getline(input, line)) {
                if (!line.empty() && line.back() == '\r') line.pop_back();
                if (line.empty() || line[0] == '#') continue;
                const auto tab = line.find('\t');
                if (tab == std::string::npos || tab == 0 ||
                    !policy.emplace(line.substr(0, tab), parse(line.substr(tab+1))).second)
                    throw std::runtime_error("Malformed or duplicate PoTal anchor policy key");
            }
        }
        if (const char * path = std::getenv("GGML_GEMMINI_SENSITIVITY_LOG")) {
            log.open(path, std::ios::out | std::ios::trunc);
            if (!log) throw std::runtime_error("Cannot open PoTal sensitivity log");
        }
    }
};
Settings & settings() { static Settings value; return value; }

bool eligible(const ggml_tensor * node) {
    return node && node->op == GGML_OP_MUL_MAT && node->src[0] && node->src[1] &&
        (ggml_gemmini_get_attention_role(node) != GGML_GEMMINI_ATTENTION_NONE ||
         node->src[0]->type == GGML_TYPE_MXINT4 || node->src[0]->type == GGML_TYPE_MXINT8);
}

float value(const ggml_tensor & tensor, size_t k, size_t row, size_t head, size_t batch) {
    const auto * p = static_cast<const char *>(tensor.data)+k*tensor.nb[0]+row*tensor.nb[1]+
                     head*tensor.nb[2]+batch*tensor.nb[3];
    if (tensor.type == GGML_TYPE_F32) { float v; std::memcpy(&v,p,sizeof(v)); return v; }
    if (tensor.type == GGML_TYPE_F16) { ggml_fp16_t v; std::memcpy(&v,p,sizeof(v)); return ggml_fp16_to_fp32(v); }
    throw std::runtime_error("Unsupported sensitivity reference operand");
}

struct Samples {
    std::vector<size_t> indices;
    std::vector<double> reference;
    double energy = 0;
};
Samples reference(const ggml_tensor & node) {
    const auto & w = *node.src[0];
    const auto & x = *node.src[1];
    const size_t count = std::min<size_t>(128, ggml_nelements(&node));
    Samples samples;
    std::vector<float> row(w.ne[0]);
    for (size_t s = 0; s < count; ++s) {
        // Stratification covers rows, columns and attention heads without sampling the diagonal only.
        const size_t begin = s*size_t(ggml_nelements(&node))/count;
        const size_t end = (s+1)*size_t(ggml_nelements(&node))/count;
        uint64_t random = s+0x9e3779b97f4a7c15ULL;
        random = (random^(random>>30))*0xbf58476d1ce4e5b9ULL;
        random = (random^(random>>27))*0x94d049bb133111ebULL;
        const size_t index = begin+(random^(random>>31))%(end-begin);
        size_t rest = index;
        const size_t col = rest%node.ne[0]; rest /= node.ne[0];
        const size_t r = rest%node.ne[1]; rest /= node.ne[1];
        const size_t h = rest%node.ne[2], b = rest/node.ne[2];
        const size_t wh = h/(x.ne[2]/w.ne[2]), wb = b/(x.ne[3]/w.ne[3]);
        if (ggml_is_quantized(w.type)) {
            const auto convert = ggml_get_type_traits(w.type)->to_float;
            if (!convert) throw std::runtime_error("Missing sensitivity weight dequantizer");
            convert(static_cast<const char *>(w.data)+col*w.nb[1]+wh*w.nb[2]+wb*w.nb[3], row.data(), w.ne[0]);
        } else for (size_t k = 0; k < row.size(); ++k) row[k] = value(w,k,col,wh,wb);
        double dot = 0;
        for (size_t k = 0; k < row.size(); ++k) dot += double(row[k])*value(x,k,r,h,b);
        samples.indices.push_back(index);
        samples.reference.push_back(dot);
        samples.energy += dot*dot;
    }
    return samples;
}

nlohmann::json measurement(const std::vector<float> & output, const Samples & samples,
                           Snapshot first, Snapshot last) {
    double sse = 0, max_abs = 0, dot = 0, norm = 0;
    for (size_t s = 0; s < samples.indices.size(); ++s) {
        const double y = output[samples.indices[s]], ref = samples.reference[s], delta = y-ref;
        if (!std::isfinite(y) || !std::isfinite(ref)) throw std::runtime_error("Nonfinite sensitivity sample");
        sse += delta*delta; max_abs = std::max(max_abs,std::abs(delta)); dot += y*ref; norm += y*y;
    }
    return {{"sse",sse},{"max_abs_error",max_abs},{"output_energy",norm},{"reference_dot",dot},
            {"main_fragments",last.main-first.main},{"residual_fragments",last.residual-first.residual},
            {"saturations",last.saturations-first.saturations},{"seconds",last.seconds-first.seconds}};
}
} // namespace

std::string key(const ggml_tensor * node) {
    const auto role = ggml_gemmini_get_attention_role(node);
    if (role == GGML_GEMMINI_ATTENTION_QK) return std::string("qk:")+node->name;
    if (role == GGML_GEMMINI_ATTENTION_PV) return std::string("pv:")+node->name;
    return node->src[0]->name;
}
int anchor(const ggml_tensor * node) {
    if (override_anchor) return override_anchor;
    auto & config = settings();
    std::lock_guard<std::mutex> lock(config.mutex);
    const auto found = config.policy.find(key(node));
    if (found == config.policy.end()) return config.fallback;
    config.used.insert(found->first);
    return found->second;
}
bool profiling() { return settings().log.is_open(); }
bool shadow() { return override_anchor != 0; }
int reported_anchor() { return settings().policy.empty() ? settings().fallback : 0; }

Snapshot snapshot() {
    Snapshot result;
#if defined(GGML_GEMMINI_CUDA_CPU_EXACT)
    const auto dense = ggml_cuda_hp1_scu_stats();
    const auto attention = ggml_cuda_potal_attention_get_stats();
    result.main = dense.main_fragments+attention.main_fragments;
    result.residual = dense.residual_fragments+attention.residual_fragments;
    result.saturations = dense.main_shift_saturations+dense.main_acc_saturations+
                         dense.residual_shift_saturations+dense.residual_acc_saturations;
#endif
    result.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
    return result;
}

void observe(ggml_tensor * node, Snapshot before, const std::function<void()> & replay) {
    if (!eligible(node)) return;
    auto & config = settings();
    const int selected = anchor(node);
    const auto after = snapshot();
    const auto identity = key(node);
    std::vector<float> original;
    nlohmann::json row{{"schema","potal-sensitivity-v1"},{"key",identity},{"node",node->name},
                       {"bits",GGML_GEMMINI_ACTIVATION_BITS},{"selected_anchor",selected},
                       {"logical_macs",uint64_t(ggml_nelements(node))*uint64_t(node->src[0]->ne[0])}};
    if (profiling()) {
        if (node->type != GGML_TYPE_F32 || !ggml_is_contiguous(node))
            throw std::runtime_error("Sensitivity requires contiguous E8M23 outputs");
        original.resize(ggml_nelements(node));
        std::memcpy(original.data(),node->data,ggml_nbytes(node));
        const auto samples = reference(*node);
        row["samples"] = samples.indices.size();
        row["reference_energy"] = samples.energy;
        row["reference"] = "double_dot_stored_weight_same_input";
        row[std::to_string(selected)] = measurement(original,samples,before,after);
        const auto start = snapshot();
        override_anchor = selected == 2 ? 3 : 2;
        try { replay(); } catch (...) { override_anchor = 0; std::memcpy(node->data,original.data(),ggml_nbytes(node)); throw; }
        override_anchor = 0;
        const auto end = snapshot();
        std::vector<float> alternate(original.size());
        std::memcpy(alternate.data(),node->data,ggml_nbytes(node));
        std::memcpy(node->data,original.data(),ggml_nbytes(node));
        row[std::to_string(selected == 2 ? 3 : 2)] = measurement(alternate,samples,start,end);
    } else {
        row["main_fragments"] = after.main-before.main;
        row["residual_fragments"] = after.residual-before.residual;
        row["saturations"] = after.saturations-before.saturations;
    }
    std::lock_guard<std::mutex> lock(config.mutex);
    row["call"] = config.calls[identity]++;
    if (config.log.is_open()) { config.log << row.dump() << '\n'; config.log.flush(); }
    else if (std::getenv("GGML_GEMMINI_ANCHOR_TRACE")) std::fprintf(stderr,"POTAL_ANCHOR_NODE %s\n",row.dump().c_str());
}

bool finish() {
    auto & config = settings();
    const bool ok = config.used.size() == config.policy.size();
    nlohmann::json result{{"schema","potal-anchor-policy-v1"},{"complete",ok},
        {"rounding","ties_away"},
        {"default_anchor",config.fallback},{"policy_keys",config.policy.size()},
        {"used_policy_keys",config.used.size()},{"profiling",profiling()}};
    std::fprintf(stderr,"POTAL_ANCHOR_POLICY_PROOF %s\n",result.dump().c_str());
    return ok;
}
} // namespace ggml::gemmini::sensitivity
