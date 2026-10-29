// H3-7B host tests. The runner uses the existing RTL-semantics emulator, not
// attention::reference. These tests prove software semantics, not FPGA behavior.
// QK produces raw, unmasked logits. Causal masking belongs to CPU softmax.
#include "ggml.h"
#include "ggml-backend-impl.h"
#include "ggml-gemmini.h"
#include "matmul/attention.hpp"
#include "matmul/stripe-api.hpp"
#include "support/attention-golden.hpp"
#include "support/attention-emulator.hpp"
#include "support/potal-emulator.hpp"

#include <algorithm>
#include <array>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace {

namespace attention = ggml::potal::attention;
namespace golden = potal_test::attention_golden;
using attention::role;
using attention::stationary;
using attention::stream;
using namespace ggml::potal;

// Preserve the historical normalized P input for both dimensions. Even when
// DIM32 shares this microtile, the certified bound is 64*2^24 = 2^30.
constexpr size_t pv_peak = 16;

int failures = 0;

bool check(const std::string & name, bool ok) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", name.c_str());
    failures += !ok;
    return ok;
}

// Hand-built GGML metadata permits invalid-shape tests without invoking GGML's
// graph-construction assertions. Byte strides include both element and row gaps.
struct tensor {
    ggml_tensor t{};
    std::vector<uint8_t> storage;

    tensor(ggml_type type, size_t k, size_t rows, size_t heads = 1, size_t batches = 1, bool gaps = false) {
        t.type = type;
        t.ne[0] = k; t.ne[1] = rows; t.ne[2] = heads; t.ne[3] = batches;
        const size_t bytes = type == GGML_TYPE_F16 ? 2 : 4;
        t.nb[0] = bytes * (gaps ? 2 : 1);
        t.nb[1] = t.nb[0] * k + (gaps ? 12 : 0);
        t.nb[2] = t.nb[1] * rows + (gaps ? 20 : 0);
        t.nb[3] = t.nb[2] * heads + (gaps ? 28 : 0);
        storage.assign(std::max<size_t>(1, t.nb[3] * batches), 0);
        t.data = storage.data();
    }

    size_t offset(size_t k, size_t row, size_t head = 0, size_t batch = 0) const {
        return k*t.nb[0] + row*t.nb[1] + head*t.nb[2] + batch*t.nb[3];
    }
    void set(size_t k, size_t row, float value, size_t head = 0, size_t batch = 0) {
        auto * p = static_cast<uint8_t *>(t.data) + offset(k, row, head, batch);
        if (t.type == GGML_TYPE_F16) {
            const auto half = ggml_fp32_to_fp16(value);
            std::memcpy(p, &half, sizeof(half));
        } else {
            std::memcpy(p, &value, sizeof(value));
        }
    }
    float get(size_t k, size_t row, size_t head = 0, size_t batch = 0) const {
        const auto * p = static_cast<const uint8_t *>(t.data) + offset(k, row, head, batch);
        if (t.type == GGML_TYPE_F16) {
            ggml_fp16_t half;
            std::memcpy(&half, p, sizeof(half));
            return ggml_fp16_to_fp32(half);
        }
        float value;
        std::memcpy(&value, p, sizeof(value));
        return value;
    }
    void fill(float value) {
        for (size_t b = 0; b < size_t(t.ne[3]); ++b)
            for (size_t h = 0; h < size_t(t.ne[2]); ++h)
                for (size_t i = 0; i < size_t(t.ne[1]); ++i)
                    for (size_t k = 0; k < size_t(t.ne[0]); ++k) set(k, i, value, h, b);
    }
};

struct graph {
    tensor w, a, out;
    graph(role r, size_t k, size_t n, size_t m, bool gaps = false,
          size_t heads = 1, size_t head_repeat = 1, size_t batches = 1, size_t batch_repeat = 1,
          ggml_type wt = GGML_TYPE_F16, ggml_type at = GGML_TYPE_F32) :
        w(wt, k, n, heads, batches, gaps),
        a(at, k, m, heads*head_repeat, batches*batch_repeat, gaps),
        out(GGML_TYPE_F32, n, m, heads*head_repeat, batches*batch_repeat) {
        out.t.op = GGML_OP_MUL_MAT;
        out.t.src[0] = &w.t;
        out.t.src[1] = &a.t;
        ggml_gemmini_set_attention_role(&out.t, r == role::qk ?
            GGML_GEMMINI_ATTENTION_QK : GGML_GEMMINI_ATTENTION_PV);
        out.fill(12345.0f);
    }
};

stream make_stream(size_t rows, size_t k, const std::vector<int32_t> & codes, role r, size_t count = 5) {
    stream a;
    a.rows = rows; a.k = k; a.kp = (k + 31)/32*32; a.limb_count = count;
    a.codes.assign(rows*a.kp, 0);
    for (auto & limb : a.limbs) limb.assign(rows*a.kp, 0);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t x = 0; x < k; ++x) {
            a.codes[i*a.kp+x] = codes[i*k+x];
            const auto d = golden::digits(codes[i*k+x], r == role::pv);
            for (size_t l = 0; l < count; ++l) a.limbs[l][i*a.kp+x] = d[l];
        }
    }
    return a;
}

stationary make_weight(size_t k, size_t n, const std::vector<int32_t> & codes, role r,
                       uint32_t carrier = 0) {
    stationary b;
    b.k = k; b.n = n; b.kp = (k+31)/32*32;
    b.base.assign(n, 0);
    b.carriers.assign(b.kp/32*n, carrier);
    b.limbs[0].assign(b.kp*n, 0);
    if (r == role::qk) b.limbs[1].assign(b.kp*n, 0);
    for (size_t x = 0; x < k; ++x) {
        for (size_t j = 0; j < n; ++j) {
            if (r == role::pv) b.limbs[0][x*n+j] = static_cast<int8_t>(codes[x*n+j]);
            else {
                const auto d = golden::digits(codes[x*n+j]);
                b.limbs[0][x*n+j] = d[0];
                b.limbs[1][x*n+j] = d[1];
            }
        }
    }
    return b;
}

void compare_reference(const std::string & name, const stream & a, const stationary & b, role r) {
    std::vector<int64_t> want, got = {778899};
    if (!check(name+" independent golden fits INT64", golden::evaluate(a, b, r, want))) return;
    const auto status = attention::reference(a, b, r, got);
    check(name+" reference status", status == POTAL_OK);
    check(name+" independent integer values", got == want);
}

void test_reference() {
    const std::vector<int32_t> q = {INT32_MAX, INT32_MIN, -1, 0, 127, 128, -128, -129, 65535};
    auto a = make_stream(q.size(), 1, q, role::qk);
    auto b = make_weight(1, 4, {-32768, 32639, 255, -257}, role::qk);
    check("INT32_MAX uses fifth Q limb", a.limbs[4][0] == 1 && a.limbs[0][0] == -1);
    check("INT32_MIN stays exact", a.limbs[3][a.kp] == -128 && a.limbs[4][a.kp] == 0);
    compare_reference("all five Q limbs and K limits", a, b, role::qk);

    for (size_t k : {size_t(1), size_t(17), size_t(31), size_t(32), size_t(33), size_t(64), size_t(65)}) {
        std::vector<int32_t> ac(2*k), wc(k*3);
        for (size_t i = 0; i < ac.size(); ++i) ac[i] = int32_t((i*773)%100003)-50001;
        for (size_t i = 0; i < wc.size(); ++i) wc[i] = int32_t((i*997)%65408)-32768;
        auto x = make_stream(2, k, ac, role::qk);
        auto w = make_weight(k, 3, wc, role::qk);
        for (size_t block = 0; block < w.kp/32; ++block)
            for (size_t j = 0; j < w.n; ++j)
                w.carriers[block*w.n+j] = (block+j)%3 == 0 ? POTAL_CARRIER_ZERO : uint32_t((block+j)%5);
        compare_reference("QK tail "+std::to_string(k), x, w, role::qk);
    }

    // Independent passes saturate before radix weighting. High uses Delta,
    // not Delta+8; the obsolete shared recurrence finishes at INT32_MIN
    // for two DIM16 partials, or -1 for one DIM32 partial.
    a = make_stream(1, 32, std::vector<int32_t>(32, 127), role::qk, 1);
    b = make_weight(32, 1, std::vector<int32_t>(32, -129), role::qk, 20);
    std::vector<int64_t> shared, split;
    check("QK independent low/high signed raw ACCs",
          golden::qk_pass(a,b,0,0) == std::vector<int32_t>{INT32_MAX} &&
          golden::qk_pass(a,b,0,1) == std::vector<int32_t>{INT32_MIN});
    check("QK independent and legacy witnesses evaluate", golden::evaluate(a,b,role::qk,shared) &&
          golden::evaluate(a,b,role::qk,split,true));
    check("QK independent passes reject old shared result",
          shared == std::vector<int64_t>{-547608330241LL} &&
          split == std::vector<int64_t>{POTAL_HW_DIM == 16 ? INT32_MIN : -1} && shared != split);
    compare_reference("QK complete low/high passes at original Delta", a,b,role::qk);
    auto corrupted = a;
    corrupted.limbs[0].assign(corrupted.codes.size(),0);
    std::vector<int64_t> independent;
    check("QK oracle derives Q digits from original codes",
          golden::evaluate(corrupted,b,role::qk,independent) && independent == shared);

    const std::vector<int32_t> p = {0, 1, 127, 128, 255, 256, 32768, 65408, 65535};
    a = make_stream(p.size(), 1, p, role::pv);
    b = make_weight(1, 3, {-128, 127, -1}, role::pv);
    check("P upper 255 has signed digit and carry",
          a.limbs[1][8*a.kp] == -1 && a.limbs[2][8*a.kp] == 1);
    check("P low uses unsigned byte minus 128", a.limbs[0][0] == -128 && a.limbs[0][8*a.kp] == 127);
    compare_reference("PV unbiased reconstruction including upper carry", a,b,role::pv);
    a = make_stream(1, 32, std::vector<int32_t>(32, 128), role::pv, 1);
    std::vector<int32_t> vcodes(32,0);
    vcodes[0] = 1; vcodes[16] = -1;
    b = make_weight(32, 1, vcodes, role::pv, 24);
    int32_t bias = 99;
    check("PV exact block bias cancels before INT32 check",golden::column_bias(b,0,bias) && bias == 0);
    check("PV carrier24 independent cancellation expects zero",
          golden::evaluate(a,b,role::pv,shared) && shared == std::vector<int64_t>{0});
    compare_reference("PV carrier24 bias cancellation",a,b,role::pv);

    // Mixed-sign blocks cancel in exact wide arithmetic even though each
    // individual weighted block bias is outside INT32.
    a = make_stream(1,64,std::vector<int32_t>(64,128),role::pv,1);
    vcodes.assign(64,0); vcodes[0] = 1; vcodes[32] = -1;
    b = make_weight(64,1,vcodes,role::pv,24);
    compare_reference("PV cross-block exact bias cancellation",a,b,role::pv);
    auto wide_weight = b;
    wide_weight.limbs[0].assign(wide_weight.kp, 0);
    wide_weight.limbs[0][0] = 1;
    wide_weight.limbs[0][32] = -2;
    wide_weight.carriers = {200, 199};
    std::vector<int32_t> wide_bias;
    check("PV exact cancellation beyond INT128 width",
          attention::column_bias(wide_weight, wide_bias) == POTAL_OK && wide_bias == std::vector<int32_t>{0});
    wide_weight.limbs[0][32] = -1;
    check("PV noncancelling wide bias explicitly unsupported",
          attention::column_bias(wide_weight, wide_bias) == POTAL_ERR_UNSUPPORTED);
    b.limbs[0][32] = 0;
    std::vector<int64_t> rejected = {778899};
    check("PV outside-INT32 exact bias rejected by golden",!golden::column_bias(b,0,bias));
    check("PV outside-INT32 bias now selects exact bitplanes",
          attention::reference(a,b,role::pv,rejected) == POTAL_OK &&
          rejected == std::vector<int64_t>{2147483648LL});

    b.limbs[0][0] = -2;
    check("PV negative outside-INT32 bias now selects exact bitplanes",
          attention::reference(a,b,role::pv,rejected) == POTAL_OK &&
          rejected == std::vector<int64_t>{-4294967296LL});
    b.limbs[0][0] = -1;
    check("PV exact INT32_MIN bias remains supported",golden::column_bias(b,0,bias) && bias == INT32_MIN);
    compare_reference("PV final atomic addition at negative bias limit",a,b,role::pv);

    // The low ACC saturates negative and positive in consecutive microtiles.
    // The exact bias is zero, but that does not undo low-ACC saturation.
    a = make_stream(1,2*POTAL_HW_DIM,std::vector<int32_t>(2*POTAL_HW_DIM,0),role::pv,1);
    vcodes.assign(2*POTAL_HW_DIM,0); vcodes[0] = 1; vcodes[POTAL_HW_DIM] = -1;
    b = make_weight(2*POTAL_HW_DIM,1,vcodes,role::pv,24);
    check("PV mixed saturation independent expected minus one",
          golden::evaluate(a,b,role::pv,shared) && shared == std::vector<int64_t>{-1});
    compare_reference("PV mixed saturation signs",a,b,role::pv);
    std::swap(b.limbs[0][0],b.limbs[0][POTAL_HW_DIM]);
    compare_reference("PV reversed mixed saturation signs",a,b,role::pv);
    a = make_stream(1,4*POTAL_HW_DIM,std::vector<int32_t>(4*POTAL_HW_DIM,0),role::pv,1);
    vcodes.assign(4*POTAL_HW_DIM,0);
    vcodes[0] = vcodes[POTAL_HW_DIM] = 1;
    vcodes[2*POTAL_HW_DIM] = vcodes[3*POTAL_HW_DIM] = -1;
    b = make_weight(4*POTAL_HW_DIM,1,vcodes,role::pv,24);
    check("PV repeated mixed signs expose fixed microtile order",
          golden::evaluate(a,b,role::pv,shared) && shared == std::vector<int64_t>{INT32_MAX-1});
    compare_reference("PV two negative then two positive partials",a,b,role::pv);
    for (auto & v : b.limbs[0]) v = -v;
    check("PV reversed repeated signs have different saturation",
          golden::evaluate(a,b,role::pv,shared) && shared == std::vector<int64_t>{INT32_MIN});
    compare_reference("PV two positive then two negative partials",a,b,role::pv);
    b.carriers.assign(b.carriers.size(), POTAL_CARRIER_ZERO);
    compare_reference("zero carriers suppress biased low and correction", a,b,role::pv);

    std::vector<int32_t> pcodes(96,128);
    pcodes[0] = 255; pcodes[32] = 0;
    a = make_stream(1,96,pcodes,role::pv,1);
    vcodes.assign(96,0);
    vcodes[0] = 1; vcodes[32] = -1; vcodes[64] = 1;
    b = make_weight(96,1,vcodes,role::pv);
    b.carriers = {24,24,0};
    check("PV positive final correction saturates exactly once",
          golden::column_bias(b,0,bias) && bias == 128 &&
          golden::evaluate(a,b,role::pv,shared) && shared == std::vector<int64_t>{INT32_MAX});
    compare_reference("PV positive final Sat32 addition",a,b,role::pv);
    for (auto & v : b.limbs[0]) v = -v;
    check("PV negative final correction saturates exactly once",
          golden::column_bias(b,0,bias) && bias == -128 &&
          golden::evaluate(a,b,role::pv,shared) && shared == std::vector<int64_t>{INT32_MIN});
    compare_reference("PV negative final Sat32 addition",a,b,role::pv);

    // A fifth-limb contribution and a fourth-limb contribution have the same
    // sign here. Both completed ACCs are large positive values; reconstruction
    // overflows only after the complete fifth limb is added.
    a = make_stream(1, 4*POTAL_HW_DIM, std::vector<int32_t>(4*POTAL_HW_DIM, 0), role::qk);
    a.codes[0] = INT32_MAX;
    a.codes[POTAL_HW_DIM] = 0x7f000000;
    a.codes[2*POTAL_HW_DIM] = 0x7f000000;
    for (size_t x = 0; x < a.k; ++x) {
        const auto d = golden::digits(a.codes[x]);
        for (size_t l = 0; l < 5; ++l) a.limbs[l][x] = d[l];
    }
    b = make_weight(a.k, 1, std::vector<int32_t>(a.k, 1), role::qk, 31);
    std::vector<int64_t> got = {778899};
    check("independent completed-limb reconstruction overflows", !golden::evaluate(a,b,role::qk,shared));
    check("reference rejects INT64 overflow", attention::reference(a,b,role::qk,got) != POTAL_OK);
    check("overflow publishes no partial reference output", got == std::vector<int64_t>{778899});

    // Limb 3 overflows INT64 before limb 4 cancels it. A wide final-only
    // range check would wrongly accept this otherwise representable result.
    std::vector<int32_t> overflow_q(33,0), overflow_k(33,0);
    overflow_q[0] = 0x7f000000; overflow_k[0] = 257;
    overflow_q[32] = INT32_MAX; overflow_k[32] = -1;
    a = make_stream(1,33,overflow_q,role::qk);
    b = make_weight(33,1,overflow_k,role::qk);
    b.carriers = {31,23};
    const golden::wide final_only = golden::wide{8388608} +
        golden::wide{257} * INT32_MAX * (golden::wide{1} << 24) -
        (golden::wide{1} << 55);
    check("QK later cancellation would fit final INT64",
          final_only >= INT64_MIN && final_only <= INT64_MAX);
    shared = {778899};
    check("QK intermediate overflow rejects before later cancellation",
          !golden::evaluate(a,b,role::qk,shared) && shared == std::vector<int64_t>{778899});
    got = {778899};
    check("QK reference intermediate overflow publishes nothing",
          attention::reference(a,b,role::qk,got) == POTAL_ERR_CAPACITY &&
          got == std::vector<int64_t>{778899});
}

bool limbs_match(const stream & a, role r) {
    if (a.limb_count == 0 || a.limb_count > attention::max_limbs || a.codes.size() != a.rows*a.kp) return false;
    for (size_t l = 0; l < a.limb_count; ++l) {
        if (a.limbs[l].size() != a.codes.size()) return false;
        for (size_t i = 0; i < a.rows; ++i) {
            for (size_t k = 0; k < a.kp; ++k) {
                const int want = k < a.k ? golden::digits(a.codes[i*a.kp+k], r == role::pv)[l] : 0;
                if (a.limbs[l][i*a.kp+k] != want) return false;
            }
        }
    }
    return true;
}

bool same_stream(const stream & a, const stream & b) {
    return a.rows == b.rows && a.k == b.k && a.kp == b.kp && a.limb_count == b.limb_count &&
           a.scale == b.scale && a.codes == b.codes && a.limbs == b.limbs;
}
bool same_weight(const stationary & a, const stationary & b) {
    return a.k == b.k && a.n == b.n && a.kp == b.kp && a.base == b.base &&
           a.carriers == b.carriers && a.limbs == b.limbs;
}

void test_prepare() {
    for (ggml_type type : {GGML_TYPE_F16, GGML_TYPE_F32}) {
        tensor t(type, 33, 2, 2, 2, true);
        t.fill(0);
        for (size_t k = 0; k < 33; ++k) {
            t.set(k,0,k < 32 ? 1.0f : 2.0f,1,1);
            t.set(k,1,-1.0f,1,1);
        }
        stationary w;
        if (check("prepare K strided head/batch", attention::prepare_stationary(t.t,1,1,role::qk,w) == POTAL_OK)) {
            check("K dimensions and block bases", w.k == 33 && w.n == 2 && w.kp == 64 &&
                  w.base == std::vector<int16_t>({-14,-14}));
            check("K exact block exponent differences", w.carriers == std::vector<uint32_t>({0,0,1,0}));
            bool exact = w.limbs[0].size() == 128 && w.limbs[1].size() == 128;
            for (size_t k = 0; exact && k < 64; ++k) {
                exact = w.limbs[0][2*k] == 0 && w.limbs[0][2*k+1] == 0 &&
                        w.limbs[1][2*k] == (k < 33 ? 64 : 0) &&
                        w.limbs[1][2*k+1] == (k < 33 ? -64 : 0);
            }
            check("K independently calculated signed limbs and padding", exact);
        }
        if (check("prepare V signed8", attention::prepare_stationary(t.t,1,1,role::pv,w) == POTAL_OK)) {
            bool exact = w.base == std::vector<int16_t>({-6,-6}) && w.limbs[0].size() == 128;
            for (size_t k = 0; exact && k < 64; ++k)
                exact = w.limbs[0][2*k] == (k < 33 ? 64 : 0) &&
                        w.limbs[0][2*k+1] == (k < 33 ? -64 : 0);
            check("V independently calculated signed8 and padding", exact);
        }
        // A single exponent bucket has zero magnitude variance. No local
        // adaptation is required: Q=+/-1 maps exactly to +/-16384 at scale -14.
        t.fill(1);
        for (size_t k = 0; k < 33; ++k) t.set(k,1,-1,1,1);
        stream s;
        if (check("prepare simple Q", attention::prepare_stream(t.t,1,1,0,2,role::qk,s) == POTAL_OK)) {
            bool exact = s.scale == -14 && s.k == 33 && s.kp == 64 && s.rows == 2 &&
                         s.codes.size() == 128;
            for (size_t i = 0; exact && i < 2; ++i)
                for (size_t k = 0; exact && k < 64; ++k)
                    exact = s.codes[i*64+k] == (k < 33 ? (i ? -16384 : 16384) : 0);
            check("Q simple F16/F32 scale, codes and limbs", exact && limbs_match(s,role::qk));
        }
        t.fill(0);
        if (check("prepare zero stationary", attention::prepare_stationary(t.t,0,0,role::qk,w) == POTAL_OK))
            check("zero blocks have zero carriers", w.carriers.size() == 4 &&
                  std::all_of(w.carriers.begin(),w.carriers.end(),
                  [](uint32_t v) { return v == POTAL_CARRIER_ZERO; }) &&
                  std::all_of(w.limbs[0].begin(),w.limbs[0].end(),[](int8_t v) { return v == 0; }) &&
                  std::all_of(w.limbs[1].begin(),w.limbs[1].end(),[](int8_t v) { return v == 0; }));
        if (check("prepare zero stream",attention::prepare_stream(t.t,0,0,0,2,role::qk,s) == POTAL_OK))
            check("zero Q codes and padding",s.codes.size() == 128 &&
                  std::all_of(s.codes.begin(),s.codes.end(),[](int32_t v) { return v == 0; }) &&
                  limbs_match(s,role::qk));
    }

    tensor ties(GGML_TYPE_F32,32,1);
    ties.fill(1);
    // P scale -15; these values are exactly 0.5, 1.5, 2.5 and 3.5 quanta.
    for (size_t k = 0; k < 4; ++k) ties.set(k,0,float(2*k+1)/65536.0f);
    const int saved_round = std::fegetround();
    for (int mode : {FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO, FE_TONEAREST}) {
        if (!check("set caller rounding mode", std::fesetround(mode) == 0)) continue;
        stream p;
        const auto status = attention::prepare_stream(ties.t,0,0,0,1,role::pv,p);
        check("P quantization restores caller rounding", std::fegetround() == mode);
        if (check("prepare P ties",status == POTAL_OK))
            check("P RNE ties, unsigned upper carry", p.scale == -15 && p.codes.size() == 32 &&
                  p.codes[0] == 0 && p.codes[1] == 2 && p.codes[2] == 2 && p.codes[3] == 4 &&
                  p.codes[4] == 32768 && limbs_match(p,role::pv));
        tensor qties(GGML_TYPE_F32,32,1);
        qties.fill(1);
        qties.set(0,0,1.0f+1.0f/32768);
        qties.set(1,0,1.0f+3.0f/32768);
        qties.set(2,0,-1.0f-1.0f/32768);
        qties.set(3,0,-1.0f-3.0f/32768);
        stream q;
        const auto qstatus = attention::prepare_stream(qties.t,0,0,0,1,role::qk,q);
        check("Q quantization restores caller rounding",std::fegetround() == mode);
        if (check("prepare Q ties",qstatus == POTAL_OK))
            check("Q signed RNE ties",q.scale == -14 && q.codes.size() == 32 &&
                  q.codes[0] == 16384 && q.codes[1] == 16386 &&
                  q.codes[2] == -16384 && q.codes[3] == -16386 && limbs_match(q,role::qk));
    }
    check("restore original rounding mode", saved_round != -1 && std::fesetround(saved_round) == 0);
    tensor pclip(GGML_TYPE_F32,1,1);
    pclip.set(0,0,std::nextafter(1.0f,0.0f));
    stream clipped_p;
    if (check("prepare P unsigned16 clipping",attention::prepare_stream(pclip.t,0,0,0,1,role::pv,clipped_p) == POTAL_OK))
        check("P clips RNE overflow to 65535 with upper carry",clipped_p.scale == -16 &&
              clipped_p.codes.size() == 32 && clipped_p.codes[0] == 65535 &&
              clipped_p.limb_count >= 3 && limbs_match(clipped_p,role::pv));

    // The K block maximum is just below 2. At theta=-14 the positive code
    // clips to 32639 (not 32767); the negative endpoint remains -32768.
    tensor clip(GGML_TYPE_F32,32,1);
    clip.fill(0);
    clip.set(0,0,std::nextafter(2.0f,0.0f));
    clip.set(1,0,-std::nextafter(2.0f,0.0f));
    stationary w;
    if (check("prepare K clipping",attention::prepare_stationary(clip.t,0,0,role::qk,w) == POTAL_OK))
        check("K asymmetric clipping endpoints", w.base == std::vector<int16_t>{-14} &&
              w.limbs[0].size() >= 2 && w.limbs[1].size() >= 2 &&
              w.limbs[0][0] == 127 && w.limbs[1][0] == 127 &&
              w.limbs[0][1] == 0 && w.limbs[1][1] == -128);
    if (check("prepare V clipping",attention::prepare_stationary(clip.t,0,0,role::pv,w) == POTAL_OK))
        check("V signed8 endpoints",w.limbs[0].size() >= 2 && w.limbs[0][0] == 127 && w.limbs[0][1] == -128);

    stream sentinel = make_stream(1,1,{7},role::qk);
    for (float bad : {-0.01f, std::numeric_limits<float>::infinity(),
                      -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) {
        ties.set(7,0,bad);
        auto out = sentinel;
        check("P negative/nonfinite rejected",attention::prepare_stream(ties.t,0,0,0,1,role::pv,out) != POTAL_OK);
        check("failed P prepare restores rounding",std::fegetround() == saved_round);
        check("failed P prepare leaves output unchanged",same_stream(out,sentinel));
        ties.set(7,0,1);
    }
    for (float bad : {std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) {
        clip.set(0,0,bad);
        const auto old = w;
        auto out = w;
        check("nonfinite stationary rejected",attention::prepare_stationary(clip.t,0,0,role::qk,out) != POTAL_OK);
        check("failed stationary prepare publishes nothing",same_weight(out,old));
        auto qout = sentinel;
        check("nonfinite Q rejected",attention::prepare_stream(clip.t,0,0,0,1,role::qk,qout) != POTAL_OK);
        check("failed Q prepare publishes nothing",same_stream(qout,sentinel));
    }
    auto out = sentinel;
    check("zero requested rows rejected",attention::prepare_stream(ties.t,0,0,0,0,role::pv,out) != POTAL_OK);
    check("out-of-range stream row rejected",attention::prepare_stream(ties.t,0,0,1,1,role::pv,out) != POTAL_OK);
    check("out-of-range head rejected",attention::prepare_stream(ties.t,1,0,0,1,role::pv,out) != POTAL_OK);
    check("out-of-range batch rejected",attention::prepare_stationary(ties.t,0,1,role::pv,w) != POTAL_OK);
}

struct emu_runner {
    potal_test::emulator emu;
    potal_ws_ops ops = emu.ops();
    potal_stripe_session session = {{&ops,0,{}}};
    uint64_t calls = 0;
    bool panels_ok = true;
    size_t logical_k;
    potal_stripe_gemm_desc low{};
    std::vector<int8_t> low_a;
    std::vector<uint32_t> low_carriers;

    explicit emu_runner(size_t k) : logical_k((k+31)/32*32) {}

    planned_result run(const potal_stripe_gemm_desc & d) {
        ++calls;
        // Independent complete passes retain the old doubled-K tile selection,
        // but never duplicate A or add 8 to the high pass's carriers.
        // DMA row alignment can pad lda beyond logical K; that padding is
        // zero, not a second copy of the Q limb.
        panels_ok = panels_ok && d.k == logical_k && d.lda >= logical_k &&
                    d.scale_block_k == 32 && !emu.in_flight;
        potal_tile_plan frozen;
        const auto selected = potal_tile_select(d.m,d.n,2*logical_k,32/POTAL_HW_DIM,&frozen);
        panels_ok = panels_ok && selected == POTAL_OK;
        if (selected == POTAL_OK) {
            frozen.tile_K = std::min(frozen.tile_K,logical_k/POTAL_HW_DIM);
            while (frozen.tile_K && (logical_k/POTAL_HW_DIM % frozen.tile_K || frozen.tile_K % (32/POTAL_HW_DIM)))
                --frozen.tile_K;
            panels_ok = panels_ok && d.tile_i == frozen.tile_I && d.tile_j == frozen.tile_J &&
                        d.tile_k == frozen.tile_K;
            if (logical_k == 4128)
                panels_ok = panels_ok && frozen.tile_K * POTAL_HW_DIM < logical_k;
        }
        if (d.k == logical_k && d.carriers && d.b && d.a) {
            std::vector<int8_t> activation;
            std::vector<uint32_t> carriers;
            for (size_t i = 0; i < d.m; ++i)
                for (size_t k = 0; k < logical_k; ++k)
                    activation.push_back(d.a[i*d.lda+k]);
            for (size_t i = 0; i < d.m; ++i)
                for (size_t k = logical_k; k < d.lda; ++k)
                    panels_ok = panels_ok && d.a[i*d.lda+k] == 0;
            for (size_t block = 0; block < logical_k/32; ++block) {
                for (size_t j = 0; j < d.n; ++j)
                    carriers.push_back(d.carriers[block*d.carrier_stride+j]);
            }
            if (calls % 2) {
                low = d;
                low_a = std::move(activation);
                low_carriers = std::move(carriers);
            } else {
                panels_ok = panels_ok && d.m == low.m && d.n == low.n &&
                            d.work_kind == low.work_kind && activation == low_a && carriers == low_carriers;
            }
        }
        return run_planned_stripe_gemm(session,d);
    }
};

void execute_case(role r, size_t k, size_t m, bool gaps, size_t repeat = 1,
                  ggml_type wt = GGML_TYPE_F16, ggml_type at = GGML_TYPE_F32, size_t n = 3) {
    graph g(r,k,n,m,gaps,2,repeat,2,repeat,wt,at);
    // Per-row constant magnitudes make the Q local stage independent of
    // adaptation policy. Every product is an exactly representable power of two.
    for (size_t b = 0; b < 2; ++b)
        for (size_t h = 0; h < 2; ++h)
            for (size_t j = 0; j < n; ++j)
                for (size_t x = 0; x < k; ++x)
                    g.w.set(x,j,std::ldexp(j == 1 ? -1.0f : 1.0f,int(h+b+j%3)),h,b);
    for (size_t b = 0; b < 2*repeat; ++b)
        for (size_t h = 0; h < 2*repeat; ++h)
            for (size_t i = 0; i < m; ++i)
                for (size_t x = 0; x < k; ++x)
                    g.a.set(x,i,r == role::pv ? std::ldexp(1.0f,-int(i%3+1)) :
                            std::ldexp(i%2 ? -1.0f : 1.0f,int(i%3)),h,b);
    check("execute contract accepts strided/repeated graph",attention::check(&g.out.t,true) == GGML_POTAL_REJECT_NONE);
    emu_runner e(k);
    const auto result = attention::execute(&g.out.t,[&](const potal_stripe_gemm_desc & d) { return e.run(d); });
    const std::string name = (r == role::qk ? "QK " : "PV ")+std::to_string(k)+"x"+std::to_string(m);
    if (!check(name+" emulator execution",result.status == POTAL_OK && !result.fatal)) return;
    bool exact = true;
    for (size_t b = 0; b < 2*repeat; ++b)
        for (size_t h = 0; h < 2*repeat; ++h)
            for (size_t i = 0; i < m; ++i)
                for (size_t j = 0; j < n; ++j) {
                    // Direct float math is exact for this fixture; no production
                    // preparation or attention reference derives the answer.
                    const float want = float(k)*g.a.get(0,i,h,b)*g.w.get(0,j,h/repeat,b/repeat);
                    exact = exact && g.out.get(j,i,h,b) == want;
                }
    check(name+" independent values across all heads/batches",exact);
    check(name+" independent low/high descriptors and frozen tiles",
          e.panels_ok && e.calls > 0 && e.calls % 2 == 0 && result.qk_limb_jobs == e.calls);
    check(name+" QK has no PV correction",result.correction_jobs == 0);
    check(name+" balanced scales and no live DMA",e.emu.writes == e.emu.releases && e.emu.scales.empty() &&
          !e.emu.in_flight && !e.emu.scale_miss);
    check(name+" returned command counts",result.jobs == e.calls && result.loops == e.emu.loops &&
          result.scale_writes == e.emu.writes && result.scale_releases == e.emu.releases &&
          result.tlb_flushes == e.emu.tlb_flushes && e.emu.fences == result.fences+result.tlb_flushes);
    check(name+" actual LOOP command count",uint64_t(std::count(e.emu.commands.begin(),e.emu.commands.end(),'L')+
          std::count(e.emu.commands.begin(),e.emu.commands.end(),'G')) == e.emu.loops);
}

void test_execute() {
    graph masked(role::qk, 32, 3, 2);
    masked.w.fill(1); masked.a.fill(1);
    tensor mask(GGML_TYPE_F32, 3, 2);
    mask.fill(0);
    mask.set(2, 0, -std::numeric_limits<float>::infinity());
    masked.out.t.src[2] = &mask.t;
    emu_runner unmasked(32);
    const auto raw = attention::execute(&masked.out.t,
        [&](const potal_stripe_gemm_desc & descriptor) { return unmasked.run(descriptor); });
    check("QK leaves causal mask to the later CPU softmax",
          raw.status == POTAL_OK && masked.out.get(2, 0) == 32.0f && masked.out.get(0, 0) == 32.0f);
    // PV execution needs the parent's new corrected_runner. Do not emulate it
    // with the disproved all-ones/SCU+7 recurrence.
    for (role r : {role::qk}) {
        execute_case(r,32,4,false); // Prefill.
        for (size_t k : {size_t(1),size_t(17),size_t(33),size_t(64),size_t(65)}) execute_case(r,k,1,true,2);
        execute_case(r,4097,1,true); // Multiple frozen K panels within each independent pass.
        execute_case(r,33,3,true,2,GGML_TYPE_F32,GGML_TYPE_F16,19);
        graph g(r,33,3,2,true);
        g.w.fill(0); g.a.fill(r == role::qk ? 1 : 0.5f);
        emu_runner zero(33);
        const auto z = attention::execute(&g.out.t,[&](const potal_stripe_gemm_desc & d) { return zero.run(d); });
        check("zero-block execute status",z.status == POTAL_OK);
        bool all_zero = true;
        for (size_t i = 0; i < 2; ++i)
            for (size_t j = 0; j < 3; ++j) all_zero = all_zero && g.out.get(j,i) == 0;
        check("zero-block output is zero",all_zero);

        // A successful first logical job must not leak a partially published
        // destination if a later job fails. The injected result is an explicit
        // runner failure, not a fake successful production implementation.
        for (int failure : {0,1,2,3,4}) {
            g.w.fill(1); g.a.fill(r == role::qk ? 1 : 0.5f); g.out.fill(12345);
            const auto before = g.out.storage;
            emu_runner e(33);
            if (failure == 0) e.emu.fail_issue = true;
            if (failure == 1) e.emu.fail_write_at = 0;
            if (failure == 2) e.emu.fail_tlb = true;
            if (failure == 3) e.emu.fail_fence_at = 1;
            if (failure == 4) e.emu.fail_release = true;
            const auto failed = attention::execute(&g.out.t,[&](const potal_stripe_gemm_desc & d) { return e.run(d); });
            check("emulator injected failure propagates",failed.status != POTAL_OK);
            check("emulator failure publishes no partial output",g.out.storage == before);
        }
        g.w.fill(1); g.a.fill(r == role::qk ? 1.00390625f : 0.5f); g.out.fill(12345);
        const auto before = g.out.storage;
        emu_runner later(33);
        size_t calls = 0;
        const auto failed = attention::execute(&g.out.t,[&](const potal_stripe_gemm_desc & d) {
            if (++calls == 2) {
                planned_result error;
                error.status = POTAL_ERR_BAD_RESPONSE;
                error.stage = "between_qk_passes";
                return error;
            }
            return later.run(d);
        });
        check("failure between low/high preserves status and completed count",
              calls == 2 && failed.status == POTAL_ERR_BAD_RESPONSE && !failed.fatal &&
              failed.stage && std::strcmp(failed.stage,"between_qk_passes") == 0 &&
              failed.qk_limb_jobs == 1 && later.calls == 1 && !later.emu.in_flight);
        check("later-job failure publishes no earlier results",g.out.storage == before);

        emu_runner fatal_high(33);
        calls = 0;
        const auto fatal = attention::execute(&g.out.t,[&](const potal_stripe_gemm_desc & d) {
            if (++calls == 2)
                fatal_high.emu.fail_fence_at = int(fatal_high.emu.fence_calls + 1);
            return fatal_high.run(d);
        });
        check("fatal high completion preserves fence failure and stops",
              calls == 2 && fatal.status == POTAL_HW_BAD_COMMAND && fatal.fatal &&
              fatal.stage && std::strcmp(fatal.stage,"fence") == 0 && fatal.qk_limb_jobs == 1);
        check("fatal high completion retains pending scales and sticky owner",
              fatal_high.emu.in_flight && !fatal_high.emu.scales.empty() &&
              fatal_high.emu.writes > fatal_high.emu.releases && fatal_high.session.ws.report.fatal);
        check("fatal high completion publishes no output",g.out.storage == before);
    }

    // A calculable K block with mixed low/high signs, plus a tiny second block
    // establishing carrier 20. Q's single exponent bucket gives code 16384.
    // This distinguishes independent original-Delta passes from shared/high+8.
    graph g(role::qk,64,1,1);
    g.a.fill(1);
    std::vector<int32_t> codes(64,16384);
    for (size_t k = 0; k < 32; ++k) codes[k] = -129;
    codes[16] = 16384;
    for (size_t k = 0; k < 64; ++k)
        g.w.set(k,0,k < 32 ? float(codes[k])/16384.0f : std::ldexp(1.0f,-20));
    auto a = make_stream(1,64,std::vector<int32_t>(64,16384),role::qk,2);
    auto b = make_weight(64,1,codes,role::qk);
    b.base[0] = -34; b.carriers = {20,0};
    // Q limb 1 is 64, with reconstruction weight 256. DIM16 high ACC:
    // Sat32(-1024*2^20) + Sat32(3136*2^20) + 131072 = 1073872895.
    // DIM32 merges that first block: Sat32(2112*2^20) = INT32_MAX;
    // adding 131072 stays saturated. Low is INT32_MAX in both cases.
    std::vector<int64_t> expected, forbidden;
    if (check("QK execute saturation witness fits",
              golden::evaluate(a,b,role::qk,expected) && golden::evaluate(a,b,role::qk,forbidden,true))) {
        check("QK execute exact independent and obsolete shared witnesses",
              expected == std::vector<int64_t>{(POTAL_HW_DIM == 16 ? 70927089860352LL : 141287244103424LL)} &&
              forbidden == std::vector<int64_t>{549755813632LL});
        check("QK execute independent active-limb raw ACC witnesses",
              golden::qk_pass(a,b,1,0) == std::vector<int32_t>{INT32_MAX} &&
              golden::qk_pass(a,b,1,1) == std::vector<int32_t>{(POTAL_HW_DIM == 16 ? 1073872895 : INT32_MAX)});
        emu_runner e(64);
        std::vector<std::vector<int32_t>> raw_passes;
        const auto result = attention::execute(&g.out.t,[&](const potal_stripe_gemm_desc & d) {
            auto completed = e.run(d);
            if (completed.ok) {
                std::vector<int32_t> values;
                for (size_t i = 0; i < d.m; ++i)
                    for (size_t j = 0; j < d.n; ++j) values.push_back(d.c[i*d.ldc+j]);
                raw_passes.push_back(std::move(values));
            }
            return completed;
        });
        const float want = static_cast<float>(std::ldexp(static_cast<double>(expected[0]),-48));
        check("QK execute independent-pass saturation result",
              result.status == POTAL_OK && want == (POTAL_HW_DIM == 16 ? 0.251983642578125f : 0.501953125f) && g.out.get(0,0) == want);
        bool raw_exact = raw_passes.size() == 2*a.limb_count;
        for (size_t pass = 0; raw_exact && pass < raw_passes.size(); ++pass)
            raw_exact = raw_passes[pass] == golden::qk_pass(a,b,pass/2,pass%2);
        check("QK execute every completed raw pass matches independent oracle",
              raw_exact && result.qk_limb_jobs == raw_passes.size());
        check("QK execute saturation panels and balanced scales",e.panels_ok && e.emu.loops > 0 &&
              e.emu.writes == e.emu.releases && e.emu.scales.empty());
    }

    // Real preparation produces five folded Q digits. The completed limb-3
    // term reaches INT64_MIN before an earlier negative term is included;
    // the later carry limb must not repair this checked intermediate overflow.
    graph overflow_graph(role::qk,64,1,1,false,1,1,1,1,GGML_TYPE_F32);
    for (size_t k = 0; k < 64; ++k) {
        overflow_graph.a.set(k,0,k < 32 ? std::ldexp(1.0f,20) : 1.0f);
        overflow_graph.w.set(k,0,k < 32 ? 1.0f : std::ldexp(1.0f,-30));
    }
    stream overflow_activation;
    stationary overflow_weight;
    const bool overflow_prepared =
        attention::prepare_stream(overflow_graph.a.t,0,0,0,1,role::qk,overflow_activation) == POTAL_OK &&
        attention::prepare_stationary(overflow_graph.w.t,0,0,role::qk,overflow_weight) == POTAL_OK;
    std::vector<int64_t> overflow_expected{778899};
    check("QK execute overflow fixture preserves folded codes and Delta",
          overflow_prepared && overflow_activation.limb_count == 5 &&
          overflow_activation.codes[0] == INT32_MAX &&
          overflow_weight.carriers == std::vector<uint32_t>{30,0});
    check("QK execute overflow is independently required",
          overflow_prepared && !golden::evaluate(overflow_activation,overflow_weight,role::qk,overflow_expected) &&
          overflow_expected == std::vector<int64_t>{778899});
    const auto untouched = overflow_graph.out.storage;
    emu_runner overflow_runner(64);
    const auto overflow_result = attention::execute(&overflow_graph.out.t,
        [&](const potal_stripe_gemm_desc & d) { return overflow_runner.run(d); });
    check("QK execute rejects overflow after completed passes",
          overflow_result.status == POTAL_ERR_CAPACITY && !overflow_result.fatal &&
          overflow_result.qk_limb_jobs == 8 && overflow_runner.calls == 8);
    check("QK execute overflow publishes nothing and leaves no DMA",
          overflow_graph.out.storage == untouched && !overflow_runner.emu.in_flight &&
          overflow_runner.emu.scales.empty() && overflow_runner.emu.writes == overflow_runner.emu.releases);
}

void test_pv_execute() {
    for (size_t k : {size_t(1), size_t(17), size_t(33), size_t(65)}) {
        graph g(role::pv, k, 19, k == 33 ? 4 : 1, true, 2, 2, 2, 2);
        for (size_t batch = 0; batch < 2; ++batch)
            for (size_t head = 0; head < 2; ++head)
                for (size_t j = 0; j < 19; ++j)
                    for (size_t x = 0; x < k; ++x)
                        g.w.set(x, j, std::ldexp(j % 2 ? -1.0f : 1.0f, int(head + batch + j % 3)), head, batch);
        for (size_t batch = 0; batch < 4; ++batch)
            for (size_t head = 0; head < 4; ++head)
                for (size_t i = 0; i < size_t(g.a.t.ne[1]); ++i)
                    for (size_t x = 0; x < k; ++x)
                        g.a.set(x, i, i % 2 ? 0.5f : 1.0f, head, batch);
        potal_test::emulator upper;
        auto ops = upper.ops();
        potal_stripe_session upper_session = {{&ops, 0, {}}};
        potal_test::attention_emulator low;
        auto sink = low.sink();
        uint64_t upper_calls = 0, low_calls = 0;
        bool upper_contract = true, bias_contract = true;
        const auto result = attention::execute(&g.out.t,
            [&](const potal_stripe_gemm_desc & descriptor) {
                ++upper_calls;
                upper_contract = upper_contract && descriptor.k == (k + 31) / 32 * 32 &&
                                 descriptor.work_kind == POTAL_STRIPE_WORK_RESIDUAL;
                return run_planned_stripe_gemm(upper_session, descriptor);
            },
            [&](const potal_stripe_gemm_desc & descriptor, const int32_t * bias, size_t columns) {
                ++low_calls;
                bias_contract = bias_contract && columns == 19 && descriptor.k == (k + 31) / 32 * 32;
                for (size_t j = 0; j < columns; ++j) {
                    const int32_t expected = int32_t(128 * 64 * k) * (j % 2 ? -1 : 1);
                    bias_contract = bias_contract && bias[j] == expected;
                }
                return attention::run_corrected(descriptor, bias, columns, &sink);
            });
        const std::string name = "PV command path K=" + std::to_string(k);
        if (!check(name + " status", result.status == POTAL_OK && !result.fatal)) continue;
        bool exact = true;
        for (size_t batch = 0; batch < 4; ++batch)
            for (size_t head = 0; head < 4; ++head)
                for (size_t i = 0; i < size_t(g.a.t.ne[1]); ++i)
                    for (size_t j = 0; j < 19; ++j)
                        exact = exact && g.out.get(j, i, head, batch) ==
                            float(k) * g.a.get(0, i, head, batch) * g.w.get(0, j, head / 2, batch / 2);
        check(name + " independent full output", exact);
        check(name + " unsigned upper carry uses separate jobs", upper_calls >= 2 * low_calls && upper_contract);
        check(name + " exact once-only bias operand", bias_contract && result.correction_jobs == low_calls &&
              result.bias_updates == low.bias_mvins && low.bias_mvins > 0);
        check(name + " real checked commands interpreted", low.errors.empty() && !low.scale_miss &&
              low.main_loops > 0 && low.mvouts == low.bias_mvins && low.scale_writes == low.scale_releases);
        check(name + " total command counts", result.jobs == upper_calls + low_calls &&
              result.loops == upper.loops + low.main_loops &&
              result.scale_writes == upper.writes + low.scale_writes &&
              result.scale_releases == upper.releases + low.scale_releases);
        check(name + " old representable bias never selects bitplanes",
              result.pv_bitplane_stripes == 0 && result.pv_bitplane_jobs == 0);
    }
    graph g(role::pv, 33, 3, 2);
    g.w.fill(1); g.a.fill(1);
    potal_test::attention_emulator failed;
    failed.fail_funct = POTAL_FUNCT_MVIN;
    failed.fail_status = POTAL_HW_BAD_COMMAND;
    auto sink = failed.sink();
    const auto before = g.out.storage;
    const auto result = attention::execute(&g.out.t,
        [](const potal_stripe_gemm_desc &) { return planned_result{}; },
        [&](const potal_stripe_gemm_desc & descriptor, const int32_t * bias, size_t columns) {
            return attention::run_corrected(descriptor, bias, columns, &sink);
        });
    check("PV failed bias MVIN propagates without partial publication",
          result.status == POTAL_HW_BAD_COMMAND && g.out.storage == before && failed.bias_mvins == 0);
    potal_paired_session * fresh = nullptr;
    check("PV successful cleanup releases accelerator ownership", potal_paired_session_open(&sink, &fresh) == POTAL_OK);
    check("PV cleanup session closes", potal_paired_session_close(fresh) == POTAL_OK);
}

// Test-only math never serves as a runner or CPU fallback.
void pv_plane_reference(const std::string & name, const stream & a, const stationary & b,
                        bool accepted) {
    golden::pv_planes planes;
    if (!check(name+" original-code plane oracle",golden::pv_bitplanes(a,b,planes))) return;
    bool digits_ok = !planes.digits.empty() && planes.digits.size() <= 16;
    for (size_t x = 0; x < a.codes.size(); ++x) {
        int32_t reconstructed = 0, place = 1;
        for (const auto & plane : planes.digits) {
            digits_ok = digits_ok && (plane[x] == 0 || plane[x] == 1);
            reconstructed += plane[x]*place;
            place *= 2;
        }
        digits_ok = digits_ok && reconstructed == a.codes[x];
    }
    check(name+" unsigned digits reconstruct original codes",digits_ok);
    std::vector<int64_t> got{778899};
    const auto status = attention::reference(a,b,role::pv,got);
    if (!accepted) {
        check(name+" certificate refuses before publication",
              !planes.certified && status == POTAL_ERR_UNSUPPORTED &&
              got == std::vector<int64_t>{778899});
        return;
    }
    std::vector<golden::wide> exact;
    check(name+" certificate and independent wide dot",planes.certified && golden::pv_exact(a,b,exact));
    bool equal = exact.size() == planes.integers.size();
    for (size_t x = 0; x < exact.size(); ++x)
        equal = equal && exact[x] == planes.integers[x];
    check(name+" raw planes reconstruct exact wide quantized dot",equal);
    check(name+" production reference equals independent planes",
          status == POTAL_OK && got == planes.integers);
}

void test_pv_bitplane_reference() {
    // New acceptance is triggered by column bias, never by a normal PV job.
    for (int sign : {-1,1}) {
        auto a = make_stream(1,64,std::vector<int32_t>(64,0),role::pv);
        a.codes[0] = 1; a.codes[pv_peak] = 65535;
        std::vector<int32_t> v(64,0);
        v[0] = sign*64; v[32] = sign*64;
        auto b = make_weight(64,1,v,role::pv,24);
        b.carriers[1] = 0;
        pv_plane_reference("PV signed overflow "+std::to_string(sign),a,b,true);
        // All zeros still execute one plane when the bias requires extension.
        a.codes.assign(a.codes.size(),0);
        golden::pv_planes zero;
        check("PV zero codes retain one unsigned plane",
              golden::pv_bitplanes(a,b,zero) && zero.digits.size() == 1);
        pv_plane_reference("PV zero codes and overflowing bias",a,b,true);
    }
    auto a = make_stream(1,96,std::vector<int32_t>(96,128),role::pv);
    std::vector<int32_t> v(96,0);
    v[0] = 127; v[32] = 127;
    for (size_t k = 64; k < 72; ++k) v[k] = 127;
    v[72] = 7;
    auto b = make_weight(96,1,v,role::pv);
    b.carriers = {17,10,0};
    int32_t bias = 0;
    check("PV largest positive multiple128 bias endpoint",
          golden::column_bias(b,0,bias) && bias == 2147483520);
    compare_reference("PV positive endpoint stays old biased limbs",a,b,role::pv);
    v.assign(96,0); v[0] = -1;
    b = make_weight(96,1,v,role::pv,24);
    check("PV exact negative bias endpoint",golden::column_bias(b,0,bias) && bias == INT32_MIN);
    compare_reference("PV negative endpoint stays old biased limbs",a,b,role::pv);

    // Boundaries derive from ORIGINAL codes, not the deliberately stale limbs.
    for (int32_t code : {127,128,255,256,65535}) {
        a = make_stream(1,64,std::vector<int32_t>(64,0),role::pv);
        a.codes[0] = code;
        v.assign(64,0); v[0] = 1;
        b = make_weight(64,1,v,role::pv,24);
        // 65535 can saturate its exact dot, but each unsigned bit is safe.
        pv_plane_reference("PV radix boundary "+std::to_string(code),a,b,true);
        const auto balanced_digits = golden::digits(code);
        int64_t balanced_code = 0, radix = 1;
        for (int8_t digit : balanced_digits) {
            balanced_code += int64_t(digit)*radix;
            radix *= 256;
        }
        golden::pv_planes planes;
        check("PV boundary independent balanced candidate and minimal unsigned planes",
              balanced_code == code && golden::pv_bitplanes(a,b,planes) &&
              planes.digits.size() == (code == 127 ? 7 : code <= 255 ? 8 : code == 256 ? 9 : 16));
    }
    a = make_stream(1,32,std::vector<int32_t>(32,0),role::pv);
    a.codes[0] = 128;
    v.assign(32,0); v[0] = 127;
    b = make_weight(32,1,v,role::pv,24);
    const auto candidate = golden::digits(128);
    const int64_t balanced = int64_t(golden::shifted(golden::wide{candidate[0]}*127,24,0)) +
                            256LL*golden::shifted(golden::wide{candidate[1]}*127,24,0);
    check("PV unrestricted balanced256 counterexample",
          balanced == 543313362944LL && int64_t(128)*127*(1LL<<24) == 272730423296LL);
    pv_plane_reference("PV unsigned bit7 avoids balanced256 saturation",a,b,true);

    a = make_stream(1,32,std::vector<int32_t>(32,0),role::pv);
    a.codes[0] = 1;
    v.assign(32,0); v[0] = -1;
    b = make_weight(32,1,v,role::pv,31);
    pv_plane_reference("PV exact negative SCU endpoint",a,b,true);
    b.limbs[0][0] = 1;
    pv_plane_reference("PV positive carrier31 cannot be certified",a,b,false);
    b.carriers[0] = 32;
    pv_plane_reference("PV nonzero carrier32 cannot be certified",a,b,false);
    a.codes[0] = 0;
    pv_plane_reference("PV zero contribution at carrier32",a,b,true);
    a.codes[0] = 128;
    b = make_weight(32,1,v,role::pv,24);

    // A mixed digit interval loses correlation with V and must be conservative.
    a.codes[0] = 1; a.codes[1] = 0;
    v[0] = 127; v[1] = -127; v[2] = 127;
    b = make_weight(32,1,v,role::pv,24);
    b.limbs[0][16] = 1; // Outside-INT32 bias; mixed signs in first physical microtile.
    pv_plane_reference("PV mixed-sign interval SCU refusal",a,b,false);
    a = make_stream(1,4*POTAL_HW_DIM,std::vector<int32_t>(4*POTAL_HW_DIM,1),role::pv);
    v.assign(4*POTAL_HW_DIM,0);
    v[0] = 127; v[POTAL_HW_DIM] = 127; v[2*POTAL_HW_DIM] = -127;
    b = make_weight(4*POTAL_HW_DIM,1,v,role::pv,24);
    pv_plane_reference("PV ascending K prefix ACC refusal",a,b,false);
    // Large carriers are safe only for proven-zero intervals.
    a.codes.assign(a.codes.size(),0);
    b.carriers.assign(b.carriers.size(),199);
    b.carriers[0] = 200;
    pv_plane_reference("PV large carriers proven zero",a,b,true);
    std::vector<golden::wide> exact;
    check("PV large carrier exact zero has no undefined shift",golden::pv_exact(a,b,exact) && exact[0] == 0);
    a.codes[0] = 1;
    pv_plane_reference("PV large nonzero carrier refuses",a,b,false);
    b.carriers.assign(b.carriers.size(),POTAL_CARRIER_ZERO);
    compare_reference("PV ZERO carriers retain biased path",a,b,role::pv);
    golden::pv_planes zero;
    check("PV ZERO raw plane outputs",golden::pv_bitplanes(a,b,zero) &&
          zero.certified && zero.integers == std::vector<int64_t>{0});

    // Exact cross-block cancellation outside INT128 remains representable.
    v.assign(64,0); v[0] = 1; v[32] = -2;
    b = make_weight(64,1,v,role::pv);
    b.carriers = {200,199};
    a = make_stream(1,64,std::vector<int32_t>(64,128),role::pv);
    check("PV independent wide cancellation beyond carrier width",
          golden::pv_exact(a,b,exact) && exact == std::vector<golden::wide>{0});
    golden::wide near_limit = 0;
    check("PV exact wide near-limit cancellation remains representable",
          golden::binary_sum({{127,1},{0,-1}},near_limit) &&
          near_limit == std::numeric_limits<golden::wide>::max());
    std::vector<int64_t> cancelled{778899};
    check("PV wide cross-block bias cancellation remains old",
          attention::reference(a,b,role::pv,cancelled) == POTAL_OK &&
          cancelled == std::vector<int64_t>{0});

    // Invalid code extent, range, padding, and stationary metadata publish nothing.
    a = make_stream(1,33,std::vector<int32_t>(33,1),role::pv);
    v.assign(33,0); v[0] = 64;
    b = make_weight(33,1,v,role::pv,24);
    auto refuse = [&](const stream & x, const stationary & w) {
        std::vector<int64_t> got{778899};
        golden::pv_planes invalid;
        check("PV invalid original codes/metadata reject atomically",
              !golden::pv_bitplanes(x,w,invalid) &&
              attention::reference(x,w,role::pv,got) != POTAL_OK &&
              got == std::vector<int64_t>{778899});
    };
    auto invalid = a; invalid.codes.pop_back(); refuse(invalid,b);
    invalid = a; invalid.codes[0] = -1; refuse(invalid,b);
    invalid = a; invalid.codes[0] = 65536; refuse(invalid,b);
    invalid = a; invalid.codes[33] = 1; refuse(invalid,b);
    auto bad_weight = b; bad_weight.carriers.pop_back(); refuse(a,bad_weight);
    bad_weight = b; bad_weight.limbs[0].pop_back(); refuse(a,bad_weight);
    bad_weight = b; bad_weight.base.clear(); refuse(a,bad_weight);
    bad_weight = b; bad_weight.carriers[0] = 0xffffffffu; refuse(a,bad_weight);
    invalid = a; invalid.kp = 32; refuse(invalid,b);
}

void pv_bitplane_execute(size_t k, size_t m, size_t n, size_t heads, bool reject = false) {
    graph g(role::pv,k,n,m,true,heads,1,1,1,GGML_TYPE_F32);
    g.w.fill(0); g.a.fill(0);
    for (size_t h = 0; h < heads; ++h) {
        for (size_t j = 0; j < n; ++j) {
            const float sign = j%2 ? -1.0f : 1.0f;
            g.w.set(0,j,sign,h);
            if (k > 32) g.w.set(32,j,sign*(h+1 == heads ? std::ldexp(1.0f,-24) : 0.5f),h);
            if (reject) g.w.set(1,j,sign,h);
        }
        for (size_t i = 0; i < m; ++i) {
            g.a.set(0,i,std::ldexp(1.0f,reject ? -15 : -16),h);
            // Moving the maximum into V's nonzero physical microtile creates an unsafe
            // unsigned-plane contribution, before any command may be issued.
            g.a.set(reject ? 1 : pv_peak,i,1.0f-std::ldexp(1.0f,reject ? -15 : -16),h);
        }
    }
    struct job {
        stream a;
        stationary b;
        size_t plane;
        bool bitplane;
        std::vector<int32_t> raw;
    };
    std::vector<job> jobs;
    std::vector<float> expected(heads*m*n);
    size_t selected_stripes = 0, selected_jobs = 0, old_stripes = 0;
    potal_tile_plan stripes;
    if (!check("PV fixture stripe selection",potal_tile_select(m,n,(k+31)/32*32,32/POTAL_HW_DIM,&stripes) == POTAL_OK)) return;
    for (size_t h = 0; h < heads; ++h) {
        stationary b;
        if (!check("PV real V preparation",attention::prepare_stationary(g.w.t,h,0,role::pv,b) == POTAL_OK)) return;
        for (size_t begin = 0; begin < m; begin += stripes.tile_I*POTAL_HW_DIM) {
            stream a;
            const size_t rows = std::min(m-begin,stripes.tile_I*POTAL_HW_DIM);
            if (!check("PV real P preparation",attention::prepare_stream(g.a.t,h,0,begin,rows,role::pv,a) == POTAL_OK)) return;
            std::vector<int32_t> bias;
            const auto bias_status = attention::column_bias(b,bias);
            const bool bitplane = bias_status == POTAL_ERR_UNSUPPORTED;
            check("PV extension trigger is only strict column bias",
                  bias_status == POTAL_OK || bitplane);
            std::vector<int64_t> integers;
            if (bitplane) {
                golden::pv_planes planes;
                if (!check("PV prepared unsigned planes",golden::pv_bitplanes(a,b,planes))) return;
                if (reject) {
                    check("PV rejected fixture requires interval refusal",!planes.certified);
                    continue;
                }
                std::vector<golden::wide> exact;
                if (!check("PV accepted fixture independent certificate",
                           planes.certified && golden::pv_exact(a,b,exact))) return;
                bool equal = exact.size() == planes.integers.size();
                for (size_t x = 0; x < exact.size(); ++x) equal = equal && exact[x] == planes.integers[x];
                check("PV accepted fixture exact wide integer equality",equal);
                integers = planes.integers;
                for (size_t p = 0; p < planes.raw.size(); ++p)
                    jobs.push_back({a,b,p,true,planes.raw[p]});
                ++selected_stripes;
                selected_jobs += planes.raw.size();
                if (k > 32) {
                    check("PV normalized guard fixture constants",
                          a.scale == -16 && a.codes[0] == 1 && a.codes[pv_peak] == 65535 &&
                          b.base[0] == -30 && b.carriers[0] == 24 &&
                          b.carriers[b.n] == 0 && b.limbs[0][0] == 64 &&
                          b.limbs[0][32*b.n] == 64);
                    bool normalized = true;
                    for (size_t i = 0; i < rows; ++i)
                        for (size_t j = 0; j < n; ++j) {
                            normalized = normalized &&
                                integers[i*n+j] == (j%2 ? -1LL : 1LL)*(1LL<<30);
                            for (size_t block = 2; block < b.kp/32; ++block)
                                normalized = normalized && b.carriers[block*b.n+j] == POTAL_CARRIER_ZERO;
                        }
                    check("PV normalized exact output is signed 2^-16",normalized);
                }
            } else {
                if (!check("PV old path independent arithmetic",golden::evaluate(a,b,role::pv,integers))) return;
                for (size_t p = 1; p < a.limb_count; ++p) jobs.push_back({a,b,p,false,{}});
                ++old_stripes;
            }
            for (size_t i = 0; i < rows; ++i)
                for (size_t j = 0; j < n; ++j)
                    expected[(h*m+begin+i)*n+j] = float(std::ldexp(double(integers[i*n+j]),a.scale+b.base[j]));
        }
    }
    potal_test::emulator emu;
    auto ops = emu.ops();
    potal_stripe_session session = {{&ops,0,{}}};
    potal_test::attention_emulator corrected;
    auto sink = corrected.sink();
    size_t ordinary_calls = 0, corrected_calls = 0;
    bool descriptors = true, raw_equal = true;
    const auto before = g.out.storage;
    const auto result = attention::execute(&g.out.t,
        [&](const potal_stripe_gemm_desc & d) {
            const size_t index = ordinary_calls++;
            if (index >= jobs.size()) {
                descriptors = false;
                return planned_result{};
            }
            const auto & job = jobs[index];
            const auto & a = job.a;
            const auto & b = job.b;
            descriptors = descriptors && !emu.in_flight && d.k == a.kp && d.m == a.rows && d.n == b.n &&
                          d.work_kind == (job.plane == 0 ? POTAL_STRIPE_WORK_MAIN : POTAL_STRIPE_WORK_RESIDUAL);
            potal_tile_plan plan;
            if (potal_tile_select(a.rows,b.n,a.kp,32/POTAL_HW_DIM,&plan) != POTAL_OK) descriptors = false;
            else {
                plan.tile_K = std::min(plan.tile_K,a.kp/POTAL_HW_DIM);
                if (job.bitplane)
                    plan.tile_K = std::min(plan.tile_K,size_t(POTAL_HW_BANK_ROWS)/
                                          (std::max(plan.tile_I,plan.tile_J)*POTAL_HW_DIM));
                while (plan.tile_K && (a.kp/POTAL_HW_DIM % plan.tile_K || plan.tile_K%(32/POTAL_HW_DIM))) --plan.tile_K;
                descriptors = descriptors && d.tile_i == plan.tile_I && d.tile_j == plan.tile_J &&
                              d.tile_k == plan.tile_K;
            }
            int64_t place = 1;
            for (size_t p = 0; p < job.plane; ++p) place *= 2;
            for (size_t i = 0; i < d.m; ++i)
                for (size_t x = 0; x < d.lda; ++x) {
                    const int digit = x >= a.k ? 0 : job.bitplane ?
                        int(a.codes[i*a.kp+x]/place%2) : a.limbs[job.plane][i*a.kp+x];
                    descriptors = descriptors && d.a[i*d.lda+x] == digit;
                }
            for (size_t x = 0; x < d.k; ++x)
                for (size_t j = 0; j < d.n; ++j)
                    descriptors = descriptors && d.b[x*d.ldb+j] == b.limbs[0][x*b.n+j];
            for (size_t block = 0; block < d.k/32; ++block)
                for (size_t j = 0; j < d.n; ++j)
                    descriptors = descriptors && d.carriers[block*d.carrier_stride+j] == b.carriers[block*b.n+j];
            const auto completed = run_planned_stripe_gemm(session,d);
            if (job.bitplane && completed.ok)
                for (size_t i = 0; i < d.m; ++i)
                    for (size_t j = 0; j < d.n; ++j)
                        raw_equal = raw_equal && d.c[i*d.ldc+j] == job.raw[i*d.n+j];
            return completed;
        },
        [&](const potal_stripe_gemm_desc & d, const int32_t * bias, size_t columns) {
            ++corrected_calls;
            return attention::run_corrected(d,bias,columns,&sink);
        });
    const std::string name = "PV H3-7C "+std::to_string(m)+"x"+std::to_string(n)+"x"+std::to_string(k)+
                             " heads="+std::to_string(heads);
    if (reject) {
        check(name+" refuses before any command or output",
              result.status == POTAL_ERR_UNSUPPORTED && !result.fatal && ordinary_calls == 0 &&
              corrected_calls == 0 && emu.commands.empty() && corrected.records.empty() &&
              g.out.storage == before && result.pv_bitplane_jobs == 0);
    } else {
        bool bits_equal = result.status == POTAL_OK;
        for (size_t h = 0; h < heads; ++h)
            for (size_t i = 0; i < m; ++i)
                for (size_t j = 0; j < n; ++j) {
                    const float got = g.out.get(j,i,h);
                    const float want = expected[(h*m+i)*n+j];
                    bits_equal = bits_equal && std::memcmp(&got,&want,sizeof(float)) == 0;
                }
        check(name+" bitwise F32 independent output",bits_equal);
        check(name+" raw completed command C and descriptors",raw_equal && descriptors && ordinary_calls == jobs.size());
        check(name+" selects only required stripes and completed planes",
              result.pv_bitplane_stripes == selected_stripes && result.pv_bitplane_jobs == selected_jobs &&
              corrected_calls == old_stripes && result.correction_jobs == old_stripes &&
              result.jobs == ordinary_calls+corrected_calls);
        if (heads == 12)
            check(name+" historical guard class preserves first eleven heads",
                  old_stripes == 11 && selected_stripes == 1 && selected_jobs == 16);
        if (old_stripes == 0)
            check(name+" no corrected runner or bias MVIN",corrected_calls == 0 && corrected.bias_mvins == 0);
    }
    check(name+" balanced scales and clean DMA",
          !emu.in_flight && !emu.scale_miss && emu.scales.empty() && emu.writes == emu.releases &&
          corrected.errors.empty() && !corrected.scale_miss && corrected.scale_writes == corrected.scale_releases);
}

void test_pv_bitplane_execute() {
    // Padded K32/64/96; partial M/N; multi-tile M/N; single-row Decode.
    pv_bitplane_execute(32,1,3,1);
    pv_bitplane_execute(33,3,19,1);
    pv_bitplane_execute(65,17,33,1);
    pv_bitplane_execute(96,33,65,1);
    pv_bitplane_execute(4097,1,3,1);
    pv_bitplane_execute(33,2,3,1,true);
    // Historical shape and guard class, NOT the unrecorded original operands.
    pv_bitplane_execute(288,1,64,12);

    for (bool fatal : {false,true}) {
        graph g(role::pv,64,1,1);
        g.w.fill(0); g.a.fill(0);
        g.w.set(0,0,1); g.w.set(32,0,std::ldexp(1.0f,-24));
        g.a.set(0,0,std::ldexp(1.0f,-16));
        g.a.set(pv_peak,0,1.0f-std::ldexp(1.0f,-16));
        const auto before = g.out.storage;
        potal_test::emulator emu;
        auto ops = emu.ops();
        potal_stripe_session session = {{&ops,0,{}}};
        size_t calls = 0;
        const auto result = attention::execute(&g.out.t,
            [&](const potal_stripe_gemm_desc & d) {
                if (++calls == 2) {
                    if (!fatal) {
                        planned_result failed;
                        failed.status = POTAL_ERR_BAD_RESPONSE;
                        failed.stage = "between_pv_bitplanes";
                        return failed;
                    }
                    emu.fail_fence_at = int(emu.fence_calls + 1);
                }
                return run_planned_stripe_gemm(session,d);
            },
            [](const potal_stripe_gemm_desc &, const int32_t *, size_t) { return planned_result{}; });
        check("PV later-plane failure preserves completed jobs and publication",
              calls == 2 && result.status != POTAL_OK && result.fatal == fatal &&
              result.pv_bitplane_stripes == 1 && result.pv_bitplane_jobs == 1 &&
              result.correction_jobs == 0 && result.bias_updates == 0 && g.out.storage == before);
        if (fatal) {
            check("PV fatal plane retains unknown DMA and Scale ownership",
                  result.status == POTAL_HW_BAD_COMMAND && result.stage &&
                  std::strcmp(result.stage,"fence") == 0 && emu.in_flight &&
                  !emu.scales.empty() && emu.writes > emu.releases && session.ws.report.fatal);
        } else {
            check("PV recoverable plane failure leaves completed DMA clean",
                  result.status == POTAL_ERR_BAD_RESPONSE && result.stage &&
                  std::strcmp(result.stage,"between_pv_bitplanes") == 0 &&
                  !emu.in_flight && emu.scales.empty() && emu.writes == emu.releases);
        }
    }
}

void test_contract() {
    graph g(role::qk,32,3,2,true);
    g.w.fill(1); g.a.fill(1);
    check("metadata role QK",attention::get_role(&g.out.t) == role::qk);
    ggml_gemmini_set_attention_role(&g.out.t,GGML_GEMMINI_ATTENTION_PV);
    check("metadata role PV",attention::get_role(&g.out.t) == role::pv);
    check("null role",attention::get_role(nullptr) == role::none);
    auto refuse = [&](const std::string & name) {
        check(name+" rejected",attention::check(&g.out.t,true) != GGML_POTAL_REJECT_NONE);
        const auto before = g.out.storage;
        size_t calls = 0;
        const auto result = attention::execute(&g.out.t,[&](const potal_stripe_gemm_desc &) {
            ++calls;
            return planned_result{};
        });
        check(name+" no command or partial output",result.status != POTAL_OK && calls == 0 && before == g.out.storage);
    };
    auto saved = g.out.t;
    g.out.t.op_params[1] = 0; refuse("missing role"); g.out.t = saved;
    g.out.t.op_params[1] = 7; refuse("unknown role"); g.out.t = saved;
    g.out.t.op_params[2] = 1; refuse("unsupported op_params"); g.out.t = saved;
    g.out.t.op = GGML_OP_ADD; refuse("unsupported operation"); g.out.t = saved;
    g.out.t.type = GGML_TYPE_F16; refuse("unsupported output type"); g.out.t = saved;
    g.out.t.src[0] = nullptr; refuse("missing source"); g.out.t = saved;
    const auto wsaved = g.w.t;
    const auto asaved = g.a.t;
    g.w.t.type = GGML_TYPE_I8; refuse("unsupported stationary type"); g.w.t = wsaved;
    g.a.t.type = GGML_TYPE_I32; refuse("unsupported stream type"); g.a.t = asaved;
    g.w.t.ne[0] = 31; refuse("mismatched K"); g.w.t = wsaved;
    g.a.t.ne[0] = 0; g.w.t.ne[0] = 0; refuse("zero K"); g.a.t = asaved; g.w.t = wsaved;
    g.a.t.ne[0] = -1; g.w.t.ne[0] = -1; refuse("negative K"); g.a.t = asaved; g.w.t = wsaved;
    g.a.t.ne[2] = 3; g.w.t.ne[2] = 2; refuse("nonintegral head repeat"); g.a.t = asaved; g.w.t = wsaved;
    g.a.t.ne[3] = 3; g.w.t.ne[3] = 2; refuse("nonintegral batch repeat"); g.a.t = asaved; g.w.t = wsaved;
    g.out.t.ne[0] = 4; refuse("wrong destination shape"); g.out.t = saved;
    g.out.t.ne[1] = 0; refuse("zero destination rows"); g.out.t = saved;
    g.w.t.ne[1] = 0; refuse("zero stationary columns"); g.w.t = wsaved;
    g.a.t.data = nullptr;
    check("metadata support check does not require data",attention::check(&g.out.t,false) == GGML_POTAL_REJECT_NONE);
    g.a.t = asaved;
    g.a.t.data = nullptr; refuse("unallocated source"); g.a.t = asaved;
    g.out.t.data = nullptr; refuse("unallocated destination"); g.out.t = saved;
    ggml_backend_buffer_type nonhost_type{};
    ggml_backend_buffer nonhost_buffer{};
    nonhost_type.iface.is_host = [](ggml_backend_buffer_type_t) { return false; };
    nonhost_buffer.buft = &nonhost_type;
    g.w.t.buffer = &nonhost_buffer; refuse("nonhost stationary memory"); g.w.t = wsaved;
    g.a.t.buffer = &nonhost_buffer; refuse("nonhost stream memory"); g.a.t = asaved;
    g.out.t.buffer = &nonhost_buffer; refuse("nonhost destination memory"); g.out.t = saved;
    // P validation is data-dependent and belongs before the first runner call.
    for (float bad : {-1.0f,std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN()}) {
        g.a.set(31,1,bad);
        const auto before = g.out.storage;
        size_t calls = 0;
        const auto result = attention::execute(&g.out.t,[&](const potal_stripe_gemm_desc &) {
            ++calls;
            return planned_result{};
        });
        check("invalid P rejected before runner",result.status != POTAL_OK && calls == 0 && g.out.storage == before);
        g.a.set(31,1,1);
    }
}

} // namespace

int main() {
    // Initialize GGML's F16 conversion tables without constructing a backend.
    const ggml_init_params params = {1u << 20,nullptr,true};
    ggml_context * context = ggml_init(params);
    if (!check("GGML initialization",context != nullptr)) return 1;
    if (!check("DIM16/DIM32 attention contract",POTAL_HW_DIM == 16 || POTAL_HW_DIM == 32)) {
        ggml_free(context);
        return 1;
    }
    test_reference();
    test_prepare();
    test_contract();
    test_execute();
    test_pv_execute();
    test_pv_bitplane_reference();
    test_pv_bitplane_execute();
    ggml_free(context);
    std::printf("POTAL_ATTENTION_TEST %s\n",failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}
