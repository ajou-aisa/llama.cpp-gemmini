#include "pb-attention.hpp"

#if defined(GGML_GEMMINI_PB)
#include "pb-exec.hpp"
#include "ggml-impl.h"
#include "../ggml-gemmini-args.h"
#include "../quants/act/block/block.hpp"

#include <cstring>
#include <new>
#include <stdexcept>
#include <vector>

namespace ggml::gemmini::pb {
bool attention_supported(const ggml_tensor * op) {
    const auto role = ggml_gemmini_get_attention_role(op);
    if ((role != GGML_GEMMINI_ATTENTION_QK && role != GGML_GEMMINI_ATTENTION_PV) ||
        op->type != GGML_TYPE_F32 || !ggml_is_contiguous(op) || !op->src[0] || !op->src[1]) return false;
    const auto & a = *op->src[0];
    const auto & b = *op->src[1];
    if ((a.type != GGML_TYPE_F16 && a.type != GGML_TYPE_F32) || b.type != GGML_TYPE_F32) return false;
    for (int d = 0; d < 4; ++d)
        if (a.ne[d] <= 0 || b.ne[d] <= 0 || a.nb[d] < ggml_type_size(a.type) ||
            b.nb[d] < sizeof(float)) return false;
    return a.ne[0] == b.ne[0] && a.ne[0] % 32 == 0 &&
           (role == GGML_GEMMINI_ATTENTION_QK ? a.ne[0] == 64 : a.ne[1] == 64) &&
           b.ne[2] % a.ne[2] == 0 && b.ne[3] % a.ne[3] == 0 &&
           op->ne[0] == a.ne[1] && op->ne[1] == b.ne[1] &&
           op->ne[2] == b.ne[2] && op->ne[3] == b.ne[3];
}

namespace {
std::vector<float> read_matrix(const ggml_tensor & source, size_t head, size_t batch) {
    const size_t k = source.ne[0], rows = source.ne[1];
    std::vector<float> values(rows * k);
    for (size_t row = 0; row < rows; ++row)
        for (size_t lane = 0; lane < k; ++lane) {
            const char * p = static_cast<const char *>(source.data) +
                batch * source.nb[3] + head * source.nb[2] + row * source.nb[1] + lane * source.nb[0];
            if (source.type == GGML_TYPE_F16) {
                ggml_fp16_t half;
                std::memcpy(&half, p, sizeof(half));
                values[row * k + lane] = ggml_fp16_to_fp32(half);
            } else {
                std::memcpy(&values[row * k + lane], p, sizeof(float));
            }
        }
    return values;
}

#if defined(GGML_GEMMINI_PB_Q8_0)
Status quantize(const ggml_tensor & source, size_t head, size_t batch,
                std::vector<int8_t> & codes, std::vector<float> & scales) {
    const auto values = read_matrix(source, head, batch);
    return quantize_q8_0_rows(values.data(), source.ne[1], source.ne[0], codes, scales);
}
#else
Status quantize(const ggml_tensor & source, size_t head, size_t batch,
                std::vector<int8_t> & codes, std::vector<float> & scales) {
    const size_t k = source.ne[0], rows = source.ne[1];
    auto values = read_matrix(source, head, batch);
    ggml_gemmini_args_t args{};
    args.I = rows; args.J = 1; args.K = k;
    if (!args.A.allocate(rows, k, 8)) return Status::invalid_arguments;
    args.act_quant.storage().emplace<quants::act::block::Meta>();
    ggml_tensor packed{};
    packed.type = GGML_TYPE_F32;
    packed.data = values.data();
    packed.ne[0] = k; packed.ne[1] = rows; packed.ne[2] = packed.ne[3] = 1;
    packed.nb[0] = sizeof(float); packed.nb[1] = k * sizeof(float);
    packed.nb[2] = packed.nb[3] = rows * packed.nb[1];
    if (!quants::act::block::quantize_mxint(&packed, args)) return Status::invalid_arguments;
    codes.resize(rows * k);
    for (size_t row = 0; row < rows; ++row)
        for (size_t lane = 0; lane < k; ++lane)
            codes[row * k + lane] = static_cast<int8_t>(args.A.get(row, lane));
    scales = std::get<quants::act::block::Meta>(args.act_quant.storage()).scales;
    return Status::success;
}
#endif
}

Status execute_attention(ggml_tensor * op) {
    const auto fail = [op](Status status) {
        GGML_LOG_ERROR("Gemmini-PB attention %s failed (status=%d)\n",
                       op ? op->name : "<null>", static_cast<int>(status));
        return status;
    };
    if (!attention_supported(op) || !op->data || !op->src[0]->data || !op->src[1]->data)
        return fail(Status::invalid_arguments);
    Session session;  // keeps the five-core lanes awake across the heads of this node
    try {
        const auto & a = *op->src[0];
        const auto & b = *op->src[1];
        const size_t k = a.ne[0], rows = b.ne[1], columns = a.ne[1], blocks = k / 32;
        const size_t head_repeat = b.ne[2] / a.ne[2], batch_repeat = b.ne[3] / a.ne[3];
        const auto type = ggml_gemmini_get_attention_role(op) == GGML_GEMMINI_ATTENTION_QK ?
                          GGML_GEMMINI_PB_QK : GGML_GEMMINI_PB_PV;
        std::vector<float> staged(size_t(ggml_nelements(op)));
        std::vector<int8_t> activation, weight;
        std::vector<float> sa, sb, weight_scales(blocks * columns);
        for (size_t batch = 0; batch < size_t(b.ne[3]); ++batch)
            for (size_t head = 0; head < size_t(b.ne[2]); ++head) {
                Status quantized = quantize(b, head, batch, activation, sa);
                if (quantized == Status::success)
                    quantized = quantize(a, head / head_repeat, batch / batch_repeat, weight, sb);
                if (quantized != Status::success) return fail(quantized);
                for (size_t block = 0; block < blocks; ++block)
                    for (size_t col = 0; col < columns; ++col)
                        weight_scales[block * columns + col] = sb[col * blocks + block];
                const auto status = execute(activation.data(), sa.data(), weight.data(), weight_scales.data(),
                                            rows, columns, k,
                                            staged.data() + (batch * b.ne[2] + head) * rows * columns,
                                            columns, 1, type);
                if (status != Status::success) return fail(status);
            }
        std::memcpy(op->data, staged.data(), staged.size() * sizeof(float));
        return Status::success;
    } catch (const std::bad_alloc &) {
        return fail(Status::allocation_failure);
    } catch (const std::length_error &) {
        return fail(Status::allocation_failure);
    }
}
}
#endif
