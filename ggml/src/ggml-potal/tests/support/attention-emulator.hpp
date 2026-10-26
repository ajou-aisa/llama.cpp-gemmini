#pragma once

#include "support/paired-h2-emulator.hpp"

namespace potal_test {

// Extend the existing command-level model only for full-width accumulator
// LOAD_CMD. Expected numerical results still come from attention-golden.hpp.
struct attention_emulator : h2_emulator {
    uint64_t bias_mvins = 0;
    uint64_t load_config = 0;

    potal_paired_sink sink() { return { &attention_emulator::on_cmd, this }; }

    static potal_status on_cmd(void * context, const potal_paired_cmd_record * record) {
        return static_cast<attention_emulator *>(context)->command(*record);
    }

    potal_status command(const potal_paired_cmd_record & record) {
        if (record.funct != POTAL_FUNCT_MVIN) {
            if (record.funct == POTAL_FUNCT_CONFIG && (record.rs1 & 3) == 1 &&
                ((record.rs1 >> 3) & 3) == 0) {
                load_config = record.rs1;
            }
            return h2_emulator::command(record);
        }
        records.push_back(record);
        if (record.funct == fail_funct) {
            fail_funct = 0xffffffffu;
            return fail_status;
        }
        const uint32_t rows = record.rs2 >> 48, columns = (record.rs2 >> 32) & 0xffff;
        const uint32_t local = uint32_t(record.rs2), row = local & 0x1fffffffu;
        const uint64_t expected_config = (uint64_t(D) << 16) | (uint64_t(1) << 8) | 1;
        if (record.xd || load_config != expected_config || ld_stride[0] != 0 ||
            (local & 0xe0000000u) != 0xc0000000u || !rows || rows > D ||
            !columns || columns > D || row + rows > acc_rows || record.rs1 == 0 ||
            record.rs1 % (D * sizeof(int32_t))) {
            err("full-width ACC bias MVIN fields");
            return POTAL_HW_BAD_COMMAND;
        }
        const auto * source = reinterpret_cast<const int32_t *>(uintptr_t(record.rs1));
        for (uint32_t i = 0; i < rows; ++i) {
            for (uint32_t j = 0; j < columns; ++j) {
                const int64_t sum = int64_t(acc[(size_t(row) + i) * D + j]) + source[j];
                acc[(size_t(row) + i) * D + j] =
                    sum > INT32_MAX ? INT32_MAX : sum < INT32_MIN ? INT32_MIN : int32_t(sum);
            }
        }
        ++bias_mvins;
        return POTAL_OK;
    }
};

} // namespace potal_test
