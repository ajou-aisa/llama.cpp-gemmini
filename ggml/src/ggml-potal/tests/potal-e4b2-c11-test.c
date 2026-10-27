#include "potal.h"
#include <string.h>

int potal_e4b2_c11_header_test(void);

int potal_e4b2_c11_header_test(void) {
    const potal_resident_ws_view view = {1, 0, 0, 0, 1, 4672, 144, 0, 80};
    potal_rocc_cmd command = {UINT64_MAX, UINT64_MAX};
    if (potal_resident_ws_view_encode(&view, &command) != POTAL_OK ||
        command.rs1 != (UINT64_C(80) << 48 | UINT64_C(144) << 16 | UINT64_C(4672)) ||
        command.rs2 != UINT64_C(0x11)) return 1;
    const potal_rocc_cmd saved = command;
    potal_resident_ws_view bad = view;
    bad.acc_base = 81;
    if (potal_resident_ws_view_encode(&bad, &command) != POTAL_ERR_INVALID_ARGUMENT ||
        memcmp(&saved, &command, sizeof(saved))) return 2;
    bad = view;
    bad.indexed = 1;
    bad.b_base = 4160;
    bad.indexed_j_origin = 5;
    bad.acc_bank = 1;
    if (potal_resident_ws_view_encode(&bad, &command) != POTAL_OK ||
        command.rs1 != (UINT64_C(80) << 48 | UINT64_C(5) << 32 |
                        UINT64_C(144) << 16 | UINT64_C(4160)) ||
        command.rs2 != UINT64_C(0x1d)) return 3;
    const potal_resident_ws_view disabled = {0};
    if (potal_resident_ws_view_encode(&disabled, &command) != POTAL_OK ||
        command.rs1 || command.rs2) return 4;
    uint32_t carrier = 123;
    if (potal_resident_ws_shift_carrier(-1, &carrier) != POTAL_ERR_INVALID_ARGUMENT ||
        potal_resident_ws_shift_carrier(INT64_MIN, &carrier) != POTAL_ERR_INVALID_ARGUMENT ||
        carrier != 123) return 5;
    if (potal_resident_ws_shift_carrier(0, &carrier) != POTAL_OK || carrier != 0 ||
        potal_resident_ws_shift_carrier(32767, &carrier) != POTAL_OK || carrier != 32767 ||
        potal_resident_ws_shift_carrier(32768, &carrier) != POTAL_ERR_INVALID_ARGUMENT) return 6;
    return 0;
}
