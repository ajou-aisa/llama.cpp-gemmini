// Replays the golden cases (tests/golden/*.golden, captured from the legacy
// SDK tiler by legacy-capture.cpp) through matmul/tiling + the recording
// transport and requires the same tiles, command count, stream hash and
// leading commands, for every captured geometry.
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "golden-format.hpp"
#include "tiling.hpp"

namespace {

int failures = 0;

void fail(const std::string &what) {
    std::fprintf(stderr, "FAIL %s\n", what.c_str());
    ++failures;
}

struct Expected {
    size_t                   tile_i = 0, tile_j = 0, tile_k = 0;
    size_t                   commands = 0;
    std::string              hash;
    std::vector<std::string> head;
};

struct Golden {
    gemmini_hw_config_t             cfg{};
    std::map<std::string, Expected> cases;
};

bool load(const std::string &path, Golden &g) {
    std::ifstream in(path);
    if (!in) {
        return false;
    }
    std::string line, current;
    while (std::getline(in, line)) {
        unsigned d, banks, rows, acc, max_bytes;
        char     name[128];
        if (std::sscanf(line.c_str(), "GEOMETRY DIM=%u BANK_NUM=%u BANK_ROWS=%u ACC_ROWS=%u MAX_BYTES=%u", &d, &banks,
                        &rows, &acc, &max_bytes) == 5) {
            g.cfg = { d, 8, 1, 4, banks, rows, 2, acc, max_bytes };
        } else if (std::sscanf(line.c_str(), "CASE %127s", name) == 1) {
            current = name;
        } else if (line.rfind("SET_TILE_WS ", 0) == 0) {
            auto &e = g.cases[current];
            std::sscanf(line.c_str(), "SET_TILE_WS %zu %zu %zu", &e.tile_i, &e.tile_j, &e.tile_k);
        } else if (line.rfind("WS_COMMANDS ", 0) == 0) {
            auto &e = g.cases[current];
            char  hash[32];
            std::sscanf(line.c_str(), "WS_COMMANDS %zu HASH %31s", &e.commands, hash);
            e.hash = hash;
        } else if (line.rfind("  ", 0) == 0) {
            g.cases[current].head.push_back(line.substr(2));
        }
    }
    return !g.cases.empty() && g.cfg.dim != 0;
}

const uint32_t kOne = 0x3f800000u; // MVIN_SCALE_IDENTITY / ACC_SCALE_IDENTITY (1.0f)

void replay(const std::string &file, const Golden &g, const golden::Case &c) {
    const std::string tag = file + ":" + c.name;
    auto              it  = g.cases.find(c.name);
    if (it == g.cases.end()) {
        fail(tag + " missing from golden file");
        return;
    }
    const Expected &e = it->second;

    const auto tiles_ws = ggml::gemmini::tiling::select_ws_tiles(g.cfg, c.I, c.J, c.K, c.act);
    if (tiles_ws.i != e.tile_i || tiles_ws.j != e.tile_j || tiles_ws.k != e.tile_k) {
        fail(tag + " gemmini_set_tile_ws tiles differ");
    }

    const size_t rows_a = c.a_transpose ? c.K : c.I, cols_a = c.a_transpose ? c.I : c.K;
    const size_t rows_b = c.b_transpose ? c.J : c.K, cols_b = c.b_transpose ? c.K : c.J;
    std::vector<int8_t>  A(rows_a * cols_a), B(rows_b * cols_b);
    std::vector<int32_t> D32(c.repeating_bias ? c.J : c.I * c.J);
    std::vector<int8_t>  D8(D32.size());
    std::vector<int32_t> C32(c.I * c.J);
    std::vector<int8_t>  C8(c.I * c.J);
    const void *D = !c.bias ? nullptr : (c.low_d ? (const void *) D8.data() : (const void *) D32.data());
    void       *C = c.full_c ? (void *) C32.data() : (void *) C8.data();
    const std::vector<golden::Buffer> buffers = {
        { "A", (uint64_t) A.data(), A.size() },
        { "B", (uint64_t) B.data(), B.size() },
        { "D", (uint64_t) D, D ? (c.low_d ? D8.size() : D32.size() * 4) : 0 },
        { "C", (uint64_t) C, c.full_c ? C32.size() * 4 : C8.size() },
    };

    ggml::gemmini::tiling::Matmul mm;
    mm.dim_i = c.I;
    mm.dim_j = c.J;
    mm.dim_k = c.K;
    mm.a = (uint64_t) A.data();
    mm.b = (uint64_t) B.data();
    mm.d = (uint64_t) D;
    mm.c = (uint64_t) C;
    mm.stride_a = cols_a;
    mm.stride_b = cols_b;
    mm.stride_d = c.J;
    mm.stride_c = c.J;
    mm.a_scale_bits = mm.b_scale_bits = mm.d_scale_bits = kOne;
    mm.act = c.act;
    mm.acc_scale_bits = kOne;
    mm.config_ex_acc_scale_bits = kOne;
    mm.repeating_bias = c.repeating_bias;
    mm.a_transpose = c.a_transpose;
    mm.b_transpose = c.b_transpose;
    mm.full_c = c.full_c;
    mm.low_d = c.low_d;

    std::vector<gemmini_record_t> records(1u << 16);
    gemmini_recorder_t            rec;
    gemmini_recorder_init(&rec, records.data(), records.size());
    const auto tiles = ggml::gemmini::tiling::select_tiles(g.cfg, c.I, c.J, c.K, c.act, true);
    const gemmini_status_t status = ggml::gemmini::tiling::run_ws(&rec.transport, g.cfg, mm, tiles);
    if (status != GEMMINI_OK) {
        fail(tag + " run_ws: " + gemmini_status_name(status));
        return;
    }
    uint64_t                 hash = golden::fnv1a(nullptr, 0);
    std::vector<std::string> lines;
    for (size_t i = 0; i < rec.count; ++i) {
        const gemmini_record_t &r = records[i];
        lines.push_back(r.kind == GEMMINI_RECORD_DRAIN
                            ? std::string("FENCE")
                            : golden::command_line(buffers, r.command.funct, r.command.rs1, r.command.rs2));
        hash = golden::fnv1a(lines.back().data(), lines.back().size(), hash);
        hash = golden::fnv1a("\n", 1, hash);
    }
    char hash_text[32];
    std::snprintf(hash_text, sizeof(hash_text), "%016llx", (unsigned long long) hash);
    if (lines.size() != e.commands) {
        fail(tag + " command count " + std::to_string(lines.size()) + " != " + std::to_string(e.commands));
    }
    for (size_t i = 0; i < e.head.size() && i < lines.size(); ++i) {
        if (lines[i] != e.head[i]) {
            fail(tag + " command " + std::to_string(i) + ": " + lines[i] + " != " + e.head[i]);
            break;
        }
    }
    if (e.hash != hash_text) {
        fail(tag + " stream hash " + hash_text + " != " + e.hash);
    }
}

void test_invalid(const gemmini_hw_config_t &cfg) {
    ggml::gemmini::tiling::Matmul mm;
    mm.dim_i = mm.dim_j = mm.dim_k = 64;
    mm.a = 0x1000;
    mm.b = 0x2000;
    mm.c = 0x3000;
    mm.stride_a = mm.stride_b = mm.stride_c = 64;
    std::vector<gemmini_record_t> records(64);
    gemmini_recorder_t            rec;
    gemmini_recorder_init(&rec, records.data(), records.size());
    const char *message = nullptr;

    if (ggml::gemmini::tiling::check_ws(cfg, mm, { 0, 1, 1 }, &message) != GEMMINI_ERR_INVALID_ARGUMENT ||
        std::strcmp(message, "tile_I is non-positive\n") != 0) {
        fail("zero tile rejected with the SDK message");
    }
    if (ggml::gemmini::tiling::check_ws(cfg, mm, { 8, 8, 8 }, &message) != GEMMINI_ERR_INVALID_ARGUMENT) {
        fail("tile larger than the padded matrix rejected");
    }
    ggml::gemmini::tiling::Matmul big = mm;
    big.dim_i = big.dim_j = big.dim_k = 4096;
    if (ggml::gemmini::tiling::check_ws(cfg, big, { 1, 2, 4096 / cfg.dim }, &message) != GEMMINI_ERR_CAPACITY) {
        fail("SPAD capacity boundary rejected");
    }
    if (ggml::gemmini::tiling::check_ws(cfg, big, { 8, 8, 1 }, &message) != GEMMINI_ERR_CAPACITY) {
        fail("ACC capacity boundary rejected");
    }
    mm.a_transpose = mm.b_transpose = true;
    if (ggml::gemmini::tiling::run_ws(&rec.transport, cfg, mm, { 1, 1, 1 }) != GEMMINI_ERR_UNSUPPORTED || rec.count != 0) {
        fail("double transpose rejected before any command");
    }
    mm.a_transpose = mm.b_transpose = false;
    mm.act = GEMMINI_ACT_LAYERNORM;
    if (ggml::gemmini::tiling::run_ws(&rec.transport, cfg, mm, { 1, 1, 1 }) != GEMMINI_ERR_UNSUPPORTED || rec.count != 0) {
        fail("unported CONFIG_NORM activation rejected before any command");
    }
    mm.act = GEMMINI_ACT_NONE;
    rec.fail_seq = 2;
    rec.fail_status = GEMMINI_ERR_TRANSPORT;
    if (ggml::gemmini::tiling::run_ws(&rec.transport, cfg, mm, { 1, 1, 1 }) != GEMMINI_ERR_TRANSPORT || rec.count != 3) {
        fail("transport error stops the stream");
    }
}

} // namespace

int main(int argc, char **argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s <golden dir>\n", argv[0]);
        return 2;
    }
    const char *files[] = { "gemmini-d16-4x4096-acc1024.golden", "gemmini-d16-4x2048-acc512.golden",
                            "gemmini-d32-4x2048-acc512.golden", "gemmini-d64-4x1024-acc256.golden" };
    for (const char *file : files) {
        Golden g;
        if (!load(std::string(argv[1]) + "/" + file, g)) {
            fail(std::string("cannot read ") + file);
            continue;
        }
        for (const auto &c : golden::cases()) {
            replay(file, g, c);
        }
        test_invalid(g.cfg);
    }
    if (failures) {
        std::fprintf(stderr, "gemmini-tiling-test: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("GEMMINI_TILING_TEST PASS\n");
    return 0;
}
