// The window calls (potal_paired_scale_write_columns, potal_paired_scale_release_columns)
// against the per-column calls they replace: the same raw commands, return status,
// column count and session state, on the success path and on every error path.
//
// Each side of a scenario runs in its own forked child, because a failed session
// with live entries keeps the accelerator owned for the rest of its process.
//
// Output: "PASS <name>" / "FAIL <name>", final "POTAL_SCALE_WINDOW_TEST PASS|FAIL".
#include "potal.h"
#include "potal_hw_params.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

namespace {

constexpr size_t MAX_RECORDS = 20000;
constexpr uint32_t DIM = POTAL_HW_DIM;
constexpr uint32_t BAD_CARRIER = 0x7FFFFFFFu;

struct outcome {
    int      ran;
    int      write_status, release_status, cleanup_status, close_status;
    uint64_t write_done, release_done;
    potal_paired_session_status after_issue, after_cleanup;
    char     stage[32];
    size_t   records;
    potal_paired_cmd_record record[MAX_RECORDS];
};

struct recorder {
    outcome *out;
    size_t   fail_at;
    potal_status fail_status;
};

potal_status record_command(void *ctx, const potal_paired_cmd_record *r) {
    auto &self = *static_cast<recorder *>(ctx);
    const size_t index = self.out->records;
    if (index < MAX_RECORDS) self.out->record[index] = *r;
    ++self.out->records;
    return index == self.fail_at ? self.fail_status : POTAL_OK;
}

struct live_column {
    uint32_t slot, generation, column;
};

struct scenario {
    std::string name;
    potal_paired_scale_columns write{};
    potal_paired_scale_columns release{};
    uint32_t first_column = 0;
    bool do_write = true, do_release = true;
    bool no_sink = false, null_session = false, scale_cpu = false;
    size_t fail_at = SIZE_MAX;
    potal_status fail_status = POTAL_HW_INVALID_ADDRESS;
    size_t bad_carrier = SIZE_MAX; // ordinal of the column whose carrier is invalid
    std::vector<live_column> live;
    std::vector<potal_paired_scale_columns> earlier; // complete windows written and released first
};

std::vector<uint32_t> carriers_for(const potal_paired_scale_columns &w, size_t bad_carrier) {
    const size_t blocks = std::max<uint32_t>(w.blocks, 1), row = size_t(std::max<uint32_t>(w.j_tiles, 1)) * DIM + 64;
    std::vector<uint32_t> values(blocks * row);
    for (size_t i = 0; i < values.size(); ++i) values[i] = i % 7 == 0 ? POTAL_CARRIER_ZERO : uint32_t(i % (POTAL_CARRIER_MAX_SHIFT + 1));
    size_t ordinal = 0;
    for (uint32_t b = 0; b < w.blocks; ++b) {
        for (uint32_t j = 0; j < w.j_tiles; ++j) {
            const uint32_t columns = j + 1 == w.j_tiles ? w.last_columns : DIM;
            for (uint32_t c = 0; c < columns && b * row + j * DIM + c < values.size(); ++c, ++ordinal) {
                if (ordinal == bad_carrier) values[b * row + j * DIM + c] = BAD_CARRIER;
            }
        }
    }
    return values;
}

// As the executor issues them before the window calls existed.
potal_status per_column_write(potal_paired_session *s, const potal_paired_scale_columns &w, const uint32_t *carriers,
                              size_t row, uint64_t &done) {
    done = 0;
    for (uint32_t b = 0; b < w.blocks; ++b) {
        for (uint32_t j = 0; j < w.j_tiles; ++j) {
            const uint32_t columns = j + 1 == w.j_tiles ? w.last_columns : DIM;
            for (uint32_t c = 0; c < columns; ++c) {
                const potal_status st = potal_paired_scale_write(s, w.slot_base + b * w.j_tiles + j, w.generation, c,
                                                                 carriers[b * row + j * DIM + c]);
                if (st != POTAL_OK) return st;
                ++done;
            }
        }
    }
    return POTAL_OK;
}

potal_status per_column_release(potal_paired_session *s, const potal_paired_scale_columns &w, uint32_t first_column,
                                uint64_t &done) {
    done = 0;
    for (uint32_t b = 0; b < w.blocks; ++b) {
        for (uint32_t j = 0; j < w.j_tiles; ++j) {
            const uint32_t columns = j + 1 == w.j_tiles ? w.last_columns : DIM;
            for (uint32_t c = b == 0 && j == 0 ? first_column : 0; c < columns; ++c) {
                const potal_status st = potal_paired_scale_release(s, w.slot_base + b * w.j_tiles + j, w.generation, c);
                if (st != POTAL_OK) return st;
                ++done;
            }
        }
    }
    return POTAL_OK;
}

void run_side(const scenario &sc, bool window, outcome *out) {
    if (sc.scale_cpu) setenv("POTAL_SCALE_CPU", "1", 1);
    recorder rec{out, sc.fail_at, sc.fail_status};
    const potal_paired_sink sink{record_command, &rec};
    potal_paired_session *s = nullptr;
    if (!sc.null_session && potal_paired_session_open(sc.no_sink ? nullptr : &sink, &s) != POTAL_OK) return;
    out->ran = 1;

    for (const live_column &l : sc.live) {
        if (potal_paired_scale_write(s, l.slot, l.generation, l.column, 0) != POTAL_OK) out->ran = 0;
    }
    for (const auto &w : sc.earlier) {
        const std::vector<uint32_t> values = carriers_for(w, SIZE_MAX);
        const size_t row = size_t(w.j_tiles) * DIM + 64;
        uint64_t done = 0;
        if (window) {
            std::vector<const uint32_t *> rows(w.blocks);
            for (uint32_t b = 0; b < w.blocks; ++b) rows[b] = values.data() + b * row;
            if (potal_paired_scale_write_columns(s, &w, rows.data(), &done) != POTAL_OK ||
                potal_paired_scale_release_columns(s, &w, 0, &done) != POTAL_OK) out->ran = 0;
        } else if (per_column_write(s, w, values.data(), row, done) != POTAL_OK ||
                   per_column_release(s, w, 0, done) != POTAL_OK) {
            out->ran = 0;
        }
    }

    const potal_paired_scale_columns &w = sc.write;
    const std::vector<uint32_t> values = carriers_for(w, sc.bad_carrier);
    const size_t row = size_t(std::max<uint32_t>(w.j_tiles, 1)) * DIM + 64;
    if (sc.do_write) {
        if (window) {
            std::vector<const uint32_t *> rows(std::max<uint32_t>(w.blocks, 1));
            for (size_t b = 0; b < rows.size(); ++b) rows[b] = values.data() + b * row;
            out->write_status = potal_paired_scale_write_columns(s, &w, rows.data(), &out->write_done);
        } else {
            out->write_status = per_column_write(s, w, values.data(), row, out->write_done);
        }
    }
    if (sc.do_release) {
        out->release_status = window ? potal_paired_scale_release_columns(s, &sc.release, sc.first_column, &out->release_done)
                                     : per_column_release(s, sc.release, sc.first_column, out->release_done);
    }
    if (s == nullptr) return;
    out->after_issue = potal_paired_session_query(s);
    if (out->after_issue.error_stage != nullptr) {
        std::strncpy(out->stage, out->after_issue.error_stage, sizeof(out->stage) - 1);
        out->after_issue.error_stage = nullptr;
    }
    out->cleanup_status = potal_paired_session_cleanup(s);
    out->after_cleanup = potal_paired_session_query(s);
    out->after_cleanup.error_stage = nullptr;
    out->close_status = potal_paired_session_close(s);
}

outcome *shared_outcome() {
    void *p = mmap(nullptr, sizeof(outcome), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) {
        std::perror("mmap");
        std::exit(2);
    }
    return static_cast<outcome *>(p);
}

bool fork_side(const scenario &sc, bool window, outcome *out) {
    std::fflush(nullptr);
    const pid_t child = fork();
    if (child == 0) {
        std::memset(out, 0, sizeof(*out));
        run_side(sc, window, out);
        std::fflush(nullptr);
        _exit(0);
    }
    int status = 0;
    return child > 0 && waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0;
}

uint64_t stream_hash(const outcome &o) {
    uint64_t h = 1469598103934665603ull;
    const auto mix = [&](uint64_t v) {
        for (int i = 0; i < 8; ++i) h = (h ^ ((v >> (8 * i)) & 255)) * 1099511628211ull;
    };
    for (size_t i = 0; i < std::min(o.records, MAX_RECORDS); ++i) {
        mix(o.record[i].funct);
        mix(o.record[i].xd);
        mix(o.record[i].rs1);
        mix(o.record[i].rs2);
    }
    return h;
}

bool same_status(const potal_paired_session_status &a, const potal_paired_session_status &b) {
    return a.failed == b.failed && a.error == b.error && a.outstanding_loops == b.outstanding_loops &&
           a.outstanding_readbacks == b.outstanding_readbacks && a.outstanding_acc_updates == b.outstanding_acc_updates &&
           a.index_transaction_open == b.index_transaction_open && a.live_scale_slots == b.live_scale_slots &&
           a.commands == b.commands && a.drains == b.drains;
}

int failures = 0;

void check(const std::string &name, bool ok, const std::string &detail = "") {
    std::printf("%s %s%s%s\n", ok ? "PASS" : "FAIL", name.c_str(), detail.empty() ? "" : " ", detail.c_str());
    failures += ok ? 0 : 1;
}

uint64_t all_scenarios_hash = 1469598103934665603ull;

outcome *off_side, *on_side;

void compare(const scenario &sc) {
    const bool ran = fork_side(sc, false, off_side) && fork_side(sc, true, on_side);
    const outcome &a = *off_side, &b = *on_side;
    bool same = ran && a.ran == b.ran && a.write_status == b.write_status && a.release_status == b.release_status &&
                a.cleanup_status == b.cleanup_status && a.close_status == b.close_status && a.write_done == b.write_done &&
                a.release_done == b.release_done && a.records == b.records && std::strcmp(a.stage, b.stage) == 0 &&
                same_status(a.after_issue, b.after_issue) && same_status(a.after_cleanup, b.after_cleanup);
    for (size_t i = 0; same && i < std::min(a.records, MAX_RECORDS); ++i) {
        same = a.record[i].funct == b.record[i].funct && a.record[i].xd == b.record[i].xd &&
               a.record[i].rs1 == b.record[i].rs1 && a.record[i].rs2 == b.record[i].rs2;
    }
    char detail[256];
    std::snprintf(detail, sizeof(detail),
                  "commands=%zu write=%d/%" PRIu64 " release=%d/%" PRIu64 " cleanup=%d close=%d stage=%s hash=%016" PRIx64,
                  b.records, b.write_status, b.write_done, b.release_status, b.release_done, b.cleanup_status,
                  b.close_status, b.stage[0] ? b.stage : "-", stream_hash(b));
    check(sc.name, same && a.records <= MAX_RECORDS, detail);
    all_scenarios_hash = (all_scenarios_hash ^ stream_hash(b)) * 1099511628211ull;
}

scenario base(const std::string &name, uint32_t slot_base, uint32_t blocks, uint32_t j_tiles, uint32_t last_columns,
              uint32_t generation) {
    scenario s;
    s.name = name;
    s.write = {slot_base, blocks, j_tiles, last_columns, generation};
    s.release = s.write;
    return s;
}

// Literals from PoTalScaleCommandIngress.scala: rs1 [7:0] slot, [15:8] generation, [21:16] column.
void test_encoding_literal() {
    scenario sc = base("encoding", 3, 1, 1, 6, 7);
    if (!fork_side(sc, true, on_side)) {
        check("encoding literal", false);
        return;
    }
    const outcome &o = *on_side;
    check("encoding literal, Scale WRITE column 5 of slot 3 generation 7",
          o.records == 12 && o.record[5].funct == POTAL_FUNCT_SCALE_WRITE && o.record[5].xd == 1 &&
          o.record[5].rs1 == 0x050703u && o.record[5].rs2 == 5u);
    check("encoding literal, Scale RELEASE column 5 of slot 3 generation 7",
          o.records == 12 && o.record[11].funct == POTAL_FUNCT_SCALE_RELEASE && o.record[11].xd == 1 &&
          o.record[11].rs1 == 0x050703u && o.record[11].rs2 == 0u);
    check("column counts and drains", o.write_done == 6 && o.release_done == 6 && o.after_issue.drains == 6 &&
          o.after_issue.live_scale_slots == 0 && o.close_status == POTAL_OK);
}

void test_argument_refusals() {
    scenario sc = base("arguments", 0, 1, 1, 1, 1);
    potal_paired_session *s = nullptr;
    outcome *out = on_side;
    std::fflush(nullptr);
    const pid_t child = fork();
    if (child == 0) {
        recorder rec{out, SIZE_MAX, POTAL_OK};
        const potal_paired_sink sink{record_command, &rec};
        int ok = potal_paired_session_open(&sink, &s) == POTAL_OK;
        uint64_t done = 99;
        const uint32_t value = 0;
        const uint32_t *row = &value;
        ok = ok && potal_paired_scale_write_columns(s, nullptr, &row, &done) == POTAL_ERR_INVALID_ARGUMENT;
        ok = ok && potal_paired_scale_write_columns(s, &sc.write, nullptr, &done) == POTAL_ERR_INVALID_ARGUMENT;
        ok = ok && potal_paired_scale_write_columns(s, &sc.write, &row, nullptr) == POTAL_ERR_INVALID_ARGUMENT;
        ok = ok && potal_paired_scale_release_columns(s, nullptr, 0, &done) == POTAL_ERR_INVALID_ARGUMENT;
        ok = ok && potal_paired_scale_release_columns(s, &sc.write, 0, nullptr) == POTAL_ERR_INVALID_ARGUMENT;
        _exit(ok ? 0 : 1);
    }
    int status = 0;
    check("null window, carriers and count refused", child > 0 && waitpid(child, &status, 0) == child && WIFEXITED(status) &&
          WEXITSTATUS(status) == 0);
}

} // namespace

int main() {
    off_side = shared_outcome();
    on_side = shared_outcome();
    test_encoding_literal();
    test_argument_refusals();

    const uint32_t full = 4 * 2 * DIM;
    compare(base("full window", 0, 4, 2, DIM, 7));
    compare(base("padded last J tile", 8, 3, 3, 7, 12));
    compare(base("one column", 0, 1, 1, 1, 1));
    compare(base("last slot, last generation", POTAL_SCALE_ENTRIES - 1, 1, 1, DIM, POTAL_GENERATION_MAX));
    compare(base("generation 0", 5, 1, 2, DIM, 0));
    compare(base("all slots", 0, POTAL_SCALE_ENTRIES / 4, 4, DIM, 3));
    {
        scenario s = base("first release issued by the caller", 0, 2, 2, DIM, 9);
        s.first_column = 1;
        compare(s);
        scenario t = base("first column skipped in a one-column first slot", 0, 2, 1, 1, 9);
        t.first_column = 1;
        compare(t);
    }

    compare(base("generation above the maximum", 0, 1, 1, DIM, POTAL_GENERATION_MAX + 1));
    compare(base("window crosses the last slot", POTAL_SCALE_ENTRIES - 4, 2, 4, DIM, 5));
    compare(base("first slot out of range", POTAL_SCALE_ENTRIES, 1, 1, DIM, 5));
    compare(base("too many columns", 0, 1, 2, DIM + 1, 5));
    compare(base("empty last J tile", 0, 2, 2, 0, 5));
    compare(base("no blocks", 0, 0, 2, DIM, 5));
    compare(base("no J tiles", 0, 2, 0, DIM, 5));

    for (size_t bad : {size_t(0), size_t(5), size_t(DIM), size_t(full - 1)}) {
        scenario s = base("invalid carrier at column " + std::to_string(bad), 0, 4, 2, DIM, 7);
        s.bad_carrier = bad;
        compare(s);
    }
    for (size_t at : {size_t(0), size_t(1), size_t(DIM), size_t(full / 2), size_t(full - 1)}) {
        scenario s = base("write response fails at column " + std::to_string(at), 0, 4, 2, DIM, 7);
        s.fail_at = at;
        compare(s);
        s.name = "write response is lost at column " + std::to_string(at);
        s.fail_status = POTAL_ERR_BAD_RESPONSE;
        compare(s);
    }
    for (size_t at : {size_t(0), size_t(1), size_t(DIM), size_t(full / 2), size_t(full - 1)}) {
        scenario s = base("release response fails at column " + std::to_string(at), 0, 4, 2, DIM, 7);
        s.fail_at = full + at;
        compare(s);
        s.name = "release response is lost at column " + std::to_string(at);
        s.fail_status = POTAL_ERR_BAD_RESPONSE;
        compare(s);
    }

    {
        scenario s = base("live column, same generation", 0, 2, 2, DIM, 7);
        s.live = {{3, 7, 9}};
        s.do_release = false;
        compare(s);
        s.name = "live column, other generation";
        s.live = {{3, 8, 9}};
        compare(s);
        s.name = "live column in the first slot";
        s.live = {{0, 7, 0}};
        compare(s);
        scenario t = base("live column outside the columns written", 0, 1, 1, 16, 7);
        t.live = {{0, 7, 20}};
        compare(t);
    }
    {
        scenario s = base("overlaps a window still live", 4, 2, 2, DIM, 7);
        s.earlier = {};
        s.live = {{6, 7, 0}, {6, 7, 1}};
        compare(s);
    }
    {
        scenario s = base("slots reused after release, next generation", 0, 2, 2, DIM, 8);
        s.earlier = {{0, 2, 2, DIM, 7}};
        compare(s);
        s.name = "slots reused across the generation wrap";
        s.write.generation = s.release.generation = 1;
        s.earlier = {{0, 2, 2, DIM, POTAL_GENERATION_MAX}};
        compare(s);
    }
    {
        scenario s = base("release without a write", 0, 2, 2, DIM, 7);
        s.do_write = false;
        compare(s);
        scenario t = base("release wider than the write", 0, 1, 2, DIM, 7);
        t.release = {0, 3, 2, DIM, 7};
        compare(t);
        scenario u = base("release with another generation", 0, 1, 2, DIM, 7);
        u.release.generation = 8;
        compare(u);
        scenario v = base("release of more columns than written", 0, 1, 1, 7, 7);
        v.release.last_columns = 9;
        compare(v);
        scenario w = base("release starts past the first slot's columns", 0, 2, 2, DIM, 7);
        w.first_column = DIM + 1;
        compare(w);
    }
    {
        scenario s = base("no sink, no hardware", 0, 1, 1, DIM, 7);
        s.no_sink = true;
        compare(s);
        scenario t = base("null session", 0, 1, 1, DIM, 7);
        t.null_session = true;
        compare(t);
        scenario u = base("POTAL_SCALE_CPU keeps its per-column records", 0, 2, 2, DIM, 7);
        u.scale_cpu = true;
        compare(u);
    }

    std::printf("scenarios_hash=%016" PRIx64 "\n", all_scenarios_hash);
    std::printf("POTAL_SCALE_WINDOW_TEST %s (%d failure(s))\n", failures == 0 ? "PASS" : "FAIL", failures);
    return failures == 0 ? 0 : 1;
}
