// Host test of the PoTal backend logging integration (no accelerator needed):
// every PoTal record goes to the existing Gemmini debug JSONL, nothing routine
// reaches stdout/stderr, and the cycle/semantic records carry the node's
// inference identity and the POTAL classification.
//
//   potal-log-test        uses a fresh GEMMINI_LOG_DIR it creates under $TMPDIR
//   (child mode)          re-executed with --default-dir-child in a fresh CWD
//                         without GEMMINI_LOG_DIR: ./output/log/debug-log.jsonl
//
// Off RISC-V the HP1 MUL_MAT fails at its first hardware command
// (POTAL_ERR_NO_HARDWARE): the test checks the failure records and that no
// hardware success or NPU cycle is invented. The success path (dispatch=HARDWARE,
// the POTAL NPU record) runs on FireSim only.
//
// Final line: POTAL_LOG_TEST PASS | POTAL_LOG_TEST FAIL (<n>)

#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml-potal.h"
#include "log.hpp"

#include "json.hpp"

#include <gemmini/log.hpp>
#include <gemmini/performance.hpp>
#include <gemmini/semantic.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {

using json = nlohmann::ordered_json;

int g_failures = 0;

void check(const std::string & name, bool ok, const std::string & detail = "") {
    std::printf("%s %s%s%s\n", ok ? "PASS" : "FAIL", name.c_str(), detail.empty() ? "" : ": ", detail.c_str());
    if (!ok) {
        g_failures++;
    }
}

std::string read_file(const std::string & path) {
    std::ifstream     in(path);
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

// Every line of a JSONL file, parsed; `bad` counts lines that are not one JSON object.
std::vector<json> read_jsonl(const std::string & path, int & bad) {
    std::vector<json> out;
    std::ifstream     in(path);
    std::string       line;
    bad = 0;
    while (std::getline(in, line)) {
        json j = json::parse(line, nullptr, false);
        if (j.is_discarded() || !j.is_object()) {
            bad++;
        } else {
            out.push_back(std::move(j));
        }
    }
    return out;
}

std::vector<json> with_layer(const std::vector<json> & recs, const std::string & layer, const char * needle = "") {
    std::vector<json> out;
    for (const auto & r : recs) {
        if (r.value("layer", "") == layer && r.value("msg", "").find(needle) != std::string::npos) {
            out.push_back(r);
        }
    }
    return out;
}

bool any_msg(const std::vector<json> & recs, const char * needle) {
    for (const auto & r : recs) {
        if (r.value("msg", "").find(needle) != std::string::npos) {
            return true;
        }
    }
    return false;
}

// stdout and stderr of the process go to files while the backend runs.
class capture_output {
  public:
    explicit capture_output(const std::string & dir) {
        std::fflush(stdout);
        std::fflush(stderr);
        saved_out_ = dup(1);
        saved_err_ = dup(2);
        const int out = open((dir + "/captured-stdout.txt").c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        const int err = open((dir + "/captured-stderr.txt").c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        dup2(out, 1);
        dup2(err, 2);
        close(out);
        close(err);
    }

    ~capture_output() {
        std::fflush(stdout);
        std::fflush(stderr);
        dup2(saved_out_, 1);
        dup2(saved_err_, 2);
        close(saved_out_);
        close(saved_err_);
    }

  private:
    int saved_out_ = -1;
    int saved_err_ = -1;
};

struct hp1_graph {
    ggml_context_ptr        wctx; // the weight, in a POTAL weight buffer (what the weight cache caches)
    ggml_backend_buffer_ptr wbuf;
    ggml_context_ptr        ctx;
    ggml_cgraph *           graph = nullptr;
    ggml_tensor *           y     = nullptr;
};

// One F32 x Q8_HP1 MUL_MAT named like GPT-2's first QKV projection.
hp1_graph make_qkv_graph(ggml_backend_dev_t dev) {
    hp1_graph        g;
    ggml_init_params wp = { ggml_tensor_overhead(), nullptr, true };
    g.wctx.reset(ggml_init(wp));
    ggml_tensor * w = ggml_new_tensor_2d(g.wctx.get(), GGML_TYPE_Q8_HP1, 64, 16);
    ggml_set_name(w, "blk.0.attn_qkv.weight");
    g.wbuf.reset(ggml_backend_alloc_ctx_tensors_from_buft(g.wctx.get(), ggml_backend_dev_buffer_type(dev)));
    ggml_backend_buffer_set_usage(g.wbuf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    std::vector<float>   wf(16 * 64, 0.5f);
    std::vector<uint8_t> wq(ggml_nbytes(w));
    ggml_quantize_chunk(GGML_TYPE_Q8_HP1, wf.data(), wq.data(), 0, 16, 64, nullptr);
    ggml_backend_tensor_set(w, wq.data(), 0, wq.size());

    ggml_init_params p = { 16 * 1024 * 1024, nullptr, false };
    g.ctx.reset(ggml_init(p));
    ggml_tensor * x = ggml_new_tensor_2d(g.ctx.get(), GGML_TYPE_F32, 64, 4);
    ggml_set_name(x, "attn_norm-0");
    std::fill(static_cast<float *>(x->data), static_cast<float *>(x->data) + 4 * 64, 1.0f);
    g.y = ggml_mul_mat(g.ctx.get(), w, x);
    ggml_set_name(g.y, "wqkv-0");
    g.graph = ggml_new_graph(g.ctx.get());
    ggml_build_forward_expand(g.graph, g.y);
    return g;
}

bool no_potal_prefix(const std::string & text, std::string & found) {
    for (const char * p : { "[POTAL_BACKEND]", "[POTAL_ISSUER]", "[POTAL_TLB]", "[POTAL_MATMUL]", "[POTAL] summary",
                            "[POTAL]" }) {
        if (text.find(p) != std::string::npos) {
            found = p;
            return false;
        }
    }
    return true;
}

// ---- child: default ./output/log location ---------------------------------

int default_dir_child() {
    ggml_backend_dev_t dev = ggml_backend_dev_by_name("POTAL");
    if (dev == nullptr) {
        return 2;
    }
    ggml_backend_t backend = ggml_backend_dev_init(dev, "gpt2");
    ggml_backend_free(backend);
    int        bad  = 0;
    const auto recs = read_jsonl("output/log/debug-log.jsonl", bad);
    return bad == 0 && !with_layer(recs, "potal.backend", "registered").empty() &&
                   !with_layer(recs, "potal.summary").empty() ?
               0 :
               1;
}

int run_default_dir_child(const std::string & base) {
    std::string       dir = base + "/cwd-XXXXXX";
    std::vector<char> buf(dir.begin(), dir.end());
    buf.push_back('\0');
    if (mkdtemp(buf.data()) == nullptr) {
        return -1;
    }
    const pid_t pid = fork();
    if (pid == 0) {
        unsetenv("GEMMINI_LOG_DIR");
        if (chdir(buf.data()) != 0) {
            _exit(3);
        }
        execl("/proc/self/exe", "potal-log-test", "--default-dir-child", static_cast<char *>(nullptr));
        _exit(4);
    }
    int status = 0;
    waitpid(pid, &status, 0);
    return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc > 1 && std::strcmp(argv[1], "--default-dir-child") == 0) {
        return default_dir_child();
    }

    const char * tmp  = std::getenv("TMPDIR");
    std::string  base = std::string(tmp && *tmp ? tmp : "/tmp") + "/potal-log-test-XXXXXX";
    std::vector<char> dir_buf(base.begin(), base.end());
    dir_buf.push_back('\0');
    if (mkdtemp(dir_buf.data()) == nullptr) {
        std::printf("FAIL mkdtemp\nPOTAL_LOG_TEST FAIL (1)\n");
        return 1;
    }
    const std::string dir = dir_buf.data();
    setenv("GEMMINI_LOG_DIR", dir.c_str(), 1);
    std::printf("POTAL_LOG_TEST GEMMINI_LOG_DIR=%s LOG_DEBUG=%d LOG_CYCLE=%d\n", dir.c_str(), LOG_DEBUG, LOG_CYCLE);

#if !LOG_DEBUG
    // No debug sink: the backend must still run and stay quiet on success paths.
    {
        capture_output cap(dir);
        ggml_backend_t b = ggml_backend_dev_init(ggml_backend_dev_by_name("POTAL"), "gpt2");
        ggml_backend_free(b);
    }
    std::string found;
    check("LOG_DEBUG=0: no routine PoTal terminal output",
          no_potal_prefix(read_file(dir + "/captured-stdout.txt") + read_file(dir + "/captured-stderr.txt"), found),
          found);
    std::printf(g_failures ? "POTAL_LOG_TEST FAIL (%d)\n" : "POTAL_LOG_TEST PASS\n", g_failures);
    return g_failures ? 1 : 0;
#else
    const std::string debug_path = dir + "/debug-log.jsonl";
    const std::string cycle_path = dir + "/cycle-log.jsonl";
    const std::string sem_path   = dir + "/semantic-graph.jsonl";

    check("CPU-only build proof is false in a GGML_POTAL build",
          !ggml::gemmini::semantic::compiled_cpu_only_build());

    ggml_backend_dev_t dev = ggml_backend_dev_by_name("POTAL");
    check("POTAL device registered", dev != nullptr);
    if (dev == nullptr) {
        std::printf("POTAL_LOG_TEST FAIL (%d)\n", g_failures);
        return 1;
    }

    ggml_backend_potal_stats st_decode = {};
    ggml_status              status_prefill = GGML_STATUS_SUCCESS, status_decode = GGML_STATUS_SUCCESS;
    bool                     prefill_nan    = false;
    {
        capture_output cap(dir);

        // Two instances: llama's (architecture) and a direct caller's (none).
        ggml_backend_t backend = ggml_backend_dev_init(dev, "gpt2");
        ggml_backend_t direct  = ggml_backend_dev_init(dev, nullptr);
        ggml::potal::log::ensure_outputs(); // repeated setup keeps the records written so far

        namespace perf     = ggml::gemmini::performance;
        namespace semantic = ggml::gemmini::semantic;
        perf::start_request(1);

        // Prefill operation under a semantic run, as llama-cli opens one.
        hp1_graph g = make_qkv_graph(dev);
        perf::begin_operation(perf::Phase::prefill, 2);
#if LOG_CYCLE
        std::shared_ptr<semantic::Session> session = semantic::Session::start(
            semantic::Source::Unspecified, "{\"model\":\"potal-log-test\"}", "{\"producer\":\"potal-log-test\"}",
            semantic::compiled_cpu_only_build());
        const int32_t                  token = 1;
        std::vector<semantic::Node>    nodes;
        for (int i = 0; i < ggml_graph_n_nodes(g.graph); ++i) {
            nodes.push_back({ ggml_graph_node(g.graph, i), "{}", false });
        }
        session->phase("prefill", {}, &token, 1);
        session->graph(nodes, "[]");
#endif
        status_prefill = ggml_backend_graph_compute(backend, g.graph);
        prefill_nan    = std::isnan(static_cast<float *>(g.y->data)[0]);
#if LOG_CYCLE
        session.reset(); // failed run (host): the next graph runs outside it
#endif
        perf::end_operation(3, status_prefill == GGML_STATUS_SUCCESS);

        // Decode operation: the same weight (cache), shape M=4 kept on purpose.
        perf::begin_operation(perf::Phase::decode, 4);
        status_decode = ggml_backend_graph_compute(backend, g.graph);
        perf::end_operation(5, status_decode == GGML_STATUS_SUCCESS);
        perf::finish_request(6);
        ggml_backend_potal_get_stats(backend, &st_decode);

        // A direct caller: no architecture, no inference context.
        hp1_graph gd = make_qkv_graph(dev);
        (void) ggml_backend_graph_compute(direct, gd.graph);

        // Concurrent writers: one JSON object per line, nothing interleaved.
        std::vector<std::thread> threads;
        for (int t = 0; t < 8; ++t) {
            threads.emplace_back([t] {
                for (int i = 0; i < 100; ++i) {
                    ggml::potal::log::debug("potal.test", "[POTAL_TEST] thread=%d record=%d \"quoted\" \\ text", t, i);
                }
            });
        }
        for (auto & th : threads) {
            th.join();
        }

        ggml_backend_free(direct);
        ggml_backend_free(backend);
        (void) ggml::gemmini::log::cycle.flush();
    }

    // ---- terminal --------------------------------------------------------
    const std::string out = read_file(dir + "/captured-stdout.txt");
    const std::string err = read_file(dir + "/captured-stderr.txt");
    std::string       found;
    check("no routine PoTal output on stdout", no_potal_prefix(out, found), found);
    check("no PoTal output on stderr (failures included: they are in the debug JSONL)", no_potal_prefix(err, found),
          found);

    // ---- debug JSONL -------------------------------------------------------
    int        bad  = 0;
    const auto recs = read_jsonl(debug_path, bad);
    check("debug JSONL under GEMMINI_LOG_DIR, every line one JSON object", !recs.empty() && bad == 0,
          std::to_string(recs.size()) + " records, " + std::to_string(bad) + " malformed");
    check("registration record exactly once", with_layer(recs, "potal.backend", "registered name=POTAL").size() == 1);
    check("device_init record per instance", with_layer(recs, "potal.backend", "device_init dim=16").size() == 2);

    const auto config = [&](const char * phase) {
        for (const auto & r : recs) {
            const std::string m = r.value("msg", "");
            if (m.find("[POTAL] config") != std::string::npos && m.find(phase) != std::string::npos) {
                return r;
            }
        }
        return json();
    };
    const json pre = config("phase=prefill");
    const json dec = config("phase=decode");
    check("HP1 config record: layer from model_arch=gpt2",
          pre.value("layer", "") == "blk.0.attn.qkv_proj" &&
              pre.value("msg", "").find("model_arch=gpt2") != std::string::npos,
          pre.dump());
    check("HP1 config record: weight, node, operation, backend",
          pre.value("msg", "").find("weight=blk.0.attn_qkv.weight") != std::string::npos &&
              pre.value("msg", "").find("node=wqkv-0") != std::string::npos &&
              pre.value("msg", "").find("operation=MUL_MAT") != std::string::npos &&
              pre.value("msg", "").find("backend=POTAL") != std::string::npos);
    check("HP1 config record: prefill identity (request, operation, graph, operator)",
          pre.value("msg", "").find("decode_ordinal=null request_id=1 operation_id=1") != std::string::npos &&
              pre.value("msg", "").find("graph_id=") != std::string::npos &&
              pre.value("msg", "").find("operator_id=") != std::string::npos);
    check("HP1 config record: decode identity with ordinal",
          dec.value("msg", "").find("phase=decode decode_ordinal=1 request_id=1 operation_id=2") != std::string::npos,
          dec.dump());
    check("tensor shape counter kept separate from the phase (rows=prefill_shaped in decode)",
          dec.value("msg", "").find("rows=prefill_shaped(M>1)") != std::string::npos);

    const auto weights = with_layer(recs, "blk.0.attn.qkv_proj", "weight_cache=");
    // Host failures are fatal (no accelerator): the prepared weight is abandoned,
    // never reused, so the decode run prepares it again (a HIT needs hardware).
    check("weight cache records: MISS, then MISS again after the fatal failure abandoned the entry",
          weights.size() == 2 && any_msg({ weights[0] }, "weight_cache=MISS") &&
              any_msg({ weights[1] }, "weight_cache=MISS"));
    check("HP1 route record", !with_layer(recs, "blk.0.attn.qkv_proj", "[POTAL] route=Tensor+block").empty());
    const auto failed = with_layer(recs, "blk.0.attn.qkv_proj", "dispatch=FAILED");
    check("host: failure stage and status recorded, no HARDWARE dispatch claimed",
          failed.size() == 2 && any_msg(failed, "status=POTAL_ERR_NO_HARDWARE") &&
              !any_msg(recs, "dispatch=HARDWARE"));
    check("host: graph_compute still FAILED, dst NaN (fail-stop kept)",
          status_prefill == GGML_STATUS_FAILED && status_decode == GGML_STATUS_FAILED && prefill_nan);
    const json direct_cfg = config("model_arch=unknown");
    check("direct caller: model_arch=unknown, phase=none, the logger's own unclassified layer",
          direct_cfg.value("layer", "") == "unclassified.blk.0.attn_qkv.weight" &&
              direct_cfg.value("msg", "").find("phase=none") != std::string::npos,
          direct_cfg.dump());

    const auto summary = with_layer(recs, "potal.summary", "[POTAL] summary");
    bool       all_fields = summary.size() == 2;
    for (const char * f :
         { "hp1_nodes=2", "hp1_completed=0", "hp1_failed=2", "prefill_shaped=2", "decode_shaped=0",
           "route_tensor_block=", "tiled_jobs=", "loop_ws_issued=", "fences=", "scale_writes=", "scale_releases=",
           "tlb_flushes=", "zero_carriers=", "shift_carriers=", "weight_cache_hits=0", "weight_cache_misses=2",
           "weight_cache_entries=0", "weight_cache_bytes=", "weight_validations=2", "weight_packs=2", "act_quant_s=",
           "weight_pack_s=", "weight_alloc_s=", "weight_lookup_s=", "route_wall_s=" }) {
        bool here = false;
        for (const auto & s : summary) {
            here = here || s.value("msg", "").find(f) != std::string::npos;
        }
        if (!here) {
            all_fields = false;
            std::printf("     summary misses %s\n", f);
        }
    }
    check("summary record per backend with every field, written before free", all_fields);
    check("stats match the records (2 misses, 0 hits, 0 completed)",
          st_decode.weight_cache_misses == 2 && st_decode.weight_cache_hits == 0 && st_decode.hp1_completed == 0);

    size_t thread_records = with_layer(recs, "potal.test", "[POTAL_TEST]").size();
    check("8 threads x 100 records, none lost or corrupted", thread_records == 800,
          std::to_string(thread_records));

    // ---- cycle / semantic --------------------------------------------------
#if LOG_CYCLE
    int        cbad   = 0;
    const auto cycles = read_jsonl(cycle_path, cbad);
    json       act;
    for (const auto & c : cycles) {
        if (c.value("op", "") == "potal.act_quant" && c.value("semantic_phase_kind", "") == "prefill") {
            act = c;
            break;
        }
    }
    check("cycle JSONL valid", !cycles.empty() && cbad == 0);
    check("potal.act_quant carries the node's semantic identity (not outside_collection)",
          !act.is_null() && act.value("layer", "") == "blk.0.attn.qkv_proj" && !act.contains("exclusion_reason") &&
              act.value("source_role", "") == "OBSERVATION_ONLY" && act.contains("semantic_node_ordinal"),
          act.dump());
    bool decode_outside = false;
    for (const auto & c : cycles) {
        decode_outside = decode_outside ||
                         (c.value("op", "") == "potal.act_quant" && c.value("exclusion_reason", "") == "outside_collection");
    }
    check("graph run without a semantic run stays outside_collection (not invented)", decode_outside);
    bool npu_invented = false;
    for (const auto & c : cycles) {
        npu_invented = npu_invented || (c.value("kind", "") == "npu" && c.value("collected", false));
    }
    check("no collected NPU cycle record on the host", !npu_invented);

    int        sbad = 0;
    const auto sem  = read_jsonl(sem_path, sbad);
    json       exec, run;
    for (const auto & s : sem) {
        if (s.value("kind", "") == "NODE_EXECUTION") {
            exec = s;
        }
        if (s.value("kind", "") == "RUN") {
            run = s;
        }
    }
    check("semantic RUN is not FULL_CPU", run.value("source_role", "") == "OBSERVATION_ONLY" &&
                                              run.value("cpu_only_build", true) == false,
          run.dump());
    check("semantic NODE_EXECUTION: POTAL MUL_MAT is TARGET_NPU",
          sbad == 0 && exec.value("actual_backend", "") == "POTAL" &&
              exec.value("execution_class", "") == "TARGET_NPU" && exec.value("semantic_phase_kind", "") == "prefill",
          exec.dump());
#else
    std::printf("SKIP cycle/semantic checks: LOG_CYCLE=0 build\n");
    (void) cycle_path;
    (void) sem_path;
#endif

    // ---- default location --------------------------------------------------
    const int child = run_default_dir_child(dir);
    check("without GEMMINI_LOG_DIR: ./output/log/debug-log.jsonl", child == 0, "child exit " + std::to_string(child));

    std::printf(g_failures ? "POTAL_LOG_TEST FAIL (%d)\n" : "POTAL_LOG_TEST PASS\n", g_failures);
    return g_failures ? 1 : 0;
#endif
}
