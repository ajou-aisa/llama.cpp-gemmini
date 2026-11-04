import csv
import os
import subprocess
import sys
import tempfile
import unittest
import xml.dom.minidom

HERE = os.path.dirname(os.path.abspath(__file__))
PARSER = os.path.join(HERE, "timeline_parse.py")
PLOTTER = os.path.join(HERE, "plot_timeline.py")
NO = 4294967295
B, E, WB, WE, SEND, RECV, INST = 1, 2, 3, 4, 5, 6, 7


def thread(slot, hart, role="issuer", overflow=0, dropped=0):
    return {"slot": slot, "hart": hart, "role": role, "overflow": overflow, "dropped": dropped}


def ev(t, slot, stage, et, mono=None, run=1, stripe=NO, tile=NO, buf=NO, edge=0, work=0):
    """mono=None means the same value as t (direct stamp, so t must be > 0); mono=0 means rdcycle-only."""
    return {"t": t, "slot": slot, "stage": stage, "et": et, "mono": t if mono is None else mono, "run": run,
            "stripe": stripe, "tile": tile, "buf": buf, "edge": edge, "work": work}


def trace_text(threads, events, legacy=False, anchors=(), unregistered=0):
    per_slot = {}
    rows = []
    for e in events:
        seq = per_slot.get(e["slot"], 0)
        per_slot[e["slot"]] = seq + 1
        cols = [e["t"], seq, e["slot"], e["stage"], e["et"], 0, 1, 1, e["run"], 0, e["stripe"], e["tile"],
                e["buf"], e["edge"], e["work"]]
        if not legacy:
            cols.append(e["mono"])
        rows.append("E," + ",".join(map(str, cols)))
    out = ["# potal_timeline v1",
           "META,format=potal_timeline_v1,model_id=test,clock_local=rdcycle,unregistered_records=%d,"
           "rejected_threads=0" % unregistered,
           "S,1,node.total", "S,21,stripe.residual_prepare", "S,45,acc.scale_window", "S,54,role.start"]
    for t in threads:
        h = t["hart"]
        n_anchor = sum(1 for a in anchors if a[0] == t["slot"])
        out.append("T,%d,%s,-1,-1,%d,%d,1,0,0,%d,0,%d,%d,%d,%d,0,0,0"
                   % (t["slot"], t["role"], h, h, h, per_slot.get(t["slot"], 0), t["dropped"], t["overflow"],
                      n_anchor))
    for i, (slot, mono, tl) in enumerate(anchors):
        cpu = next(t["hart"] for t in threads if t["slot"] == slot)
        out.append("A,%d,%d,register,0,%d,0,%d,%d" % (slot, i, mono, tl, cpu))
    out.append("#END,events=%d" % len(rows))
    return "\n".join(out + rows) + "\n"


class ParserCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def parse(self, threads, events, *extra, **kw):
        path = os.path.join(self.dir, "trace.csv")
        with open(path, "w") as f:
            f.write(trace_text(threads, events, **kw))
        self.out = os.path.join(self.dir, "out")
        p = subprocess.run([sys.executable, "-I", PARSER, path, "--out", self.out] + list(extra),
                           capture_output=True, text=True)
        self.stdout = p.stdout + p.stderr
        return p.returncode

    def rows(self, name):
        with open(os.path.join(self.out, name), newline="") as f:
            return list(csv.DictReader(f))

    def status(self):
        return {r["key"]: r["value"] for r in self.rows("status.csv")}

    def stage_row(self, sid, kind="busy"):
        return next(r for r in self.rows("stage_totals.csv") if r["stage_id"] == str(sid) and r["kind"] == kind)

    def test_01_nested_window_overlapping_children(self):
        evs = [ev(100, 0, 45, B), ev(200, 0, 48, B), ev(250, 0, 49, B), ev(300, 0, 48, E), ev(400, 0, 49, E),
               ev(500, 0, 48, B, tile=1), ev(600, 0, 48, E, tile=1), ev(1000, 0, 45, E)]
        self.assertEqual(self.parse([thread(0, 0)], evs), 0, self.stdout)
        row = self.stage_row(45)
        self.assertEqual((int(row["inclusive_ns"]), int(row["exclusive_ns"])), (900, 600))
        spans = self.rows("spans.csv")
        win = next(s for s in spans if s["stage_id"] == "45")
        kids = [s for s in spans if s["stage_id"] in ("48", "49")]
        self.assertEqual(win["depth"], "0")
        self.assertTrue(all(k["parent_id"] == win["span_id"] and k["depth"] == "1" for k in kids))
        self.assertIn("overlap-not-nested", self.stdout)

    def test_02_wait_inside_busy_identity(self):
        evs = [ev(100, 0, 21, B), ev(300, 0, 20, WB), ev(600, 0, 20, WE), ev(1100, 0, 21, E), ev(1600, 0, 50, INST)]
        self.assertEqual(self.parse([thread(0, 0)], evs), 0, self.stdout)
        u = self.rows("thread_util.csv")[0]
        busy, wait, idle, win = (int(u[k]) for k in ("busy_ns", "wait_ns", "idle_ns", "window_ns"))
        self.assertEqual((busy, wait, idle, win), (700, 300, 500, 1500))
        self.assertEqual(busy + wait + idle, win)
        self.assertEqual(int(self.stage_row(21)["wait_inside_ns"]), 300)

    def test_03_unclosed_begin(self):
        evs = [ev(10, 0, 21, B), ev(100, 0, 50, INST)]
        self.assertEqual(self.parse([thread(0, 0)], evs), 1, self.stdout)
        self.assertEqual(self.parse([thread(0, 0, overflow=1, dropped=3)], evs), 3, self.stdout)
        self.assertEqual(self.status()["result"], "INCOMPLETE")

    def test_04_end_without_begin(self):
        self.assertEqual(self.parse([thread(0, 0)], [ev(10, 0, 50, INST), ev(100, 0, 21, E)]), 1, self.stdout)
        self.assertIn("without matching begin", self.stdout)

    def test_05_wfi_segment(self):
        evs = [ev(10, 0, 50, INST), ev(500, 0, 10, B, mono=0), ev(600, 0, 10, E, mono=0),
               ev(1000, 0, 50, INST, mono=50000),
               ev(1200, 0, 11, B, mono=0), ev(1500, 0, 11, E, mono=0), ev(2000, 0, 50, INST, mono=51000)]
        self.assertEqual(self.parse([thread(0, 0)], evs), 0, self.stdout)
        self.assertEqual(self.status()["unplaced"], "2")
        self.assertEqual(self.status()["bad_segments"], "1")
        spans = {s["stage_id"]: s for s in self.rows("spans.csv")}
        self.assertEqual((spans["10"]["begin_ns"], spans["10"]["end_ns"]), ("", ""))
        self.assertEqual((spans["11"]["begin_ns"], spans["11"]["end_ns"]), ("50200", "50500"))
        self.assertEqual(self.stage_row(10)["unplaced"], "1")
        # Default rule: |dm - dc| <= 2000 + 2e-4 * dm; with dm = 1e6 the effective tolerance is 2200 ns.
        for dc, unplaced in ((997800, "0"), (997799, "1")):
            evs = [ev(10, 0, 50, INST), ev(1000, 0, 50, INST, mono=0), ev(10 + dc, 0, 50, INST, mono=1000010)]
            self.assertEqual(self.parse([thread(0, 0)], evs), 0, self.stdout)
            self.assertEqual(self.status()["unplaced"], unplaced, dc)

    def test_06_handoff_recv_before_send_direct(self):
        evs = [ev(2000, 0, 40, SEND, edge=7), ev(1000, 1, 40, RECV, edge=7)]
        self.assertEqual(self.parse([thread(0, 0), thread(1, 4, "finalizer")], evs), 1, self.stdout)
        self.assertIn("RECV precedes SEND", self.stdout)

    def test_07_mono_decrease(self):
        evs = [ev(0, 0, 50, INST, mono=10000), ev(100, 0, 50, INST, mono=5000)]
        self.assertEqual(self.parse([thread(0, 0)], evs), 1, self.stdout)
        self.assertIn("mono-order", self.stdout)

    def test_08_run_summary(self):
        evs = [ev(1000, 0, 1, B), ev(2000, 0, 1, E),
               ev(1100, 1, 30, B, run=99), ev(1400, 1, 30, E, run=99),
               ev(1300, 2, 31, B), ev(1700, 2, 31, E), ev(1700, 2, 41, WB), ev(1800, 2, 41, WE)]
        th = [thread(0, 1, "caller"), thread(1, 0), thread(2, 4, "finalizer")]
        self.assertEqual(self.parse(th, evs), 0, self.stdout)
        r = self.rows("run_summary.csv")
        self.assertEqual(len(r), 1)
        got = tuple(int(r[0][k]) for k in ("node_total_ns", "attributed_ns", "wait_only_ns", "unattributed_ns"))
        self.assertEqual(got, (1000, 600, 100, 300))
        rt = {x["thread"]: x for x in self.rows("run_thread.csv")}
        self.assertEqual(rt["1"]["busy_ns"], "300")
        gaps = self.rows("parent_gaps.csv")
        self.assertEqual([(g["stage"], g["uncovered_ns"]) for g in gaps], [("node.total", "1000")])

    def test_09_counter_payload(self):
        evs = [ev(10, 0, 45, B, work=0), ev(500, 0, 45, E, work=1000)]
        self.assertEqual(self.parse([thread(0, 0)], evs), 0, self.stdout)
        self.assertEqual(self.stage_row(45)["work_sum"], "1000")
        s = self.rows("spans.csv")[0]
        self.assertEqual((s["work"], s["work_begin"], s["work_end"]), ("1000", "0", "1000"))

    def test_10_mono_direct_span_duration(self):
        evs = [ev(500, 0, 54, B, mono=1000), ev(500, 0, 54, E, mono=301000)]
        self.assertEqual(self.parse([thread(0, 0)], evs), 0, self.stdout)
        s = self.rows("spans.csv")[0]
        self.assertEqual(int(s["end_ns"]) - int(s["begin_ns"]), 300000)
        self.assertEqual(s["dur_local"], "0")
        self.assertEqual(self.stage_row(54)["inclusive_ns"], "300000")

    def test_11_buffer_use_after_release(self):
        evs = [ev(100, 1, 44, B, buf=5), ev(200, 1, 44, E, buf=5),
               ev(300, 0, 42, B, buf=5), ev(400, 0, 42, E, buf=5)]
        self.assertEqual(self.parse([thread(0, 0), thread(1, 4, "finalizer")], evs), 1, self.stdout)
        self.assertIn("use after release", self.stdout)
        b = self.rows("buffers.csv")[0]
        self.assertEqual((b["first_ns"], b["last_ns"], b["lifetime_ns"]), ("100", "400", "300"))

    def test_12_plot_smoke(self):
        evs = [ev(10, 0, 1, B), ev(1000, 0, 1, E), ev(100, 1, 45, B, stripe=3), ev(900, 1, 45, E, stripe=3),
               ev(200, 2, 20, WB), ev(300, 2, 20, WE)]
        th = [thread(0, 1, "caller"), thread(1, 0), thread(2, 1, "exsia_local0")]
        self.assertEqual(self.parse(th, evs), 0, self.stdout)
        svg = os.path.join(self.dir, "t.svg")
        p = subprocess.run([sys.executable, "-I", PLOTTER, self.out, "--out", svg], capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        doc = xml.dom.minidom.parse(svg)
        labels = sorted(t.firstChild.data for t in doc.getElementsByTagName("text")
                        if t.getAttribute("class") == "lane-label")
        self.assertEqual(labels, ["hart 0", "hart 1"])
        self.assertEqual(len([p for p in doc.getElementsByTagName("path")
                              if p.getAttribute("class") == "run-bracket"]), 1)

    def test_13_legacy_16_columns(self):
        evs = [ev(1000, 0, 21, B), ev(2000, 0, 21, E)]
        anchors = [(0, 50000, 0), (0, 53000, 3000)]
        self.assertEqual(self.parse([thread(0, 0)], evs, legacy=True, anchors=anchors), 0, self.stdout)
        s = self.rows("spans.csv")[0]
        self.assertEqual((s["begin_ns"], s["end_ns"]), ("51000", "52000"))

    def test_14_anchor_closes_trailing_segment(self):
        # Events after the last direct stamp are placed only via a later anchor whose deltas agree.
        evs = [ev(1000, 0, 50, INST), ev(1500, 0, 10, B, mono=0), ev(1600, 0, 10, E, mono=0)]
        self.assertEqual(self.parse([thread(0, 0)], evs, anchors=[(0, 2000, 2000)]), 0, self.stdout)
        self.assertEqual(self.status()["unplaced"], "0")
        spans = {s["stage_id"]: s for s in self.rows("spans.csv")}
        self.assertEqual((spans["10"]["begin_ns"], spans["10"]["end_ns"]), ("1500", "1600"))
        self.assertEqual(self.parse([thread(0, 0)], evs, anchors=[(0, 90000, 2000)]), 0, self.stdout)
        self.assertEqual(self.status()["unplaced"], "2")
    def test_15_unpinned_migration_direct_only(self):
        # Direct-stamped events of a migrating thread: t_local may go backwards, MONOTONIC must not.
        evs = [ev(5000, 0, 50, INST, mono=1000), ev(100, 0, 50, INST, mono=2000)]
        self.assertEqual(self.parse([thread(0, 0, "caller")], evs), 0, self.stdout)
        evs = [ev(5000, 0, 50, INST, mono=1000), ev(100, 0, 50, INST, mono=0)]
        self.assertEqual(self.parse([thread(0, 0, "caller")], evs), 1, self.stdout)
        self.assertIn("t_local goes backwards", self.stdout)

if __name__ == "__main__":
    unittest.main()
