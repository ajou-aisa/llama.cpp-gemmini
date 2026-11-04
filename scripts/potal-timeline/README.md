# PoTal timeline tools (H3-8I-25Q)

Python 3.12 stdlib only. Always run with `python3 -I`.

## CLI
    python3 -I timeline_parse.py TRACE.csv [--out DIR] [--strict] [--failed-runs 3,5-7] [--wfi-tol-ns 2000]
    python3 -I plot_timeline.py DIR [--out FILE] [--run RUN_ID] [--t0-ns N] [--t1-ns N]
    python3 -I breakdown.py PARSE_DIR BENCH_OUT [--skip-runs 1]      # per-run node/fill/II/drain/overlap/stage tables
    python3 -I overhead_summary.py RESULTS_DIR                       # bench-*.out medians, MAD, ON/OFF overhead
    python3 -I -m unittest discover -v -s scripts/potal-timeline -p test_timeline_parse.py

`-I` drops the current directory from `sys.path`, so `python3 -I -m unittest test_timeline_parse` cannot import
the module. Use `discover` (above) or `python3 -I test_timeline_parse.py -v`.

Parser input: potal_timeline_v1 CSV, 17-column E rows (mono_direct) or legacy 16-column rows (anchor interpolation).
A rdcycle-only event is placed only between two reference points on its thread (direct-stamped events and the
thread's anchors) whose rdcycle and MONOTONIC deltas agree within `wfi-tol-ns + 2e-4*dm`. Otherwise it stays unplaced and is counted. A span with direct stamps on both
ends (role.start) takes its duration from MONOTONIC.
Plotter input: DIR/spans.csv plus DIR/run_summary.csv when present. It writes DIR/timeline.svg: one lane per hart,
a sub-lane per thread, colors by stage_id, gray for wait spans, inset bars for depth>0 and node.total brackets.
Sub-pixel neighbours of the same thread/stage/depth are merged into one rect. `--t0-ns`/`--t1-ns` take absolute
MONOTONIC ns.

## Outputs (DIR, default TRACE.csv.d)
| file | columns |
|---|---|
| spans.csv | I25P columns + span_id,parent_id,depth,work_begin,work_end (work = END work) |
| stage_totals.csv | I25P columns + inclusive_ns(=total_ns),exclusive_ns,wait_inside_ns,work_sum,unplaced |
| thread_stage.csv | thread,role,hart,stage,kind,count,inclusive_ns,exclusive_ns,work_sum |
| thread_util.csv | I25P columns + unplaced_events; busy+wait+idle==window is checked (ERROR) |
| run_summary.csv | run_id,layer_id,node_total_ns,attributed_ns,wait_only_ns,unattributed_ns,critical_span_ns,critical_stage,begin_ns,end_ns,envelope_thread,clipped_spans |
| run_thread.csv | run_id,thread,role,busy_ns,wait_ns,idle_ns,exclusive_by_stage (stage:ns;...) |
| parent_gaps.csv | thread,stage,stripe,begin_ns,inclusive_ns,children_union_ns,uncovered_ns,run_id,span_id |
| buffers.csv | run_id,buffer_id,stripe,first_ns,last_ns,lifetime_ns,roles,spans,release_end_ns |
| status.csv | key,value: result,exit_code,errors,warnings,events,placed,unplaced,dropped,unregistered,rejected_threads,overflow_threads,bad_segments,spans,unplaced_spans,incomplete_reasons,parse_wall_s |
| stripes, stage_ii, overlap, critical_path .csv | unchanged from I25P |

Definitions:
- Nesting: per thread and kind, sorted by (begin asc, end desc, begin seq). Parent = innermost span containing the
  child. A partial overlap gives WARN `overlap-not-nested` and the child becomes a sibling.
- exclusive = duration minus the union of the span's direct children. The per-thread check is
  sum(exclusive) == busy union within 1 ns per span: a mismatch is an ERROR, or a WARN when overlap-not-nested
  spans on that thread double count their overlap.
- Runs: node.total (stage 1) is the envelope. attributed = union of all non-envelope busy spans of every thread,
  clipped to the envelope and matched by time, not run_id (the issuer tags acc.* with its own run).
  wait_only = wait spans not covered by attributed. unattributed = rest (>= 0). critical_span = longest clipped
  busy span. Crossing the envelope gives WARN `run-clip`. run_thread busy excludes the envelope stage.
- parent_gaps covers stages 22,21,26,39,1: uncovered = inclusive minus the children union.
- buffers excludes ids 0 and 4294967295. buffer_id is a ring slot reused by later stripes, so a lifetime is
  (run, buffer, stripe): a span that begins after the final.slot_release (44) END of the same run, buffer and
  stripe is an ERROR (`use after release`).
- t_local order is checked only where at least one of two consecutive events is rdcycle-only: an unpinned
  thread (all events direct-stamped) may migrate between harts, so its order is checked on MONOTONIC.
- Unplaced spans (an end without MONOTONIC) count in the `unplaced` columns, never in ns.

## Checks and exit codes
- ERROR (exit 1): format, pairing, seq, `#END` count, `mono-order` (a placed MONOTONIC decrease in seq greater than
  --wfi-tol-ns; smaller decreases are a WARN), direct-stamped RECV before SEND, util identity, exclusive-sum,
  use after release, unclosed BEGIN on a thread that did not overflow.
- INCOMPLETE (exit 3, only without errors): dropped>0, ring overflow, unregistered_records>0, rejected_threads>0.
- WARN (exit 2 only with --strict): hart-mismatch, cpu-end-mismatch, pin-start, pin-end, anchor-cpu, migration,
  clock refusal/bad segments, run-clip, overlap-not-nested. Exit 0: PASS.
Printed messages are capped at 20 per category; the counts in status.csv are exact.
