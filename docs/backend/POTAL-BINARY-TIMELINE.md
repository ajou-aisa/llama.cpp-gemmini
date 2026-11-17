# Fast shutdown trace export on FireSim

W4 and W8 can export the CPU timeline as binary after inference. Recording,
quantization, GEMM scheduling, counter collection and timing boundaries are
unchanged. This reduces shutdown serialization work; it is not an inference
TTFT/TPOT optimization. The E2E runner flushes after recording the token times.

## Guest setup

Rebuild the W4 or W8 `potal-e2e-runner` using the same RISC-V Linux toolchain and
SDK as the existing run, and install it in the guest image. No RTL, bitstream,
RVV or new RoCC command is required. Use:

```sh
export POTAL_TIMELINE=/root/w4/results/e2e-1-A/timeline.bin
export POTAL_TIMELINE_FORMAT=binary
# Keep the experiment's existing profile, capacities and CPU counter settings.
# POTAL_TIMELINE_CPU=1 includes the same CPU samples in the binary file.
```

`POTAL_TIMELINE_FORMAT` is consulted at flush. Unset or `csv` preserves existing
CSV output. Unknown values fail the flush. Choose a fresh output path for each
run. A successful flush writes a temporary file, closes it and renames it to
the configured path. Continue the existing guest sync/shutdown and host disk
extraction procedure: close/rename alone does not promise power-loss durability.
Do not stop FireSim merely because the last token has appeared.

If a launcher overwrites `POTAL_TIMELINE`, update that assignment too. The file
extension does not select the format. A binary trace must be converted before
passing it to tools expecting CSV, even if a launcher named it `timeline.csv`.

## Host conversion

After the run has stopped, extract the guest results with the existing
`debugfs rdump` procedure. On the host, from the repository root:

```sh
python3 scripts/potal-timeline/binary_to_csv.py /path/to/timeline.bin --out /path/to/timeline.csv
python3 -I scripts/potal-timeline/timeline_parse.py /path/to/timeline.csv --out /path/to/parsed
```

The converter uses Python 3.12 standard-library modules only. Run the converter
without `-I` so it can import its adjacent `binary_format.py`. When CPU sampling
was enabled it also writes `timeline.csv.cpu.csv`, including cycle, instret,
validity flags and counter source. It refuses existing destination files and
publishes no files for malformed/truncated input. Publication uses same-filesystem
hard links so a concurrently created destination is not overwritten. The two publications are not
a cross-file atomic transaction; an OS error during publication is reported.

All stage definitions, run/thread metadata, affinity checks, clock anchors,
dropped/overflow counters and event values are retained. The converter performs
the same op/phase fallback from RUN metadata as the CSV writer. Sparse CPU
samples remain associated with the original thread slot and sequence number.
Existing parser interpretation of missing/invalid counters is unchanged.

## Binary v1 layout

The first line is `# potal_timeline_binary v1`. A `#BINARY` line specifies
little-endian, 64-byte events, 56-byte CPU samples, CPU enablement and counter
source. Existing textual metadata through `#EVENT` follows. Metadata is small;
the millions of events do not undergo numeric-to-text conversion in the guest.

Each block begins with `#BLOCK,slot=S,events=N,samples=M\n`, then contains N event
records and M CPU records, followed by a newline. N is 1..4096, M is 0..N. A
thread's sequence numbers continue across blocks. CPU records are sorted by
sequence and only exist for samples with nonzero `mono_ns`. The stream ends
with `#END,events=TOTAL\n`. Count, identity, layout and truncation checks are not
a checksum for arbitrary bit corruption.

Events use the existing 64-byte layout (`<QIHHBBBB7IQQ`): clock, sequence, thread
slot, stage, event type, status, op, phase, run, layer, stripe, tile, buffer,
edge, zero padding, work, direct monotonic stamp. Compile-time checks cover all
field offsets. The writer rejects a big-endian host rather than mislabeling it.

CPU records have explicit fields (`<5QIiII`): mono, thread CPU ns, process CPU ns,
cycles, instret, sequence, signed CPU ID, validity flags, zero reserved word.
Flags bits 0/1/2 are thread/process/counter validity. This avoids serializing
native bool padding. Conversion buffers at most one block at a time; the
writer uses at most 4096 * 56 bytes of heap scratch per ring during flush.

The hardware-window exporter was already binary and is unchanged. No compression
or background I/O was added inside measured execution.

## Verification

On local Linux (WSL is sufficient; no NPU required):

```sh
uv run --no-project --with pytest python -m pytest scripts/potal-timeline/test_binary_to_csv.py -q
```

This compiles the actual W4 and W8 timeline writers and checks conversion
against their existing CSV writer on the same rings, including 64-bit boundary
values, op/phase fallback, multiple threads, multiple blocks and CPU samples.
Flush-time anchors naturally differ between the two flush calls; the test
excludes only those new anchors and the corresponding anchor count.
Corrupt input tests cover truncation, layout/count/slot errors and trailing data.
The fixture also rejects any timeline file created before the explicit flush.

FireSim shutdown speed still requires measurement on its guest filesystem.
Local flush timings must not be reported as Rocket or FPGA speedups.

### Local validation, 2026-11-17

Both backends passed their existing C++ timeline tests (CPU sampling, overflow,
reuse, recording gate and C API), and the binary round-trip/corruption tests.
The legacy CSV parser's 15 tests passed. The event writer's recording path and
E2E timing boundaries were not modified.

An indicative x86-64 WSL `/tmp` smoke run used 1,000,000 events across two
threads, with CPU sampling enabled on selected events. Each format was flushed
once from the same captured rings; binary ran first, so these are not controlled
target benchmarks or repeated medians:

| Backend | CSV flush | Binary flush | CSV + CPU bytes | Binary bytes |
|---|---:|---:|---:|---:|
| W8 | 0.982167 s | 0.056795 s | 86,286,584 | 64,231,437 |
| W4 | 0.954943 s | 0.041399 s | 86,286,874 | 64,231,447 |

These times include file close but not a durable-storage `fsync`. No new
FireSim execution or RISC-V performance measurement was performed.
