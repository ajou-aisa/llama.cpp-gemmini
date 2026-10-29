# H3-8C: DIM32 FireSim functional and GPT-2 baseline

Date: 2026-10-29. Build and ordered numerical gates: 2026-10-28.

Status: **DIM32 Functional hardware and exact GPT-2 256/128 E2E PASS**.
Local software and targeted RTL gates pass. The first isolated
FireSim/Vivado bitstream build and U280 infrastructure setup completed with
exit 0. Separate real HWDB/package and deployed-file checks pass.
Production Paired, both attention hardware suites, the 256/2 smoke, and the
single exact 256/128 request pass. All generated token IDs and stdout
match the preserved DIM16 result. The Functional comparison shows lower
TTFT but higher TPOT and total request latency; it is not a DIM-only
performance experiment.

## Provenance and artifact isolation

| Item | Identity |
|---|---|
| Starting software HEAD, branch potal | `fc0bdcdd0dba8ccfef8a54cd2e3c7f39a0be3c9b` |
| Starting Chipyard RTL HEAD, branch potal | `10ff4a10a87662edc7efd5d1a718027c1853550b` |
| FireSim harness HEAD | `5c3458a3f54917ab9502bda4a8b3cc3dd513304d` |
| DIM32 parameter SHA256 | `3580b84f15ed8de9f9d311edea64eaa7c1f9bb308b1552a8f79453ec2c05ea33` |
| Preserved DIM16 parameter SHA256 | `b4cd8b9efef9f3cd9b598ae3cc84af6478f83e628ed73d94a77fb1c452df1fc7` |
| Preserved DIM16 bitstream SHA256 | `f4c465a451385cbda44c254fb038fc2ede3a25ea12d5e829c2a501b7d495b3de` |
| DIM32 bitstream SHA256 | `f114de218a94b61c57b6c730a5534b9c2b973ac7ddb23f70be5f25baf009aa75` |
| DIM32 package SHA256 | `2216632072477e3713f8a7ceb71ac90587356cccdf0fe234423480bcf22901c1` |

The commits identify starting revisions; H3-8 edits are uncommitted. Exact
build defaults and header provenance are in H3-8A. Port/test evidence is
in H3-8B. No commit or push was made.

The preserved DIM16 bitfile is:

```text
/home/alveo/firesim/deploy/results-build/2026-10-24--02-24-42-alveo_u280_firesim_potal_quadcore_d16_no_nic/cl_xilinx_alveo_u280-firesim-FireSim-FireSimQuadRocketSinglePoTalD16Config-BaseXilinxAlveoConfig/vivado_proj/firesim.bit
```

Its current bytes match the historical H3-7C bitstream hash. This is a
file-identity check, not a readback of programmed FPGA state. Original
DIM16 images, HWDB entry, rootfs and results have not been replaced.
Historical result/script hashes still pass their original manifest.
Evidence: `/tmp/h3-8/dim16-baseline-preservation.log`.

## Separate DIM32 build

New, isolated deployment files:

```text
/home/alveo/firesim/deploy/config_build_recipes_h3_8_d32.yaml
/home/alveo/firesim/deploy/config_build_h3_8_d32.yaml
```

Recipe: `alveo_u280_firesim_potal_quadcore_d32_no_nic`.
Target: `FireSimQuadRocketSinglePoTalD32Config`.
Platform: U280, `BaseXilinxAlveoConfig`, existing U280 bit builder.
Build destination: `/home/alveo/build_firesim_h3_8_d32`.
Strategy: TIMING. Requested FPGA host clock constraint: 30 MHz.
Nominal target clock: 1000 MHz. There is no frequency sweep.

The successful manager build wrote
`deploy/built-hwdb-entries/alveo_u280_firesim_potal_quadcore_d32_no_nic`.
The separate `deploy/config_hwdb_h3_8_d32.yaml` now names that completed
DIM32 artifact, not the DIM16 package. The package is:

```text
/home/alveo/firesim/deploy/results-build/2026-10-28--06-40-13-alveo_u280_firesim_potal_quadcore_d32_no_nic/cl_xilinx_alveo_u280-firesim-FireSim-FireSimQuadRocketSinglePoTalD32Config-BaseXilinxAlveoConfig/firesim.tar.gz
```

Its metadata identifies `FireSimQuadRocketSinglePoTalD32Config` in both
build and deployment quintuplets and the recorded FireSim harness commit.
The bitstream inside the package was independently hashed and matches the
standalone bitfile SHA256 above. The real FireSim HWDB/runtime parsers
accept the separate selection.
Evidence: `/tmp/h3-8/real-dim32-hwdb-package-validation.log`,
`packed-bitstream-match.log`, and the extracted
`package-contents/xilinx_alveo_u280/metadata`.
First-build manager output is retained in
`/tmp/h3-8/firesim-d32-build-first.log`, exit 0.

Build command, after all mandatory local gates:

```bash
source /home/alveo/miniforge3/etc/profile.d/conda.sh
cd /home/alveo/firesim/target-design/chipyard
source env.sh
conda activate /home/alveo/firesim/.conda-env
cd /home/alveo/firesim
source sourceme-manager.sh
cd deploy
firesim buildbitstream \
  -b config_build_h3_8_d32.yaml \
  -r config_build_recipes_h3_8_d32.yaml \
  -a config_hwdb.yaml
```

No DIM16 recipe or HWDB path is changed by this selection.

## Isolated runtime preparation

New runtime and workload files are separate from the DIM16 defaults:

```text
/home/alveo/firesim/deploy/config_runtime_h3_8_d32.yaml
/home/alveo/firesim/deploy/workloads/h3-8-d32-interactive.json
```

The simulation directory is `/home/alveo/FIRESIM_RUNS_DIR_H3_8_D32`,
not the existing DIM16 slot directory. The workload reuses the unchanged
interactive boot binary and uses its own
`deploy/workloads/h3-8-d32-interactive/rootfs.img`.
The actual FireSim `InnerRuntimeConfiguration` and `WorkloadConfig`
parsers accept these files and resolve the expected paths without FPGA
access. Evidence: `/tmp/h3-8/isolated-runtime-contract.log`.
The separate runtime now resolves against the actual DIM32 HWDB artifact.
Before programming, no active FireSim driver was found; the existing
one-board database maps the assigned U280 to PCI BDF `01:00.0`
(`10ee:903f`). The requested infrasetup uses only the new simulation
directory and the new HWDB:

```bash
firesim infrasetup -c config_runtime_h3_8_d32.yaml \
  -a config_hwdb_h3_8_d32.yaml -r config_build_recipes_h3_8_d32.yaml
```

It completed with exit 0:
`/tmp/h3-8/dim32-infrasetup-first.log`.
The deployed driver, bitstream, kernel and prepared image hashes match
their verified sources; the latter is still
`b488bab1ec0b451c8687547a26b03ef3c3789c07416b48cb4dc6e79d376efef1`
before guest boot.
Evidence: `/tmp/h3-8/deployed-dim32-inputs-sha256.log`.
The driver is the independently built DIM32 driver with SHA256
`94c8068ba2a1755eeca11bf0f81444c1d9d05c8a2161861e0ed5e775d7c6d51c`.

The inactive, unmounted historical image was copied into that separate
workload. Both initial images are 25769803776 bytes and have SHA256
`88dfda8fbfd9a79bfbc555b59b3af58adec814c07c47338d0f987a1b97402e11`.
The boot binary SHA256 is
`763772d3fa11b393c0b3154748afcbdc2f9c633e1301ddef05072530a0bc83a9`.
Evidence: `/tmp/h3-8/isolated-runtime-inputs-sha256.log`, exit 0.
This is the initial copied-image identity, before DIM32 payload staging.
It is not a completed DIM32 guest image or a hardware-gate result.

The isolated copy now contains the canonical DIM32 `bin` directory,
parameter manifest, original parameter header and unchanged prompt under
`/root/h3-8`. The staged CLI and three hardware-gate binaries match their
recorded source hashes. The staged prompt and existing GPT-2 model also
match the historical hashes below.
Evidence: `/tmp/h3-8/staged-d32-payload-sha256.log`.
After syncing and unmounting, the prepared image SHA256 is
`b488bab1ec0b451c8687547a26b03ef3c3789c07416b48cb4dc6e79d376efef1`.
Evidence: `/tmp/h3-8/staged-d32-rootfs-sha256.log`; staging exited 0.
No guest execution is implied.

The image is ext2. Read-only inspection used `loop,ro`; staging required
a fresh `loop,rw` mapping, not the ext4-only `noload` option or remounting
a read-only loop device writable. Its existing `/firemarshal.sh` exits 0
without launching a workload. Startup files were not changed, so merely
booting the image cannot auto-run the old DIM16 inference.

## Build prerequisite observation

The installed Vivado path is
`/tools/Xilinx/Vivado/2021.1/bin/vivado`.
The license environment points at `200@147.47.208.180`.
A real `lmstat` check returned exit 241:

```text
Cannot connect to license server system. (-15,570:115 "Operation now in progress")
```

Evidence: `/tmp/h3-8/vivado-license-preflight.log`.
This is a failed prerequisite check, not a claim that synthesis or routing
has already run or failed. The first actual build failure must be retained
separately if generation cannot complete.
The subsequent real main-IP synthesis run acquired its license:

```text
Command: synth_design -top design_1_firesim_wrapper_0_0 -part xcu280-fsvh2892-2L-e -mode out_of_context
INFO: [Common 17-349] Got license for feature 'Synthesis' and/or device 'xcu280'
```

Evidence: the isolated build's
`vivado_proj/firesim.runs/design_1_firesim_wrapper_0_0_synth_1/runme.log`.
Therefore the earlier `lmstat` failure is not an established blocker for
this build. The real main-IP synthesis subsequently completed successfully
with zero errors and zero critical warnings. The top-level `synth_1` also
finished at 16:17:40 on October 28, 2026. The subsequent first route and
bitstream build also completed successfully.

## Build acceptance ledger

| Required evidence | DIM32 status |
|---|---|
| Synthesis | main-IP synthesis successful; top-level synth_1 finished |
| Place-and-route | first route completed; reports preserved |
| Routed WNS/TNS and timing closure | setup 0.053/0.000 ns; hold 0.008/0.000 ns, first route |
| Full-image routed LUT/register/BRAM/DSP use | 521658 / 485141 / 563.5 BRAM tiles / 202 |
| ClockBridge generated frequency summary | elaborated: 1000 MHz reference, both outputs division 1 |
| Full DIM32 hardware parameter manifest | header verified; top-level elaboration and host RTL compilation complete |
| Routed physical FPGA host clock | 30.003 MHz, period 33.330 ns, from routed clock report |

The first routed report states that all user-specified timing constraints
are met. It has 1303919 setup endpoints, 1302959 hold endpoints and zero
failing endpoints. `check_timing` reports zero unclocked registers, zero
unconstrained internal endpoints and zero invalid generated clocks.
Its two inputs without input delay and one output without output delay
match the preserved DIM16 report.
This report establishes routed timing; the separate bitstream and guest
execution evidence above establishes their respective completion.

Main-IP post-synthesis utilization, not full-image routed utilization:

| Resource | Main-IP synthesized use |
|---|---:|
| CLB LUTs | 445673 |
| CLB registers | 393852 |
| Block RAM tiles | 430 |
| DSPs | 199 |

Source:
`vivado_proj/firesim.runs/design_1_firesim_wrapper_0_0_synth_1/design_1_firesim_wrapper_0_0_utilization_synth.rpt`.
This excludes separately synthesized platform IP and is not a routed
capacity or timing result.

The full-image placed report, also preserved with the first route, records
521658 CLB LUTs, 485141 CLB registers, 563.5 Block RAM tiles and 202 DSPs.
A separate read-only report of the routed checkpoint used by
`write_bitstream` independently confirms the same counts with declared
state `Routed`. It uses 40.01% of LUTs, 18.61% of registers, 27.95% of
BRAM tiles and 2.24% of DSPs.
Evidence: `/tmp/h3-8/dim32-utilization-routed.rpt` and
`routed-utilization-generation.log`, exit 0. No implementation command or
checkpoint modification was performed by that reporting step.

The unchanged U280 implementation script can invoke its standard ML QoR
strategies if the first routed setup/hold slack is negative. It refuses
bitstream completion if final setup or hold slack remains negative.
The first route passed without negative setup/hold slack. Its reports and
run log were captured under `/tmp/h3-8/first-route-evidence`; the capture
command exited 0. The preserved timing-summary SHA256 is
`35a4035f27e2321b8969907bc9480258b80fb10d82381e2412d48b6349a214c7`.
The complete hash ledger is
`/tmp/h3-8/first-route-evidence-sha256.log`.
No frequency sweep or new optimization strategy was introduced by H3-8.

Implementation emits Vivado 12-4739 warnings for the generated
`host_clock`/ClockBridge alias and its two multicycle constraints.
The preserved DIM16 `impl_1/runme.log` contains the same three warnings,
and both generated implementation XDC files are identical. This is an
existing platform constraint issue, not a new DIM32 source change.
The routed report additionally retains TIMING-4, TIMING-27 and TIMING-54
methodology warnings with the same counts as DIM16. Routed clock and
internal timing coverage have now been inspected as described above.
The pre-existing warnings remain visible; satisfying setup/hold does not
mean that the platform methodology report is warning-free.

The historical DIM16 routed timing report independently records the host
clock `clk_out1_design_1_clk_wiz_0_0` at period 33.330 ns, frequency 30.003 MHz.
That is distinct from its nominal 1 GHz target clock and cannot be copied
as a DIM32 routed result.

The new DIM32 top-level elaboration emitted:

```text
firesimRationalClockBridge Frequency Summary
  Input Reference Frequency: 1000.0 MHz
  Output clock clock_1000MHz, requested: 1000.0 MHz, actual: 1000.0 MHz (division of 1)
  Output clock harnessbinder_clock, requested: 1000.0 MHz, actual: 1000.0 MHz (division of 1)
```

This is the generated target-clock ratio, not routed FPGA frequency.
Generated files are under:

```text
/home/alveo/firesim/sim/generated-src/xilinx_alveo_u280/xilinx_alveo_u280-firesim-FireSim-FireSimQuadRocketSinglePoTalD32Config-BaseXilinxAlveoConfig/
```

The `firesim.firesim.FireSim.FireSimQuadRocketSinglePoTalD32Config` prefix
has `.fir`, `.dts` and `.firesimRationalClockBridge.freq-summary` outputs.
The generated DTS confirms four Rocket harts, IDs 0..3, and 16 GiB memory.
It does not establish Linux CPU-to-hart mapping before guest boot.
Golden Gate host-RTL compilation and the FPGA driver build completed.
The generated driver metadata uses `ClockInfo` numerator 1 and denominator
1 for each of the four hart trace bridges.
SHA256 evidence for this elaboration's parameter header, frequency summary,
DTS, FIRRTL, software manifest and both isolated recipe files is retained
in `/tmp/h3-8/generated-d32-provenance-sha256.log`.
Generated host-RTL and driver hashes are retained separately in
`/tmp/h3-8/generated-d32-host-rtl-driver-sha256.log`.
Canonical DIM32 RISC-V `llama-cli` and all three ordered hardware-gate
binaries have recorded ELF identities and hashes in
`/tmp/h3-8/canonical-d32-binary-provenance.log`.

## Mandatory ordered hardware gates

The new DIM32 guest boot and root login completed. Direct `/proc/cpuinfo`
inspection gives:

| Linux CPU | Hardware hart |
|---:|---:|
| 0 | 2 |
| 1 | 0 |
| 2 | 1 |
| 3 | 3 |

The shell affinity mask is `0xF`. Thus Hart0 is Linux CPU1, and the
historical worker mask `0xD` excludes it. The guest manifest reports DIM32,
the expected parameter SHA256 and `FireSimQuadRocketSinglePoTalD32Config`.
Evidence: `/tmp/h3-8/run/uartlog`, between `H3_8_IDENTIFY_BEGIN` and
`H3_8_IDENTIFY_END`.

`taskset -c 1 ./bin/potal-api-smoke` completed on real DIM32 hardware with
`H3_8_API_EXIT=0`. It reports all four required markers:
`POTAL_API_MAIN_GOLDEN PASS dim=32`,
`POTAL_API_INDEXED_RESIDUAL_GOLDEN PASS dim=32`,
`POTAL_API_LOOP_INDEX PASS dim=32` and `POTAL_API_SMOKE PASS`.
The checked session issued one Main and one Indexed Residual loop,
64 Index writes, 64 Scale writes and 64 Releases, then drained and closed
without outstanding work. Legacy M=N=32 numerical cases passed for K32
and K64, including passesPerBlock 1 and 2. Scale/encoding negative cases
also passed. Evidence: the same UART log and guest
`/root/h3-8/results/api.out`. This is actual hardware execution, not an
x86 emulator result.

`potal-paired-smoke --quick` then passed on the actual DIM32 FPGA with
`H3_8_PAIRED_EXIT=0` and `POTAL_PAIRED_SMOKE PASS`.
The `small_48x96x256` golden passed in Paired RTL-ordering, Paired
conservative-ordering and explicit Serial modes. The normal Paired case
issued one Main loop, four Indexed Residual loops, four Index configs,
208 Index writes and two readbacks covering 12 tiles. It used 1536 Scale
writes and 1536 Releases, with one paired stripe and zero fallback stripes.
Aggregate smoke statistics, including warmup probes, report zero failed
nodes, zero fallback stripes, balanced 4864 Scale writes/Releases and
`issuer_cpu=1`. These are smoke-test command counts, not GPT-2 counts or
accelerator timing. Evidence: the same UART log and guest
`/root/h3-8/results/paired.out`.

The full attention smoke completed with `H3_8_ATTENTION_EXIT=0` and
`POTAL_ATTENTION_SMOKE PASS`. QK K64/K33/K65/K4097 raw jobs and epochs
were exact, including the three-panel K4097 tail. PV K17/K33/K65 and
mixed-sign K128 cancellation cases were exact, with real command
completion and balanced Scale lifecycle. The destructive DIM32 QK witness
matched INT64 `141287244103424` and F32 `0.501953125`, distinct from the
forbidden shared/high+8 result. Its FP32 reference differs by design under
the declared saturation contract; this is not an FP32-quality claim.

The subsequent `--pv-range-only` suite completed with
`H3_8_RANGE_EXIT=0` and `POTAL_ATTENTION_SMOKE PASS`.
It reports M1/N64/K288, 12 heads, 49 jobs, one extended stripe, 16 bitplane
jobs, 11 corrections, 22 bias MVINs, 49 LOOP_WS commands, 71 fences and
28224 balanced Scale writes/Releases, with `exact=1 status=0`.
Evidence: the UART log and guest `results/attention.out`/`results/range.out`.
All mandatory numerical hardware gates therefore pass before GPT-2 begins.

The short request, execution `165-55127757000`, completed with exit 0.
Its completed logs were extracted read-only after the guest's `sync`;
`/tmp/h3-8/short-acceptance.json` records acceptance and
`/tmp/h3-8/short-results.sha256` hashes all six raw result files.
All 256 prompt token IDs match DIM16 exactly. Actual CPU and batch pools
each use three strictly pinned workers on Linux CPUs 0/2/3; context is 512
and batch/microbatch are 256/256.

Both output tokens were observed (`13`, `198`), with healthy logging and
978/978 successful joined node executions. All 146 MUL_MAT nodes report
`POTAL`/`TARGET_NPU`, with no required CPU GEMM fallback. The run completed
98 dense nodes, 24 QK nodes and 24 PV nodes. Paired execution reports 236
stripes, zero fallback stripes, 11292 Main loops, 81726 Indexed Residual
loops/configs, 3450582 Index writes and 32648587 balanced Scale
writes/Releases. The issuer record confirms Linux CPU1/Hart0.

Canonical guest CPU monotonic events give TTFT **51644.742 ms**, the
single token gap **7171.433 ms**, and request time **58817.091 ms**.
These exactly match independent event subtraction. They are short-request
metrics, not the full 128-token TPOT, accelerator cycles, hardware waiting
or a DIM-only speedup result.

The full request uses the same command and flags, changing only `-n 2`
to `-n 128` and its isolated result directory to `results/e2e`.
Its completion subscription was registered before its single submission.
Execution `173-130094429000` completed with `H3_8_E2E_EXIT=0`; the marker
was emitted after the guest wrote its exit file and ran `sync`.
The six completed files were then extracted read-only into
`/tmp/h3-8/full-results`. No live filesystem mount or rerun was used.

Strict completed-log acceptance is `/tmp/h3-8/full-acceptance.json`.
It confirms 256 prompt tokens, 128 sequential output-token events, context
512, batch/microbatch 256/256, seed 1, temperature 0, no warmup or prompt
cache, and both three-worker strict CPU pools on Linux CPUs 0/2/3.
The issuer record confirms Linux CPU1/Hart0. All 256 prompt token IDs and
all 128 generated token IDs match DIM16 exactly.

All 128 graphs and 62592 joined node executions complete successfully.
All 9344 MUL_MAT executions report `POTAL`/`TARGET_NPU`; required CPU GEMM
fallback is zero. There are 73 prefill and 9271 decode matmuls, 6272
completed dense nodes, 1536 completed QK nodes and 1536 completed PV nodes.
There are no failed operation events, failed node executions, failed dense
nodes or failed stripe-pipeline nodes. Final logging reports healthy
session completion. Paired stripes accepted and semantically completed
both equal 6410, fallback stripes are zero, and 1047139964 Scale writes
match 1047139964 Releases.

Independent subtraction of request-start, first-token, last-token and
request-end events agrees with the canonical final summary: TTFT
**50838.678 ms**, TPOT **7673.827929133859 ms/token** over 127 gaps, and
request latency **1025415.663 ms**. Prefill/decode phase elapsed times are
50803.437/972516.640 ms; worker CPU times are 18899.803/81555.396 ms.
These guest CPU measurements are not FPGA-host wall time or accelerator
timing. CPU stage-wall coverage remains incomplete, and no NPU cycle/time
counter is available.

Stdout is byte-identical to DIM16, valid UTF-8 and ASCII, without embedded
NULs. It contains the same eight repetitions of
`It's a ship that's been lost for a long time.` This is functional
execution acceptance, not a language-quality improvement or FP32-equivalence
claim. Original text, stderr, token IDs and counters remain in the raw
files and acceptance JSON.

Completed raw-file SHA256 values:

| File under `/tmp/h3-8/full-results` | SHA256 |
|---|---|
| `cycle-log.jsonl` | `2f4b4b422600935e9c48845671d44ad01ce9370704f00839d212f715ab3f63ee` |
| `debug-log.jsonl` | `1ba8a850bc0dabed58c94f97c9168a3a7b96c5ddba93899b1eaf98bd21f8ce85` |
| `exit.status` | `a1194dc419b036d0acb35fbfd55372ddff1c843d0dec91b03cacb7d329134afe` |
| `llama-cli.err` | `eca12e01f2cf47f89557d8e5a076b13d4086d514cacf94ffa78536a5eee02b62` |
| `llama-cli.out` | `ab7d19ee54b9e8ed832ea87f50ec982e9b5fd3eb32f995823a68248d252bfd47` |
| `semantic-graph.jsonl` | `51fc74b249299dbe8a5a67f829e32e31f4a148a9b237990f6402347814cd05e4` |

The checkable manifest is `/tmp/h3-8/full-results.sha256`.
Earlier prefix/prefill snapshots are retained as milestones, not substituted
for this completed-request evidence.

After evidence extraction, the isolated guest received `sync` and
`poweroff` through the existing driver UART. Guest shutdown completed and
the actual FireSim driver exited 0, recorded in
`/tmp/h3-8/run/driver.exit`. Its log reports `Simulation complete` and
`*** PASSED ***`. This host-driver outcome is separate from the already
verified CLI exit and numerical acceptance.

The driver reports 53683.5 seconds of host wall time for the entire boot,
ordered gates, smoke, full request and interactive session; it is not the
GPT-2 request latency. Its 29.999 MHz host-frequency and 29.449 MHz effective
emulation-frequency summaries are also separate from nominal 1 GHz target
time and the routed 30.003 MHz clock constraint report.
The final session/command manifest is `/tmp/h3-8/final-session.sha256`.
Driver-log SHA256 is
`a87732ccfc364def420a2df45ec98c6413ab474d864f39b8391f36be9a6bbc39`;
UART-log SHA256 is
`63709629d7dd8d3746ce0fd864645aa002b7af667be1a9ec75c805488ff270b7`.
There is no remaining H3-8 hardware/E2E blocker.

For reproduction, the ordered numerical gate procedure is:

1. Confirm all four harts and identify the Linux CPU mapped to Hart0.
   Run the direct C11 API smoke on that CPU, not an arbitrary boot CPU.
2. `potal-api-smoke` checks actual Scale/Index/LOOP_WS commands, raw Main
   Golden, then raw Indexed Residual Golden, in that order. Require
   `POTAL_API_MAIN_GOLDEN PASS`, `POTAL_API_INDEXED_RESIDUAL_GOLDEN PASS`,
   `POTAL_API_LOOP_INDEX PASS` and final `POTAL_API_SMOKE PASS`.
3. Run `potal-paired-smoke --quick` through the production hart-0 issuer.
   Check actual Paired Main/Residual, Index commands, exact golden output,
   no unexpected fallback, successful completion and Scale balance.
4. Run `potal-attention-smoke`, then
   `potal-attention-smoke --pv-range-only`.
   Require QK/PV hardware completion, raw-pass checks, exact output,
   PV plane completion, successful drains and balanced Scale lifecycle.

The direct API test and the backend-issuer tests have different affinity
requirements: pin the former to confirmed Hart0. Let the latter's issuer
pin itself; do not restrict the entire backend process to one CPU.
Any failure stops before model inference. An x86 command emulator or a
cross-build cannot satisfy any of these hardware gates.

## Exact historical model and request

The existing inputs were hashed without modification:

```text
models/gpt2/gpt2.Q8_HP1.gguf
SHA256=cf23a45e4db23736d7d1f67ab42018803a198cfcb01db36daca69ff56bdbc7d4

/tmp/h3-7-gpt2-e2e/prompt-256.txt
SHA256=7e13482ec87666be18c7e5e14e04b14f4d09405f1099cfd68d92696c5469cd78
```

The historical execution script remains
`/tmp/h3-7c/executed-inference.sh`. Preserve its inference options and
environment cleanup; only use separate DIM32 binary/result/rootfs paths.

```sh
export GEMMINI_MATMUL_MODE=STRIPE_PIPELINE
export GEMMINI_RMD_BACKEND=WS
export GGML_POTAL_PAIRED_AUTO=ON
export GGML_POTAL_PAIRED_FENCES=RTL
GGML_SCHED_DEBUG=2 ./bin/llama-cli \
  -m /root/workspace/llama.cpp-potal/models/gpt2/gpt2.Q8_HP1.gguf \
  -f /root/h3-8/prompt-256.txt \
  -n 128 -c 512 -b 256 -ub 256 -t 3 -tb 3 \
  --cpu-mask 0xD --cpu-mask-batch 0xD --cpu-strict 1 --cpu-strict-batch 1 \
  --temp 0 -s 1 -no-cnv --no-warmup --no-context-shift \
  --no-display-prompt --simple-io --verbose-prompt
```

Before the single full 256/128 request, run the same 256-token prompt with a
two-token output budget. This exercises Prefill and a subsequent Decode.
Do not start the full request unless every hardware and short-request gate
passes. Confirm the actual prompt IDs, token budget, issuer CPU/Hart0,
worker placement, Paired route and QK/PV offload in the new guest.

## Comparison and requested measurements

Both configurations must use the Functional profile for this baseline:
LOG_DEBUG=1, LOG_CYCLE=1, CYCLE_SIM=0, CYCLE_DETAIL=0, planning diagnostics OFF.
The historical DIM16 row is H3-7C evidence, not a new rerun.

| Measurement | Historical DIM16 Functional | DIM32 Functional |
|---|---:|---|
| Prompt / generated tokens | 256 / 128 | 256 / 128 |
| Guest monotonic TTFT | 53400.266 ms | 50838.678 ms |
| Guest monotonic TPOT | 7177.511755905512 ms/token | 7673.827929133859 ms/token |
| Total request latency | 964945.217 ms | 1025415.663 ms |
| Prefill / decode phase elapsed | 53366.037 / 909575.396 ms | 50803.437 / 972516.640 ms |
| Prefill / decode worker CPU time | 18845.633 / 75935.666 ms | 18899.803 / 81555.396 ms |
| Nominal target clock | 1 GHz | generated 1 GHz, division 1; guest boot verified |
| Physical routed host clock | 30.003 MHz | 30.003 MHz, first routed report |
| Paired stripes / fallback stripes | 6410 / 0 | 6410 / 0 |
| Main / Indexed Residual LOOP_WS | 284004 / 1669931 | 315960 / 1880918 |
| Index Config / Write | 1669931 / 31814198 | 1880918 / 64122297 |
| Readback commands / tiles | 250112 / 2279344 | 278304 / 1100288 |
| Paired fences / drains | 6410 / 1960345 | 6410 / 2203288 |
| Scale Write / Release | 1052152636 / 1052152636 | 1047139964 / 1047139964 |
| Attention QK / PV completed nodes | 1536 / 1536 | 1536 / 1536 |
| QK limb jobs / LOOP_WS | 113184 / 118368 | 113184 / 226368 |
| PV limb jobs / LOOP_WS / bias MVIN | 55871 / 55871 / 82324 | 56027 / 56027 / 38834 |
| PV extensions: nodes / stripes / plane jobs | 10 / 11 / 176 | 19 / 23 / 368 |
| Required CPU GEMM fallback | 0 | 0 |
| Paired fallback | 0 | 0 |
| Generated text and token IDs | preserved historical originals | byte-identical text; all 128 IDs identical |

Machine-readable comparison: `/tmp/h3-8/dim16-dim32-comparison.json`.
TTFT decreased 4.796957%, but TPOT increased 6.914878% and total request
latency increased 6.266723%. This is the observed comparison of one
Functional run each, not a speedup claim. DIM16 used software commit
`8fb19aa5b66ace4cc50014581e5b3fe5ad7fc546`; DIM32 used the starting software
revision recorded above plus the uncommitted H3-8 changes. Instrumentation
matches the historical Functional settings, but software and hardware
revisions differ, so DIM alone is not an isolated causal variable.

The command ledger does not show a general fourfold reduction in work:
DIM32 has more dense Main/Residual LOOP_WS and Index writes, more QK
LOOP_WS, fewer readback tiles and fewer PV bias MVINs. These are completed
command/declared-work counts, not actual DMA volume, bus time, hardware
waiting or proof of the latency contribution of any one stage. The
different PV extension counts also preclude assuming identical internal
integer intermediates from identical generated token IDs.

Architectural thread CPU cycle measurements remain unavailable in these
canonical phase summaries. They must not be relabeled accelerator cycles.
No accelerator
cycles, DMA bytes/time, hardware-wait durations or overlap measurements are
inferred from CPU API elapsed times or successful Fence responses.
Overlapping CPU-stage or command intervals are not summed as independent
graph-wall-time contributions.

No Performance-profile hardware comparison was run. Such a later
comparison would require matching reduced instrumentation on DIM16 and
DIM32. The Functional latency regression is retained, not optimized away
or hidden by changing the request, profile, numerical contract or flags.

## H3-8D condition

The DIM32 Functional and full E2E prerequisites now pass. The subsequent
plan is `H3-8D-e4c1-e4c2-implementation-plan.md`. It remains a separate
planning deliverable: no E4C1 production integration, E4C2 implementation,
unrelated optimization, repository commit or push is part of H3-8.
