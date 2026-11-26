# H3-8I-25 checkpoint: software commits and evidence index

Date: 2026-11-04. Branch `potal`. Parent: `c477d22` (feat(potal): add paired J/K selection and control-cost tooling).

This file records which results support the H3-8I-25 software checkpoint. **Every evidence file listed below is kept outside this repository (under `/tmp` on the lab host). It is not versioned here and may be removed by a host cleanup.** The SHA256 values identify the copies that existed when the checkpoint was made.

## Commits (llama.cpp-gemmini, branch potal)
| Commit | Subject | Verified by (rerun at checkpoint time) |
|---|---|---|
| e9135db | feat(potal): add balanced PV attention algorithm with checked reconstruction | exported-tree native build, 11 test binaries exit 0 |
| 791c5e9 | feat(potal): add checked Register-fed Index Queue C API and emulator model | exported-tree native build, 11 test binaries exit 0 |
| 9bcda12 | feat(potal): add opt-in five-hart role pipeline, Linux Index Queue path and timeline v2 | exported tree identical to the preserved H3-8I-25P end state; 13 test binaries exit 0; 67 ROLES PASS / 0 FAIL; RISC-V bench/probe build |
| 7756d0d | feat(potal): add window-level timeline profile and issuer stage attribution | fresh native build, 13 test binaries exit 0, 67 ROLES PASS; RISC-V bench bit-identical to the binary run on FPGA in H3-8I-25Q |
| 4ba05c3 | feat(potal): add timeline parser, breakdown and SVG plot tools | 15 unittest cases OK |
| 97e31c4 | build(potal): add release static RISC-V toolchain with pthread and libatomic | repo-path build reproduces the FPGA-run binaries bit-identically |

Not committed: `models/gpt2/` (two .gguf model files, 412 MB, local downloads).

## Hardware image used by the FPGA evidence
- Bitstream `/tmp/h3-8i-25j/fpga-platform/vivado_proj/firesim.bit`, sha256 `480020cbdc269fd08ce48aa316dd9666c99755a2de439f1c679bb23114a28b23`.
- Config `FireSimFiveRocketSinglePoTalD32PairedStreamingIndexQueueConfig` (chipyard-aisa branch potal).

## FPGA results referenced (not re-run for this checkpoint)
| Step | Result | Report (sha256) |
|---|---|---|
| H3-8I-25O | Five-core bare-metal: Scalar 8 + Register-fed Queue 8 cases PASS (16/16), golden match | /tmp/h3-8i-25o/final.md |
| H3-8I-25P | Linux five-hart ROLES pipeline and Register-fed Queue bitwise equal to the CPU golden (runs 2 and 3) | /tmp/h3-8i-25p/final.md; /tmp/h3-8i-25p/results/linux-run2.sha256, linux-run3.sha256 |
| H3-8I-25Q | Window profile +2.77% over timeline OFF (n=10/10); 54/54 runs bitwise golden; issue order identical | /tmp/h3-8i-25q/final.md; /tmp/h3-8i-25q/results/linux-run1.sha256, analysis.sha256 |

```
0af3fa000e9cae8e3df99cbbab23dc81f922222f6999ef6f2be2adabed6060d7  /tmp/h3-8i-25o/final.md
3fe62def3ff9b7f7e0bb3b0f4ad8f80e5c67d34ab4b5a9c0580dc952de5cf357  /tmp/h3-8i-25p/final.md
987023aaa9be7785394a34bc80a2c0f517b56deb85e658cd4ea42669de53a9fc  /tmp/h3-8i-25p/results/linux-run2.sha256
e757cb4807fb697c9b5e3cf643d92178e3cecb6561799afe44814247355a79c5  /tmp/h3-8i-25p/results/linux-run3.sha256
61fdedcb2931eb9d1989ce678a0e020c84d2a06adc7c805a2e9e5c6f4c253514  /tmp/h3-8i-25q/final.md
720745d0696dc2a3ed3910426a46294e3f86df66285c62309f191ed77dab69e9  /tmp/h3-8i-25q/results/linux-run1.sha256
36115ce35bf9d3078295a5a46e692323c7a99d896812c97b683af20f7eee3fbf  /tmp/h3-8i-25q/results/analysis.sha256
```

Full per-file manifest: `/tmp/h3-8i-25r/phase0/evidence-manifest.sha256` (213 entries; sha256 of the manifest `13e70b49abbc5ef1be13de3287c30ddd239f89c12385d9e6434651297ea13d47`).

## Native results recorded at checkpoint time
- Current tree: 13/13 native test binaries exit 0; potal-stripe-pipeline-test 67 ROLES PASS / 0 FAIL; parser tests 15/15.
- Earlier records, not re-run: H3-8I-25Q native 13/13, ROLES 67/67, parser 15/15.

## Known limitations at this checkpoint
- About 16.9 ms of `ggml_backend_graph_compute` per call lies outside the instrumented node (`node.total`).
- Accelerator spans (acc.*) are CPU issue/return time; hardware DMA/GEMM intervals were not observed.
- The Register-fed Queue lifetime budget is 255 generations per FPGA programming; full-model Queue use is out of reach.
