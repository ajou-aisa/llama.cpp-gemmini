# Gemmini CUDA PPL on `mxint`

The CUDA sources are in `ggml/src/ggml-cuda/cpu-exact/`. They are compiled into
`libggml-gemmini.so`; this host-graph path uses `GGML_CUDA=OFF` and `--gpu-layers 0`.
CUDA is optional and disabled by default. Existing CPU and Gemmini execution is
retained when these options are off. This is the existing experiment backend,
not a new optimized implementation.

| Path | Weight | Activation / GEMM | Attention |
|---|---|---|---|
| PoTal | Q4_HP1 / Q8_HP1, including MXINT payloads | Original EXSIA selection and folding; CUDA main/residual integer dots, SCU and INT32 saturation | Existing CUDA FP16 inputs / FP32 accumulation |
| RTN-WA MXINT | Same HP1/MXINT weights | BLOCK32 absmax RTN; no outlier selection, folding, or residual | Same FP attention |
| RTN-WA Q4_0/E4M3 | Prepared Q4_0 file with E4M3-rounded weight scales | Same A4 BLOCK32 RTN | Same FP attention |

PoTal uses DIM=BK=32 and the existing Gemmini stripe geometry. The default anchor
is TOP2; TOP3 is a build option. The original separate main/residual FP scaling
and FP32 merge are retained. The SCU kernels do not use Tensor Cores. Activation
preparation and the final merge remain on the CPU. Perplexity emits CUDA coverage
and saturation proofs; its NLL reduction uses a deterministic token order.

The custom RTN backend also accepts Q4_0, Q8_0, Q6_K and F16. For these, runtime
`GGML_CUDA_RTN_A_BITS=16` selects the weight-only/FP16 path and `4` or `8` selects
block RTN activation quantization. The HP1 RTN reader supports A4/A8 only.

## Build

Requires Linux, CMake, Ninja, a C++ compiler, CUDA/cuBLAS, and the existing Gemmini
headers. Set `GEMMINI_SW_PATH` to that installation on the target server.
`CUDAARCHS` defaults to `native` (the local GPU); `CUDACXX` defaults to
`/usr/local/cuda/bin/nvcc`. Models, datasets, build directories and experiment
results are not stored in Git.

```bash
export GEMMINI_SW_PATH=/path/to/RISC-V-DynDNN-gemmini-include
export CUDAARCHS=86 # This RTX 3070 machine configuration pins SM86.
bash scripts/experiment/build-potal-cuda.sh --bits 4 --anchor 2
bash scripts/experiment/build-potal-cuda.sh --bits 8 --anchor 2
# For the transfer queue example below:
bash scripts/experiment/build-potal-cuda.sh --bits 4 --anchor 3
bash scripts/experiment/build-potal-cuda.sh --bits 8 --anchor 3
bash scripts/experiment/build-potal-cuda.sh --rtn
```

The helper builds and runs the relevant scalar/CPU-versus-CUDA tests, then exits.
It never launches model inference. Builds use one worker by default (`--jobs N`
changes this). Logs are under `build-potal-cuda/<profile>/` as `configure.log`,
`build.log`, and `verification.log`.

For manual PoTal execution, set `GGML_GEMMINI_CUDA_CPU_EXACT=1`,
`GGML_GEMMINI_HP1_SCU=1`, and `GGML_CUDA_RTN_A_BITS=16`. The latter applies to
floating attention; it does not change the compiled EXSIA A4/A8 activation width.
For RTN-WA, use the `rtn` binary and `GGML_CUDA_RTN_A_BITS=4` or `8`.

## Queue and monitor

`scripts/experiment/cuda-ppl-jobs.example.json` preserves the earlier seven-job
transfer example. The machine configuration
`scripts/experiment/cuda-ppl-jobs.rtx3070.json` schedules exactly 24 full-corpus
runs in this order, with Qwen then Llama for each configuration:

| Order | Method | Weight / activation | Jobs |
|---|---|---|---|
| 1?2 | FP16 custom CUDA | FP16 / FP16 | 2 |
| 3?6 | RTN-W | Q4_0 / FP16, Q8_0 / FP16 | 4 |
| 7?10 | RTN-WA | Q4_0 / A4, Q8_0 / A8 | 4 |
| 11?14 | MXINT-WA, no compensation | HP1 MXINT4 / A4, HP1 MXINT8 / A8 | 4 |
| 15?18 | PoTal TOP2 | HP1 MXINT4 / A4, HP1 MXINT8 / A8 | 4 |
| 19?22 | PoTal TOP3 | HP1 MXINT4 / A4, HP1 MXINT8 / A8 | 4 |
| 23?24 | NVFP4 reference | NVFP4 W4 / A4 | 2 |

This configuration targets Ubuntu-24.04 WSL user `mojoi`, RTX 3070 SM86,
i5-10400 and 32 GB RAM. Source/models/dataset live under the Windows checkout;
executables and libraries live on ext4 under `/home/mojoi/build/`. All paths are
editable in JSON. FP16/RTN/MXINT use `potal-optimized-rtn`; PoTal uses
`potal-optimized-top{2,3}-a{4,8}`. Frozen `potal-baseline-*` builds are not used
or rebuilt by this configuration. Use `--build` only when rebuilding selected
runtimes is intended; it runs kernel tests and must not overlap another GPU user.

The common measurement protocol is context/batch/ubatch 512, seed 42, four
inference and batch threads, `--chunks -1`, stride 0, no warmup, GPU layers 0,
KV F16, and TF32 disabled. Attention uses FP16 QK/VP inputs and FP32 accumulation,
with F32 softmax. PoTal sets `GGML_CUDA_RTN_A_BITS=16` for floating attention and
requires the CUDA RTN attention build option; compiled A4/A8 still selects EXSIA
activation width. The full original WikiText-2 test file yields Qwen 584 and Llama
564 complete chunks; each chunk scores 255 tokens. Every job first runs a separate
one-chunk smoke check, then the full corpus.

NVFP4 uses existing FP16 GGUF files and the RTN executable, with
`GGML_CUDA_NVFP4=1`, `GGML_CUDA_RTN_A_BITS=4`, and an explicitly configured,
SHA256-pinned `libggml-nvfp4-preload.so`. Its reference recipe is block16 E2M1
values, E4M3FN block scales, FP32 tensor scale, and RNE rounding; GEMM dequantizes
to FP16 with FP32 accumulation. This is a reference quantization experiment on
SM86, not native FP4 hardware throughput. The queue requires both complete RTN
GPU coverage and the separate `NVFP4_PROOF` with matching scale/rounding and
operation counts. Missing library or mismatched proof fails the job. No W8/A4
fallback is silently substituted; any explicitly authorized fallback must have a
separate, clearly labelled configuration and output directory.

From PowerShell in the checkout, inspect the plan and check every input/build
before the first launch. These commands do not start inference:

```powershell
wsl -d Ubuntu-24.04 -u mojoi --exec python3 /mnt/c/Users/mojoi/Projects/PoTal/llama.cpp-gemmini/scripts/experiment/cuda-ppl-queue.py --config /mnt/c/Users/mojoi/Projects/PoTal/llama.cpp-gemmini/scripts/experiment/cuda-ppl-jobs.rtx3070.json --list
wsl -d Ubuntu-24.04 -u mojoi --exec python3 /mnt/c/Users/mojoi/Projects/PoTal/llama.cpp-gemmini/scripts/experiment/cuda-ppl-queue.py --config /mnt/c/Users/mojoi/Projects/PoTal/llama.cpp-gemmini/scripts/experiment/cuda-ppl-jobs.rtx3070.json --check
```

After integration and accuracy gates have passed, explicitly launch:

```powershell
.\scripts\experiment\start-cuda-ppl.ps1
```

The PowerShell launcher starts a hidden foreground WSL anchor, then a user systemd
service with a stable name derived from the output directory. The anchor keeps a
WSL client alive after the invoking terminal closes; systemd owns the queue and
its child processes. The launcher enables user lingering and prints the anchor
PID. It defaults to `/home/mojoi/potal-results/rtx3070-full-24`; `-Output`, `-Config`,
`-LinuxRoot`, `-Distro`, and `-LinuxUser` override machine paths. The Linux launcher
is also directly available as `bash scripts/experiment/start-cuda-ppl.sh`.
Windows sleep suspends computation. Reboot, Windows shutdown, `wsl --shutdown`,
or distro termination stops it. No automatic boot launch is installed. Invoke
the launcher again after restart to verify completed jobs and restart interrupted
phases from chunk zero; there is no mid-corpus checkpoint.

From PowerShell in the checkout, open the read-only monitor (Ctrl+C closes only
the display):

```powershell
.\scripts\experiment\watch-cuda-ppl.ps1
```

Use `-Once` for one text snapshot or `-Json` for one JSON snapshot. The wrapper
accepts the same machine path parameters as the launcher. Equivalent full command:

```powershell
wsl -d Ubuntu-24.04 -u mojoi --exec python3 /mnt/c/Users/mojoi/Projects/PoTal/llama.cpp-gemmini/scripts/experiment/watch-cuda-ppl.py --config /mnt/c/Users/mojoi/Projects/PoTal/llama.cpp-gemmini/scripts/experiment/cuda-ppl-jobs.rtx3070.json --output /home/mojoi/potal-results/rtx3070-full-24
```

Append `--once` for one text snapshot or `--json` for one JSON snapshot. The
monitor shows per-job state, completed chunks, partial cumulative PPL, elapsed
time and an approximate phase ETA. It tolerates partially appended log lines,
checks the full result proof before displaying DONE, and reports a dead inference
process as STOPPED. Closing it never stops inference. Both queue and monitor accept
`--config` and `--output`; repeated `--case ID` selects jobs in configured order.
Use a different output directory for a different selection or configuration.
`--smoke-only` runs only smoke phases and does not count as a full result.

The queue and its inference child share an exclusive Linux file lock, preserving
exclusion even if the parent dies before recording the child PID. The queue checks recorded
boot IDs and process start ticks before restarting. A surviving orphan inference
blocks all work in the same output directory, even if its parent was killed.
Systemd stops the whole process group; direct SIGTERM also terminates and reaps
the inference child. Interrupted directories are archived rather than overwritten.
The plan/config hash is immutable within an output directory. Completed phases
are reused only after matching command/environment, runner, model, corpus,
executable/library and CMake-cache hashes, exit status, all chunks, NLL/PPL, and
complete GPU/SCU proofs. Do not edit binaries or queue code during a measurement.

Each phase writes `{job}/{smoke,full}/{manifest.json,execution.json,ppl.log,
exit-status.txt,result.json}`. `queue.log`, `status.json`, `plan.json`, and
`results.tsv` are at the output root. The Linux launcher prints the systemd
service name; stop it with `systemctl --user stop <printed-service-name>` from
WSL. Results TSV distinguishes weight bits from activation bits for RTN-W.
Kernel tests and smoke checks do not establish independent full-corpus CPU parity.

The CPU-only integration harness uses actual queue, monitor, systemd and Windows
launcher subprocesses. It never runs CUDA:

```bash
python3 scripts/experiment/test-cuda-ppl-queue.py --receipts /mnt/c/Users/mojoi/Projects/PoTal/experiment-tools/queue-monitor-qa
```

It checks the exact 24-job ordering, malformed configurations, stale inputs and
configurations, reuse, live locks, orphan refusal, termination/restart, streaming
monitor parsing and incomplete GPU proofs. Fixtures use task-specific temporary
directories and are cleaned after verification. Receipts remain outside Git.
