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
`CUDAARCHS` defaults to `87` (Orin); `CUDACXX` defaults to
`/usr/local/cuda/bin/nvcc`. Models, datasets, build directories and experiment
results are not stored in Git.

```bash
export GEMMINI_SW_PATH=/path/to/RISC-V-DynDNN-gemmini-include
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

`scripts/experiment/cuda-ppl-jobs.example.json` contains the seven experiments
previously prepared for transfer: Llama RTN-WA MX4 / Q4_0-E4M3 / MX8, then Qwen
and Llama TOP3 PoTal A4/A8. It is an example configuration, not an active queue.
Edit model/dataset/binary paths before use; relative paths resolve from the repo
root. TOP2 jobs use `anchor: 2` and their corresponding TOP2 binary.

```bash
python3 scripts/experiment/cuda-ppl-queue.py --list
python3 scripts/experiment/cuda-ppl-queue.py --check
# Starts only when explicitly invoked; survives SSH logout via user systemd.
bash scripts/experiment/start-cuda-ppl.sh
python3 scripts/experiment/watch-cuda-ppl.py
```

Both programs accept `--config FILE` and `--output DIR`. Use repeated `--case ID`
arguments to select jobs in their configured order. `--smoke-only` runs just the
one-chunk checks. `watch-cuda-ppl.py --once` prints a single snapshot.
To build/test the selected runtimes in the background before measurement, use
`bash scripts/experiment/start-cuda-ppl.sh --build`. The launcher enables user
lingering so the systemd queue continues after SSH logout.

Each job runs a CUDA smoke chunk, checks the full CUDA graph/SCU proof, and then
runs `--chunks -1` (Qwen 584 / Llama 564 complete 512-token chunks for the original
WikiText-2 file). The queue checks actual totals, every chunk, NLL/PPL consistency,
input hashes and build settings. Completed results are reused only with matching
provenance. Interrupted runs are archived and restarted from their first chunk;
there is no mid-corpus checkpoint resume. Kernel tests and a smoke run do not
constitute an independent full-corpus CPU parity measurement.

Results are in `output/cuda-ppl/<job>/{smoke,full}/`; `results.tsv` collects full
results. The monitor reports build steps, live process state, chunks, cumulative
PPL, elapsed time and estimated remaining time. Closing it does not stop the
queue. The launcher prints the systemd service name; stop an explicitly launched
queue with `systemctl --user stop <printed-service-name>`.

The queue deliberately uses ordinary RTN-WA for the uncompensated baseline.
Dropping only PoTal's residual limbs is a different ablation and is not scheduled.
