# H3-6A FireSim paired performance profile

Hardware: H3-5B PoTal DIM16, FireSim U280. RTL: `dc2f52a5bd0236bc4e02a8d1109151945661755b`. Software: `potal` branch, uncommitted H3-3/H3-4 diagnostics preserved.

The benchmark reuses each Smoke case's shapes and input generator with seed `20261023`. It makes one weight tensor and one activation tensor per case and uses that pair for every route and repetition. The independent software Golden is computed before backend execution; every graph result is checked bitwise. Each route runs an untimed warmup. Cold samples clear the prepared-weight cache before every route, yielding one miss; steady samples reuse one prepared weight, yielding one hit. Route order rotates and reverses. H3-3/H3-4 diagnostic environment variables are rejected in benchmark mode. `POTAL_PAIRED_PROFILE=1` runs are separate from the baseline below.

Times are guest `steady_clock` milliseconds. `compute_ms` covers `ggml_backend_graph_compute`; `pipeline_wall_ms` is the backend pipeline counter. Dispersion is median absolute deviation (MAD). Every cell below contains seven repetitions on **one fixed input**, in execution order. Cold means a weight-cache miss after issuer and route warmup; it does not mean a cold FPGA or cold CPU cache.

| Case | Phase | Route | 7 compute_ms samples | Median [min,max]; MAD | Pipeline median [min,max]; MAD |
|---|---|---|---|---|---|
| small_48x96x256 | cold | serial | 21.250, 21.499, 22.146, 21.827, 21.488, 21.390, 21.458 | 21.488 [21.250,22.146]; 0.098 | 20.324 [20.233,20.998]; 0.088 |
| small_48x96x256 | cold | paired | 24.995, 24.403, 24.601, 24.445, 24.188, 24.518, 25.021 | 24.518 [24.188,25.021]; 0.115 | 23.388 [23.053,23.815]; 0.117 |
| small_48x96x256 | cold | paired_conservative | 24.608, 24.724, 24.289, 25.061, 24.936, 24.657, 24.613 | 24.657 [24.289,25.061]; 0.067 | 23.498 [23.112,23.864]; 0.094 |
| small_48x96x256 | steady | serial | 20.547, 21.707, 20.959, 21.288, 21.073, 20.916, 21.002 | 21.002 [20.547,21.707]; 0.086 | 20.498 [20.101,21.192]; 0.151 |
| small_48x96x256 | steady | paired | 24.112, 24.025, 23.924, 23.834, 23.838, 23.616, 23.858 | 23.858 [23.616,24.112]; 0.066 | 23.364 [23.110,23.609]; 0.101 |
| small_48x96x256 | steady | paired_conservative | 23.513, 23.764, 23.944, 23.975, 23.972, 24.144, 23.886 | 23.944 [23.513,24.144]; 0.058 | 23.432 [23.055,23.619]; 0.081 |
| decode_1x768x768 | cold | serial | 25.880, 24.403, 24.404, 24.393, 24.468, 24.386, 26.378 | 24.404 [24.386,26.378]; 0.018 | 8.502 [8.486,10.362]; 0.013 |
| decode_1x768x768 | cold | paired | 56.268, 56.541, 55.712, 56.593, 56.599, 56.509, 56.533 | 56.533 [55.712,56.599]; 0.060 | 40.578 [39.762,40.663]; 0.062 |
| decode_1x768x768 | cold | paired_conservative | 57.873, 56.719, 57.343, 57.398, 57.791, 57.419, 57.491 | 57.419 [56.719,57.873]; 0.076 | 41.469 [40.779,41.945]; 0.148 |
| decode_1x768x768 | steady | serial | 8.885, 9.301, 9.355, 9.317, 9.262, 9.420, 9.321 | 9.317 [8.885,9.420]; 0.038 | 8.789 [8.422,8.890]; 0.029 |
| decode_1x768x768 | steady | paired | 41.171, 41.382, 41.441, 41.455, 41.433, 41.383, 41.417 | 41.417 [41.171,41.455]; 0.034 | 40.924 [40.716,40.987]; 0.042 |
| decode_1x768x768 | steady | paired_conservative | 42.235, 42.186, 42.311, 42.322, 42.245, 42.189, 42.378 | 42.245 [42.186,42.378]; 0.059 | 41.782 [41.668,41.919]; 0.101 |
| gpt2_ffn_up_16x3072x768 | cold | serial | 130.114, 116.232, 116.035, 116.168, 116.267, 116.479, 116.269 | 116.267 [116.035,130.114]; 0.099 | 52.640 [52.457,56.798]; 0.130 |
| gpt2_ffn_up_16x3072x768 | cold | paired | 273.153, 278.997, 273.597, 272.322, 272.973, 269.043, 273.121 | 273.121 [269.043,278.997]; 0.476 | 209.467 [205.394,215.387]; 0.526 |
| gpt2_ffn_up_16x3072x768 | cold | paired_conservative | 275.470, 272.666, 273.854, 269.311, 273.099, 275.165, 269.701 | 273.099 [269.311,275.470]; 2.066 | 209.504 [205.695,211.839]; 2.037 |
| gpt2_ffn_up_16x3072x768 | steady | serial | 52.982, 53.582, 53.783, 53.724, 53.573, 53.748, 53.809 | 53.724 [52.982,53.809]; 0.085 | 53.185 [52.511,53.217]; 0.032 |
| gpt2_ffn_up_16x3072x768 | steady | paired | 206.441, 206.980, 209.738, 210.850, 207.965, 214.707, 211.457 | 209.738 [206.441,214.707]; 1.773 | 209.153 [205.921,214.187]; 1.722 |
| gpt2_ffn_up_16x3072x768 | steady | paired_conservative | 207.215, 208.175, 210.708, 210.010, 218.233, 212.058, 212.414 | 210.708 [207.215,218.233]; 1.706 | 210.185 [206.642,217.716]; 1.636 |
| dense_fallback_160x320x512 | cold | serial | 142.355, 144.064, 143.663, 143.416, 142.811, 141.225, 142.862 | 142.862 [141.225,144.064]; 0.554 | 138.357 [136.725,139.514]; 0.524 |
| dense_fallback_160x320x512 | cold | paired | 226.487, 228.401, 223.488, 225.271, 222.038, 226.739, 226.143 | 226.143 [222.038,228.401]; 0.872 | 221.554 [217.548,223.810]; 0.805 |
| dense_fallback_160x320x512 | cold | paired_conservative | 223.634, 224.472, 225.664, 225.631, 228.550, 228.356, 224.464 | 225.631 [223.634,228.550]; 1.167 | 221.062 [219.064,223.987]; 1.151 |
| dense_fallback_160x320x512 | steady | serial | 137.940, 141.048, 140.742, 141.150, 139.736, 138.842, 139.120 | 139.736 [137.940,141.150]; 1.006 | 139.071 [137.348,140.477]; 0.899 |
| dense_fallback_160x320x512 | steady | paired | 227.924, 226.710, 223.147, 222.397, 223.126, 224.385, 224.691 | 224.385 [222.397,227.924]; 1.259 | 223.654 [221.688,227.196]; 1.209 |
| dense_fallback_160x320x512 | steady | paired_conservative | 223.308, 222.079, 222.931, 225.520, 225.671, 225.612, 223.781 | 223.781 [222.079,225.671]; 1.702 | 223.074 [221.415,224.956]; 1.607 |


Steady median Serial/Paired RTL ratios (values below 1 mean Paired is slower):

| Case | Paired RTL | Paired Conservative | Paired RTL extra ms |
|---|---:|---:|---:|
| small_48x96x256 | 0.880 | 0.877 | 2.856 |
| decode_1x768x768 | 0.225 | 0.221 | 32.100 |
| gpt2_ffn_up_16x3072x768 | 0.256 | 0.255 | 156.014 |
| dense_fallback_160x320x512 | 0.623 | 0.624 | 84.649 |

The baseline build did not increment `act_quant_s` for STRIPE_PIPELINE. The final instrumented build times the ExSIA `quantize_activation` call, including producer capacity waits, and reports it in the profile pass. Stage times and backend counters can overlap; they must not be summed into graph wall time. Scale Release can wait for hardware completion. Main and Residual issue times are CPU-observed command latency, not accelerator execution time.

Cold versus steady prepared weights (`weight_pack_s` delta):

| Case | Route | Cold weight preparation ms | Steady weight preparation ms | Cold cache miss / steady hit |
|---|---|---:|---:|---:|
| small_48x96x256 | serial | 0.593 | 0.000 | 1 / 1 |
| small_48x96x256 | paired | 0.596 | 0.000 | 1 / 1 |
| small_48x96x256 | paired_conservative | 0.595 | 0.000 | 1 / 1 |
| decode_1x768x768 | serial | 14.446 | 0.000 | 1 / 1 |
| decode_1x768x768 | paired | 14.420 | 0.000 | 1 / 1 |
| decode_1x768x768 | paired_conservative | 14.468 | 0.000 | 1 / 1 |
| gpt2_ffn_up_16x3072x768 | serial | 59.504 | 0.000 | 1 / 1 |
| gpt2_ffn_up_16x3072x768 | paired | 59.510 | 0.000 | 1 / 1 |
| gpt2_ffn_up_16x3072x768 | paired_conservative | 59.486 | 0.000 | 1 / 1 |
| dense_fallback_160x320x512 | serial | 3.558 | 0.000 | 1 / 1 |
| dense_fallback_160x320x512 | paired | 3.557 | 0.000 | 1 / 1 |
| dense_fallback_160x320x512 | paired_conservative | 3.550 | 0.000 | 1 / 1 |

Execution counter times (ms; medians of seven baseline steady samples, except `act_quant_s` from the separate three-repeat profile pass):

| Case | Route | `act_quant_s` | `scale_s` | `issue_s` | `fence_s` | `pipeline_wall_s` |
|---|---|---:|---:|---:|---:|---:|
| small_48x96x256 | serial | 10.337 | 0.152 | 0.028 | 0.001 | 20.498 |
| small_48x96x256 | paired | 10.489 | 2.373 | 0.252 | 0.001 | 23.364 |
| small_48x96x256 | paired_conservative | 10.375 | 2.407 | 0.256 | 0.023 | 23.432 |
| decode_1x768x768 | serial | 1.156 | 3.141 | 0.195 | 0.007 | 8.789 |
| decode_1x768x768 | paired | 1.176 | 33.326 | 1.259 | 0.002 | 40.924 |
| decode_1x768x768 | paired_conservative | 1.175 | 33.891 | 1.272 | 0.159 | 41.782 |
| gpt2_ffn_up_16x3072x768 | serial | 7.410 | 14.201 | 1.311 | 0.030 | 53.185 |
| gpt2_ffn_up_16x3072x768 | paired | 7.441 | 159.183 | 7.139 | 0.002 | 209.153 |
| gpt2_ffn_up_16x3072x768 | paired_conservative | 7.444 | 158.901 | 7.114 | 0.712 | 210.185 |
| dense_fallback_160x320x512 | serial | 53.627 | 2.968 | 0.823 | 0.017 | 139.071 |
| dense_fallback_160x320x512 | paired | 53.716 | 2.968 | 0.835 | 0.016 | 223.654 |
| dense_fallback_160x320x512 | paired_conservative | 53.605 | 2.937 | 0.829 | 0.016 | 223.074 |

Command counts (median per steady graph; paired hardware command counters are zero on the serial or fallback route):

| Case | Route | Scale writes/releases | Serial Main/Residual LOOP_WS | Paired Main/Residual LOOP_WS | Index config/write | Paired readback/fence/drain | Fallback stripes | Scale µs/command |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| small_48x96x256 | serial | 1536/1536 | 1/1 | 0/0 | 0/0 | 0/0/0 | 0 | 0.049 |
| small_48x96x256 | paired | 2304/2304 | 0/0 | 1/8 | 8/256 | 2/1/10 | 0 | 0.515 |
| small_48x96x256 | paired_conservative | 2304/2304 | 0/0 | 1/8 | 8/256 | 2/20/29 | 0 | 0.522 |
| decode_1x768x768 | serial | 33024/33024 | 6/4 | 0/0 | 0/0 | 0/0/0 | 0 | 0.048 |
| decode_1x768x768 | paired | 33024/33024 | 0/0 | 12/60 | 60/978 | 12/1/73 | 0 | 0.505 |
| decode_1x768x768 | paired_conservative | 33024/33024 | 0/0 | 12/60 | 60/978 | 12/145/217 | 0 | 0.513 |
| gpt2_ffn_up_16x3072x768 | serial | 147456/147456 | 22/22 | 0/0 | 0/0 | 0/0/0 | 0 | 0.048 |
| gpt2_ffn_up_16x3072x768 | paired | 147456/147456 | 0/0 | 44/264 | 264/6314 | 44/1/309 | 0 | 0.540 |
| gpt2_ffn_up_16x3072x768 | paired_conservative | 147456/147456 | 0/0 | 44/264 | 264/6314 | 44/617/925 | 0 | 0.539 |
| dense_fallback_160x320x512 | serial | 30720/30720 | 8/16 | 0/0 | 0/0 | 0/0/0 | 0 | 0.048 |
| dense_fallback_160x320x512 | paired | 30720/30720 | 8/16 | 0/0 | 0/0 | 0/0/0 | 2 | 0.048 |
| dense_fallback_160x320x512 | paired_conservative | 30720/30720 | 8/16 | 0/0 | 0/0 | 0/0/0 | 2 | 0.048 |

`producer_capacity_waits` was 0 in every baseline steady sample. For the fallback case, both Paired selections executed two serial stripes: serial Main/Residual LOOP_WS counts were 8/16, with zero paired command counts. Serial main/residual fences and paired readback tiles are retained in the `POTAL_BENCH_STATS` output. The `act_quant_s` call timer includes any producer capacity wait; it is a CPU-observed interval, not a separate accelerator stage.


Paired operation-group profile (ms; median of steady iterations 2 and 3 of the separate `POTAL_PAIRED_PROFILE=1` pass; sums over stripes within each graph):

| Case | Route | Route plan | Validate | Pack | Scale Write | Scale Release | Index | Main issue | Residual issue | Readback | Fence |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| small_48x96x256 | paired | 0.268 | 0.131 | 0.405 | 1.333 | 1.257 | 0.171 | 0.020 | 0.059 | 0.019 | 0.002 |
| small_48x96x256 | paired_conservative | 0.286 | 0.133 | 0.404 | 1.321 | 1.270 | 0.174 | 0.019 | 0.059 | 0.019 | 0.025 |
| decode_1x768x768 | paired | 1.522 | 0.887 | 0.059 | 19.125 | 16.888 | 0.702 | 0.204 | 0.364 | 0.059 | 0.002 |
| decode_1x768x768 | paired_conservative | 1.474 | 0.903 | 0.034 | 19.788 | 18.325 | 0.730 | 0.208 | 0.378 | 0.061 | 0.172 |
| gpt2_ffn_up_16x3072x768 | paired | 8.232 | 5.582 | 0.239 | 90.331 | 83.883 | 4.633 | 0.915 | 1.784 | 0.292 | 0.002 |
| gpt2_ffn_up_16x3072x768 | paired_conservative | 8.285 | 5.590 | 0.224 | 89.520 | 83.975 | 4.684 | 0.911 | 1.727 | 0.280 | 0.741 |
| dense_fallback_160x320x512 | paired | 0.108 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dense_fallback_160x320x512 | paired_conservative | 0.107 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

Route plan is `potal_paired_plan_stripe` before route choice; Validate is paired executor request validation and execution-plan validation. The fallback route never enters the paired executor; both of its plan attempts sum to about 0.108 ms per graph. Stage profile output is printed once per paired stripe. The extra profile logging perturbs its wall time, so baseline ratios use the unprofiled seven-repeat pass.

The final fallback-only profile separated Residual decode from the route plan. Each graph had two refused stripes (`RESIDUAL_ACC_CAPACITY`). Across steady iterations 2 and 3, Paired RTL spent **83.102 ms decoding** and **0.108 ms planning** in total; Conservative spent **82.668 ms decoding** and **0.109 ms planning**. The baseline Paired extra time was 84.649 ms. This identifies repeated CPU decode as the fallback slowdown. Both routes issued the same serial GEMMs, and neither issued paired commands, so these measurements do not implicate paired hardware in this case.

Largest measured bottlenecks, ranked by impact:

1. **Paired Scale Write/Release overhead, especially FFN.** Baseline FFN `scale_s` was 159.183 ms Paired versus 14.201 ms Serial: a 144.982 ms gap within a 156.014 ms compute gap. The profiled Paired groups measured 90.331 ms Write and 83.883 ms Release. Both paths used 147,456 writes and 147,456 releases; observed scale latency was 0.540 versus 0.048 µs per command. Decode showed the same pattern: 33.326 versus 3.141 ms for equal scale command counts. Scale Release may block on hardware completion; these numbers do not isolate CPU validation from hardware wait.
2. **Fallback Residual decode.** The final separate profile measured 83.102 ms on two refused stripes. The attempted plans cost only 0.108 ms, and baseline Paired and Serial hardware command counts matched. The 84.649 ms compute gap is consistent with duplicate CPU preparation before serial fallback.
3. **Paired FFN planning and Index work.** The profiled FFN route plan took 8.232 ms, executor validation 5.582 ms, and Index Config/Write 4.633 ms across 264 configs and 6,314 writes. These intervals may overlap other pipeline work and are smaller than Scale overhead.

Proposed H3-6B optimization: first reduce per-entry overhead in the paired Scale Write/Release software path by validating a whole scale window once, then issuing the same C11 commands in the same order. Preserve live-slot, generation, and release ordering checks. The main correctness risk is a stale or prematurely released carrier corrupting Indexed Residual or Main INT32 and thus final F32. Re-run the paired Host execution and failure tests plus FireSim bitwise Golden checks. Repeat the same seven-sample benchmark; success requires lower Write/Release stage time and `scale_s` per command, lower Paired FFN/decode median `compute_ms`, unchanged command counts, and no new failed nodes. The current FFN 144.982 ms scale gap is an upper bound on the possible graph-level gain, because stages overlap.

Validation: six affected Host tests (`potal-paired-plan-test`, `potal-paired-schedule-test`, `potal-paired-unit-test`, `potal-paired-auto-test`, `potal-paired-execution-test`, `potal-stripe-pipeline-test`) passed. The DIM16 RISC-V `potal-paired-smoke` target built. Final FireSim `--quick` passed bitwise with zero failed nodes. The four seven-repeat baseline cases and the four separate three-repeat profile cases each reported `POTAL_BENCH PASS`; the final fallback decode profile also passed. No FPGA rebuild or full RISC-V model suite ran.

Limits: `act_quant_s` in the final build measures the ExSIA call, including any producer wait. The baseline build's pipeline `act_quant_s` remained zero, so activation timing is drawn from the separate profile pass. Operation issue times are CPU-observed latency, not accelerator execution time. The original 16 GiB guest image was backed up at `/tmp/h3-6a-firesim-backup.img` before the offline binary update; the image and staging copy passed `e2fsck -fn`. No commit or push was made.
