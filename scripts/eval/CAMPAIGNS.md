# Metric campaigns

Use `run_measurement.py metric activation|residual|scu|all` for independent or combined instrumentation.
`campaign_build.py` configures internal Gemmini/PoTAL headers through CMake and records build identities.
Every metric collection executes the complete numerical forward.

The native CPU and CUDA observation commands are in `end_to_end.py collect`; token-paired
CPU cost replays remain available through `paired_inputs.py`. Native CPU PMU and timing
validation helpers live in `cpu_profiling.py`.

See README.md for generated command, dependency, build, output and script inventories.
