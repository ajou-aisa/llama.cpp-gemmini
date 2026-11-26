# PoTAL historical reports

These 36 H3 reports are snapshots, not current capability or hardware validation claims.
Phase 2C moved them without changing any bytes. Source paths, commit identities, measurements, rejected designs, negative overlap/performance results, and failed GPT-2 runs remain as originally recorded. Current boundaries are in [Phase 2C](../../socmulator/phase2/experiment-isolation.md).

All referenced /tmp, /home and /media artifacts were unavailable on this workstation on 2026-11-26. They remain external lab-host provenance, not rerun evidence. The per-report SHA-256 and exact artifact-path availability ledger is build-npu/phase2c-evidence/archive-inventory.json. No historical result is promoted to a present PASS.

| Original location | Archived report |
| --- | --- |
| `H3-6A-paired-performance-profile.md` | [H3-6A-paired-performance-profile.md](performance/H3-6A-paired-performance-profile.md) |
| `H3-6B0-scale-path-analysis.md` | [H3-6B0-scale-path-analysis.md](performance/H3-6B0-scale-path-analysis.md) |
| `H3-6B1-trace-cache-optimization.md` | [H3-6B1-trace-cache-optimization.md](performance/H3-6B1-trace-cache-optimization.md) |
| `H3-6B2-trace-cold-path-report.md` | [H3-6B2-trace-cold-path-report.md](performance/H3-6B2-trace-cold-path-report.md) |
| `H3-6B3-checked-dispatch-report.md` | [H3-6B3-checked-dispatch-report.md](performance/H3-6B3-checked-dispatch-report.md) |
| `H3-6B4-remaining-scale-overhead.md` | [H3-6B4-remaining-scale-overhead.md](performance/H3-6B4-remaining-scale-overhead.md) |
| `H3-6C-fallback-decode-reuse-report.md` | [H3-6C-fallback-decode-reuse-report.md](paired-execution/H3-6C-fallback-decode-reuse-report.md) |
| `H3-6D1-paired-planning-validation-analysis.md` | [H3-6D1-paired-planning-validation-analysis.md](paired-execution/H3-6D1-paired-planning-validation-analysis.md) |
| `H3-6E-paired-dataflow-scheduling-analysis.md` | [H3-6E-paired-dataflow-scheduling-analysis.md](paired-execution/H3-6E-paired-dataflow-scheduling-analysis.md) |
| `H3-6E3-microtile-pipeline-feasibility.md` | [H3-6E3-microtile-pipeline-feasibility.md](paired-execution/H3-6E3-microtile-pipeline-feasibility.md) |
| `H3-6E3-paired-bottleneck-isolation.md` | [H3-6E3-paired-bottleneck-isolation.md](performance/H3-6E3-paired-bottleneck-isolation.md) |
| `H3-6E4A-resident-b-microtile-interface.md` | [H3-6E4A-resident-b-microtile-interface.md](resident-weight/H3-6E4A-resident-b-microtile-interface.md) |
| `H3-6E4B1-general-resident-microtile-contract.md` | [H3-6E4B1-general-resident-microtile-contract.md](resident-weight/H3-6E4B1-general-resident-microtile-contract.md) |
| `H3-6E4B2-resident-microtile-rtl-isa-implementation.md` | [H3-6E4B2-resident-microtile-rtl-isa-implementation.md](resident-weight/H3-6E4B2-resident-microtile-rtl-isa-implementation.md) |
| `H3-6E4C1-sequential-paired-microtile-integration.md` | [H3-6E4C1-sequential-paired-microtile-integration.md](paired-execution/H3-6E4C1-sequential-paired-microtile-integration.md) |
| `H3-6E4C2-0-prefetch-admission-completion-feasibility.md` | [H3-6E4C2-0-prefetch-admission-completion-feasibility.md](prefetch/H3-6E4C2-0-prefetch-admission-completion-feasibility.md) |
| `H3-7-GPT2-paired-attention-e2e-report.md` | [H3-7-GPT2-paired-attention-e2e-report.md](attention/H3-7-GPT2-paired-attention-e2e-report.md) |
| `H3-7A-llama-full-inference-baseline.md` | [H3-7A-llama-full-inference-baseline.md](attention/H3-7A-llama-full-inference-baseline.md) |
| `H3-7B-potal-attention-offload-implementation.md` | [H3-7B-potal-attention-offload-implementation.md](attention/H3-7B-potal-attention-offload-implementation.md) |
| `H3-7B5-qk-host-int64-limb-reconstruction.md` | [H3-7B5-qk-host-int64-limb-reconstruction.md](attention/H3-7B5-qk-host-int64-limb-reconstruction.md) |
| `H3-7C-attention-pv-range-resolution-report.md` | [H3-7C-attention-pv-range-resolution-report.md](attention/H3-7C-attention-pv-range-resolution-report.md) |
| `H3-8A-potal-target-build-profile.md` | [H3-8A-potal-target-build-profile.md](performance/H3-8A-potal-target-build-profile.md) |
| `H3-8B-dim32-port-validation.md` | [H3-8B-dim32-port-validation.md](performance/H3-8B-dim32-port-validation.md) |
| `H3-8C-dim32-firesim-e2e-report.md` | [H3-8C-dim32-firesim-e2e-report.md](performance/H3-8C-dim32-firesim-e2e-report.md) |
| `H3-8D-e4c1-e4c2-implementation-plan.md` | [H3-8D-e4c1-e4c2-implementation-plan.md](paired-execution/H3-8D-e4c1-e4c2-implementation-plan.md) |
| `H3-8D-final-report.md` | [H3-8D-final-report.md](performance/H3-8D-final-report.md) |
| `H3-8D0-matched-performance-baseline.md` | [H3-8D0-matched-performance-baseline.md](performance/H3-8D0-matched-performance-baseline.md) |
| `H3-8D1-ktile-paired-implementation.md` | [H3-8D1-ktile-paired-implementation.md](paired-execution/H3-8D1-ktile-paired-implementation.md) |
| `H3-8D2-next-k-prefetch-implementation.md` | [H3-8D2-next-k-prefetch-implementation.md](prefetch/H3-8D2-next-k-prefetch-implementation.md) |
| `H3-8E-potal-pipeline-full-audit-and-optimization-plan.md` | [H3-8E-potal-pipeline-full-audit-and-optimization-plan.md](performance/H3-8E-potal-pipeline-full-audit-and-optimization-plan.md) |
| `H3-8F0-original-gemmini-prefetch-audit.md` | [H3-8F0-original-gemmini-prefetch-audit.md](prefetch/H3-8F0-original-gemmini-prefetch-audit.md) |
| `H3-8F1-managed-completion-optimization.md` | [H3-8F1-managed-completion-optimization.md](performance/H3-8F1-managed-completion-optimization.md) |
| `H3-8FG-performance-comparison.md` | [H3-8FG-performance-comparison.md](performance/H3-8FG-performance-comparison.md) |
| `H3-8G-gemmini-based-early-prefetch-implementation.md` | [H3-8G-gemmini-based-early-prefetch-implementation.md](prefetch/H3-8G-gemmini-based-early-prefetch-implementation.md) |
| `H3-8H-paired-streaming-implementation.md` | [H3-8H-paired-streaming-implementation.md](prefetch/H3-8H-paired-streaming-implementation.md) |
| `H3-8I-25-checkpoint-evidence.md` | [H3-8I-25-checkpoint-evidence.md](performance/H3-8I-25-checkpoint-evidence.md) |

Live paired software and automatic-execution contract documents remain under each backend docs/ directory. Their current source and test consumers are recorded in the Phase 2C PoTAL experiment document.
