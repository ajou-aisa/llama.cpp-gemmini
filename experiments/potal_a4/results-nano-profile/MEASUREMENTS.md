# Attention limb and CPU component measurements

Original selective-linear+a4nks policy; GPT-2, first WikiText-2 test chunk at each context.
Counts cover all 12 layers × 12 heads. Timing covers layers 0/5/11 × heads 0/6/11, one warmup and five measured replays per head.
Times are measured NumPy int64 CPU component work, not Gemmini time or full model latency. QK and PV integer sums are checked against reconstructed float64 operands.
Reference-only FP QK/mask, reconstruction, unpacking, capture, checks and Python driver overhead are excluded from the component denominator.

## Context 512

| Operand | values | lane 1 nonzeros | lane 2 nonzeros | lane 1 rows | lane 2 rows | upper packet byte utilization |
|---|---:|---:|---:|---:|---:|---:|
| Q | 4,718,592 | 90.355% | 15.772% | 100.00% | 99.72% | 53.063% |
| P | 37,748,736 | 0.964% | 0.063% | 96.82% | 29.83% | 1.692% |

Q: 1152 stripes; lane 1/2 present in 1152/1152 stripes. Values with 0/1/2 nonzero upper digits: 9.097% / 75.679% / 15.224%.
P: 576 stripes; lane 1/2 present in 576/576 stripes. Values with 0/1/2 nonzero upper digits: 99.036% / 0.901% / 0.063%.
K always executes two dense passes; low/high nonzero values: 93.736% / 75.803%.
V executes one pass; nonzero values: 83.204%.


Q top-exponent selection: 9.233%; sigma selection: 3.547%; combined: 12.780%.
P/K/V have no outlier-selection stage in the supplied policy.

| Fragment category (exclusive) | fragments | fraction of all QK+PV |
|---|---:|---:|
| QK low main | 73,728 | 13.15% |
| Q upper / K low | 147,456 | 26.29% |
| K high main | 73,728 | 13.15% |
| Q upper / K high | 147,456 | 26.29% |
| PV main | 73,728 | 13.15% |
| P upper | 44,724 | 7.97% |

QK/main = 6.000; PV/main = 1.607; combined / two single-pass mains = 3.803.
Upper-limb compensation alone = 60.56% of fragments. Including the K-high main = 73.71%.

| CPU component | median batch mean ms/head | fraction of timed components |
|---|---:|---:|
| Q exponent/top + preliminary rounding | 1.7411 | 0.47% |
| Q sigma selection | 0.9848 | 0.26% |
| Q final rounding + folding | 3.1491 | 0.85% |
| Q/P digit packet preparation | 33.6841 | 9.04% |
| K/V/P range + rounding | 7.7544 | 2.07% |
| QK main, K low | 44.5134 | 11.94% |
| Q upper compensation, both K passes | 197.1884 | 53.00% |
| K high main | 44.9201 | 12.03% |
| PV main | 23.8569 | 6.38% |
| P upper compensation | 8.5215 | 2.27% |
| P zero point: colsum + broadcast | 0.4229 | 0.11% |
| Softmax + accumulator/publication | 6.5380 | 1.75% |

Component total mean/head: median 373.4442 ms; repeat p10..p90 [371.0030, 374.4705] ms.
Inclusive Q selection: 2.7176 ms, 0.73% of components, 5.74% of preparation.
Packet creation: 71.16% of preparation.

