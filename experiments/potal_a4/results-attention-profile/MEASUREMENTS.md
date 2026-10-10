# Attention limb and CPU component measurements

Original selective-linear+a4nks policy; GPT-2, first WikiText-2 test chunk at each context.
Counts cover all 12 layers × 12 heads. Timing covers layers 0/5/11 × heads 0/6/11, one warmup and five measured replays per head.
Times are measured NumPy int64 CPU component work, not Gemmini time or full model latency. QK and PV integer sums are checked against reconstructed float64 operands.
Reference-only FP QK/mask, reconstruction, unpacking, capture, checks and Python driver overhead are excluded from the component denominator.

## Context 128

| Operand | values | lane 1 nonzeros | lane 2 nonzeros | lane 1 rows | lane 2 rows | upper packet byte utilization |
|---|---:|---:|---:|---:|---:|---:|
| Q | 1,179,648 | 90.669% | 15.389% | 100.00% | 99.22% | 53.029% |
| P | 2,359,296 | 1.117% | 0.016% | 92.00% | 2.01% | 1.107% |

Q: 288 stripes; lane 1/2 present in 288/288 stripes. Values with 0/1/2 nonzero upper digits: 8.809% / 76.323% / 14.868%.
P: 144 stripes; lane 1/2 present in 144/144 stripes. Values with 0/1/2 nonzero upper digits: 98.883% / 1.101% / 0.016%.
K always executes two dense passes; low/high nonzero values: 93.701% / 74.526%.
V executes one pass; nonzero values: 82.773%.


Q top-exponent selection: 9.900%; sigma selection: 3.275%; combined: 13.175%.
P/K/V have no outlier-selection stage in the supplied policy.

| Fragment category (exclusive) | fragments | fraction of all QK+PV |
|---|---:|---:|
| QK low main | 4,608 | 12.46% |
| Q upper / K low | 9,216 | 24.93% |
| K high main | 4,608 | 12.46% |
| Q upper / K high | 9,216 | 24.93% |
| PV main | 4,608 | 12.46% |
| P upper | 4,716 | 12.76% |

QK/main = 6.000; PV/main = 2.023; combined / two single-pass mains = 4.012.
Upper-limb compensation alone = 62.61% of fragments. Including the K-high main = 75.07%.

| CPU component | median batch mean ms/head | fraction of timed components |
|---|---:|---:|
| Q exponent/top + preliminary rounding | 0.1410 | 2.08% |
| Q sigma selection | 0.0861 | 1.27% |
| Q final rounding + folding | 0.1798 | 2.66% |
| Q/P digit packet preparation | 0.9424 | 13.72% |
| K/V/P range + rounding | 0.3555 | 5.23% |
| QK main, K low | 0.6480 | 9.48% |
| Q upper compensation, both K passes | 2.7770 | 40.90% |
| K high main | 0.6378 | 9.39% |
| PV main | 0.6451 | 9.46% |
| P upper compensation | 0.2341 | 3.44% |
| P zero point: colsum + broadcast | 0.0310 | 0.45% |
| Softmax + accumulator/publication | 0.1315 | 1.93% |

Component total mean/head: median 6.7896 ms; repeat p10..p90 [6.7769, 6.8825] ms.
Inclusive Q selection: 0.2278 ms, 3.36% of components, 13.56% of preparation.
Packet creation: 55.27% of preparation.

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
| Q exponent/top + preliminary rounding | 0.4408 | 0.40% |
| Q sigma selection | 0.2694 | 0.25% |
| Q final rounding + folding | 0.6769 | 0.62% |
| Q/P digit packet preparation | 7.2890 | 6.66% |
| K/V/P range + rounding | 1.7832 | 1.62% |
| QK main, K low | 15.6483 | 14.28% |
| Q upper compensation, both K passes | 52.8940 | 48.11% |
| K high main | 15.7357 | 14.24% |
| PV main | 10.2995 | 9.38% |
| P upper compensation | 3.0090 | 2.74% |
| P zero point: colsum + broadcast | 0.1155 | 0.11% |
| Softmax + accumulator/publication | 1.5439 | 1.41% |

Component total mean/head: median 109.4844 ms; repeat p10..p90 [107.4881, 111.6011] ms.
Inclusive Q selection: 0.7097 ms, 0.65% of components, 6.77% of preparation.
Packet creation: 69.82% of preparation.

## Context 1024

| Operand | values | lane 1 nonzeros | lane 2 nonzeros | lane 1 rows | lane 2 rows | upper packet byte utilization |
|---|---:|---:|---:|---:|---:|---:|
| Q | 9,437,184 | 90.093% | 15.691% | 100.00% | 99.83% | 52.892% |
| P | 150,994,944 | 0.532% | 0.032% | 96.53% | 31.26% | 1.483% |

Q: 2304 stripes; lane 1/2 present in 2304/2304 stripes. Values with 0/1/2 nonzero upper digits: 9.343% / 75.531% / 15.127%.
P: 1152 stripes; lane 1/2 present in 1152/1152 stripes. Values with 0/1/2 nonzero upper digits: 99.468% / 0.500% / 0.032%.
K always executes two dense passes; low/high nonzero values: 93.739% / 77.039%.
V executes one pass; nonzero values: 83.300%.


Q top-exponent selection: 8.976%; sigma selection: 3.690%; combined: 12.666%.
P/K/V have no outlier-selection stage in the supplied policy.

| Fragment category (exclusive) | fragments | fraction of all QK+PV |
|---|---:|---:|
| QK low main | 294,912 | 13.55% |
| Q upper / K low | 589,824 | 27.10% |
| K high main | 294,912 | 13.55% |
| Q upper / K high | 589,824 | 27.10% |
| PV main | 294,912 | 13.55% |
| P upper | 112,026 | 5.15% |

QK/main = 6.000; PV/main = 1.380; combined / two single-pass mains = 3.690.
Upper-limb compensation alone = 59.35% of fragments. Including the K-high main = 72.90%.

| CPU component | median batch mean ms/head | fraction of timed components |
|---|---:|---:|
| Q exponent/top + preliminary rounding | 0.8105 | 0.09% |
| Q sigma selection | 0.4953 | 0.05% |
| Q final rounding + folding | 1.3499 | 0.14% |
| Q/P digit packet preparation | 21.9895 | 2.33% |
| K/V/P range + rounding | 5.6083 | 0.59% |
| QK main, K low | 145.9977 | 15.50% |
| Q upper compensation, both K passes | 564.9892 | 59.92% |
| K high main | 145.3414 | 15.55% |
| PV main | 41.0189 | 4.35% |
| P upper compensation | 7.7601 | 0.83% |
| P zero point: colsum + broadcast | 0.2256 | 0.02% |
| Softmax + accumulator/publication | 5.9406 | 0.63% |

Component total mean/head: median 942.3919 ms; repeat p10..p90 [933.6378, 944.1772] ms.
Inclusive Q selection: 1.3057 ms, 0.14% of components, 4.32% of preparation.
Packet creation: 72.66% of preparation.
