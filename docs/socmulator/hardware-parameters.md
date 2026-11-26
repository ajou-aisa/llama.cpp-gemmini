# Phase 0: hardware parameters and software correspondence

## Evidence convention

Root prefixes `G`, `C`, `P`, `W`, `H`, `S`, `L` expand exactly as defined in `isa-contract.md`. `CONFIRMED` is source confirmation, `DERIVED` is evaluation of a cited source formula, and `UNVERIFIED` identifies absent proof. No RTL build, simulation, or physical hardware measurement was performed. Common `NPU_*` names below are proposed audit vocabulary, not edits to implementation or declarations that those macros already exist.

## Confirmed configurations

Each architecture is independently traced. Gemmini D16 uses `G/src/main/scala/gemmini/Configs.scala:21-66`; selected custom baseline inherits it, changing only training convolution support (`G/src/main/scala/gemmini/CustomConfigs.scala:10-18,52-54`). PoTAL INT8 D16 is explicit, D32/D64 are allowed copies changing mesh size while retaining capacities (`P/PoTalConfigs.scala:23-71,126-143`). INT4 D32 explicitly changes input width and uses CapacityInMatrices256 (`W/PoTalConfigs.scala:142-147`). **No Gemmini D32/D64 hardware values are inferred from D16.**

### Numeric geometry

SPAD rows are **per bank**; ACC_ROWS is **total across banks**. The generator explicitly emits total ACC rows, and currently emits no ACC_BANKS macro (`G/src/main/scala/gemmini/GemminiConfigs.scala:337-341`; `P/GemminiConfigs.scala:111-121`; `W/GemminiConfigs.scala:393`).

| Architecture/config | Precision bits / D | SP banks / rows per bank | ACC banks / TOTAL rows / rows per bank | ACC bits | Evidence |
|---|---:|---:|---:|---:|---|
| Gemmini default/baseline INT8 D16 | 8 / 16 | 4 / 4096 | **2 / 1024 / 512** | 32 | `G/src/main/scala/gemmini/Configs.scala:23-42`; formulas `G/src/main/scala/gemmini/GemminiConfigs.scala:99-106`; `S/gemmini_params.h:8-20`; comparison `G/software/gemmini-rocc-tests/include/gemmini_params.h:8-20`. |
| PoTAL INT8 D16 a8w8D16Sat32 | 8 / 16 | 4 / 4096 | 2 / 1024 / 512 | 32 | `P/PoTalConfigs.scala:30-61`; `P/GemminiConfigs.scala:110-117`; `H/gemmini_potal_params.h:8-23`. |
| PoTAL INT8 D32 a8w8D32Sat32 | 8 / 32 | 4 / 2048 | 2 / 512 / 256 | 32 | `P/PoTalConfigs.scala:30-61,126-143`; `P/GemminiConfigs.scala:110-117`; `H/gemmini_potal_d32_params.h:8-23`. |
| PoTAL INT8 D64 a8w8D64Sat32 | 8 / 64 | 4 / 1024 | 2 / 256 / 128 | 32 | `P/PoTalConfigs.scala:30-61,126-143`; `P/GemminiConfigs.scala:110-117`; `H/gemmini_potal_d64_params.h:8-23`. |
| PoTAL INT4 D32 a4w4D32Sat32 | 4 / 32 | 4 / 2048 | 2 / 512 / 256 | 32 | `W/PoTalConfigs.scala:58-61,142-147`; `W/GemminiConfigs.scala:111-120,313-320`; `H/gemmini_potal_w4a4_d32_params.h:8-27`. |

Per-bank row values are **DERIVED** from configured capacities using the cited generator formulas and independently agree with emitted headers. Input/output scalar types are not proof of input arithmetic width for packed INT4 (`W/GemminiConfigs.scala:313-319`; `H/gemmini_potal_w4a4_d32_params.h:17-27`).

### Storage, DMA, and metadata

All row-byte and replication totals marked DERIVED below use each configuration's own D and precision. `MAX_BLOCK_LEN` counts D-column blocks per move, not bytes or SRAM rows. ACC block len is clamped to at least1 even when a full ACC row exceeds64 bytes (`G/src/main/scala/gemmini/GemminiConfigs.scala:343-356`; `W/GemminiConfigs.scala:406-426`).

| Config | Input scalar container / bytes | SP row bytes / ACC row bytes (DERIVED) | DMA bus bits / max bytes | MAX_BLOCK_LEN / ACC counterpart | Index entries per buffer / physical total | Scale slots per column / physical total | ADDR_LEN |
|---|---|---:|---:|---:|---:|---:|---:|
| Gemmini INT8 D16 | int8_t / 1 | 16 / 64 | 128 / 64 | 4 / 1 | N/A PoTAL extension | N/A PoTAL extension | 32 |
| PoTAL INT8 D16 | int8_t / 1 | 16 / 64 | 128 / 64 | 4 / 1 | 32 / 64 | 256 / 4096 | 32 |
| PoTAL INT8 D32 | int8_t / 1 | 32 / 128 | 128 / 64 | 2 / 1 | 64 / 128 | 256 / 8192 | 32 |
| PoTAL INT8 D64 | int8_t / 1 | 64 / 256 | 128 / 64 | 1 / 1 | 128 / 256 | 256 / 16384 | 32 |
| PoTAL INT4 D32 | int8_t logical / 1; two INT4 elements per packed byte | **16 / 128** | 128 / 64 | **4 / 1** | 64 / 128 | 256 / 8192 | 32 |

Evidence for **every cell** in the preceding table: Gemmini `G/src/main/scala/gemmini/Configs.scala:23-42,65-66`, `G/src/main/scala/gemmini/GemminiConfigs.scala:99-106,337-356`, `S/gemmini_params.h:7-21`; PoTAL INT8 `P/PoTalConfigs.scala:30-71,110-112,126-143`, `P/GemminiConfigs.scala:110-117,197-225`, and each corresponding `H/gemmini_potal_params.h:7-24`, `H/gemmini_potal_d32_params.h:7-24`, `H/gemmini_potal_d64_params.h:7-24`; INT4 `W/PoTalConfigs.scala:142-147`, `W/GemminiConfigs.scala:313-320,393-426`, `H/gemmini_potal_w4a4_d32_params.h:7-28`. Index replication is **two independent tables**: `P/ExecuteController.scala:73-95`, `P/PoTalCommandMetadata.scala:17-19`; scales instantiate one memory of256 entries **for each output column**: `P/PoTalWriteback.scala:135-139,184-200`; INT4 uses `W/PoTalWriteback.scala:135-139`. Totals are direct multiplication, not separately named header capacities.

POTAL_INDEX_ENTRIES is capacity of **one buffer's selected K-row window**, not total capacity across both physical buffers and not a count of K tiles. The formula is DIM*indexWindowKTiles with indexWindowKTiles2; each entry stores one original K-row index as UInt32. INDEX_CONFIG limits activeRows to that per-buffer capacity. Experimental queue adds two staged windows of64 entries in its own module; those do not redefine Execute's physical tables (`P/GemminiConfigs.scala:205-218`; `P/PoTalConfigs.scala:12-14`; `P/ExecuteController.scala:83-89`; `P/PoTalScaleCommandIngress.scala:250-275`; `P/PoTalIndexControllerIntegration.scala:34-36`).

PoTAL INT8 capacity is256KiB SP and64KiB ACC. INT4 capacity is128KiB SP, not256KiB: CapacityInMatrices256 * D32 rows *16 packed bytes =131072 bytes, with64KiB ACC. These byte totals are **DERIVED** from `P/PoTalConfigs.scala:49-61`, `W/PoTalConfigs.scala:142-147`, `W/GemminiConfigs.scala:111-120,318-319`. Software container `int8_t` does not double INT4's physical SRAM width (`H/gemmini_potal_w4a4_d32_params.h:17-27`).

### Local-address widths

All configs have32-bit local words: accumulator flag31, accumulate30, full-width read29, norm28:26; garbage bit at M, data M-1:0; unused padding25:M+1. Counts below are **DERIVED from the per-config geometry above**, following `G/src/main/scala/gemmini/LocalAddr.scala:6-39`, `P/LocalAddr.scala:6-39`, `W/LocalAddr.scala:6-39`; norm width comes from `G/src/main/scala/gemmini/NormCmd.scala:8-9`.

| Config | SP address bits / bank bits / bank-row bits | ACC address bits / bank bits / bank-row bits | M / garbage bit | Valid SP rows / total ACC rows |
|---|---:|---:|---:|---|
| Gemmini INT8 D16 | 14 / 2 / 12 | 10 / 1 / 9 | 14 / bit14 | 0..16383 / 0..1023 |
| PoTAL INT8 D16 | 14 / 2 / 12 | 10 / 1 / 9 | 14 / bit14 | 0..16383 / 0..1023 |
| PoTAL INT8 D32 | 13 / 2 / 11 | 9 / 1 / 8 | 13 / bit13 | 0..8191 / 0..511 |
| PoTAL INT8 D64 | 12 / 2 / 10 | 8 / 1 / 7 | 12 / bit12 | 0..4095 / 0..255 |
| PoTAL INT4 D32 | 13 / 2 / 11 | 9 / 1 / 8 | 13 / bit13 | 0..8191 / 0..511 |

The local constructor asserts `M+metadata_width<32`; here metadata_width6, so M<=25. SP/ACC row extraction uses its own address width even when common data field is wider. Local addition requires power-of-two bank rows. These constrain legal geometry; they do not authorize truncating an out-of-range user address. DMA virtual/physical address widths are integration-specific and **UNVERIFIED as a single common constant**; ADDR_LEN32 is not the CPU address width (`G/src/main/scala/gemmini/LocalAddr.scala:26-39,46-48,67-78`; `G/src/main/scala/gemmini/LoadController.scala:33,44,102`; `G/src/main/scala/gemmini/FrontendTLB.scala:119-147`).

## Common names, units, ranges, and relationships

Each row names the existing SW macro and CMake adapter where present. Lack of an exported macro is recorded rather than filled by a software guess.

| Common name | Existing names | Unit / actually established range | RTL authority and constraints |
|---|---|---|---|
| NPU_D | DIM, POTAL_HW_DIM; POTAL_EXPECTED_DIM, GGML_GEMMINI_DIM | logical rows/cols per square tile; audited Gemmini16, PoTAL INT8 {16,32,64}, packed INT4 exactly32 | `G/src/main/scala/gemmini/Configs.scala:29-32`; `P/PoTalConfigs.scala:126-143`; `W/GemminiConfigs.scala:317`; square, power-of-two geometry required: `G/src/main/scala/gemmini/GemminiConfigs.scala:267-271`. |
| NPU_PRECISION | inputType.getWidth; POTAL_INPUT_BITS only in packed header | bits per logical signed input; audited8 or packed4, not sizeof(elem_t)*8 for packed transport | `G/src/main/scala/gemmini/Configs.scala:23`; `P/PoTalConfigs.scala:30`; `W/GemminiConfigs.scala:314-317`; `H/gemmini_potal_w4a4_d32_params.h:17,24-26`. |
| NPU_SPAD_BANKS | sp_banks, BANK_NUM, POTAL_HW_BANK_NUM | banks; all audited configs4; no universal arbitrary-bank supported range established | `G/src/main/scala/gemmini/Configs.scala:41`; `P/PoTalConfigs.scala:49`; packed requires4: `W/GemminiConfigs.scala:318`; paired SW requires two banks per pair: `L/ggml/src/ggml-potal/matmul/paired.hpp:64-76`. |
| NPU_SPAD_BANK_ROWS | sp_bank_entries, BANK_ROWS, POTAL_HW_BANK_ROWS | local rows per SP bank;4096/2048/1024 per table; powers of two and divisible by D | `G/src/main/scala/gemmini/GemminiConfigs.scala:99-103,266-271`; `P/GemminiConfigs.scala:110-114`; packed fixed2048: `W/GemminiConfigs.scala:318`. |
| NPU_ACC_BANKS | acc_banks; no generated ACC_BANKS/POTAL_HW_ACC_BANKS | banks; audited2; hardcoded2 in paired SW agrees with actual config | `G/src/main/scala/gemmini/Configs.scala:42`; `P/PoTalConfigs.scala:58`; `W/GemminiConfigs.scala:319`; consumer `L/ggml/src/ggml-potal/matmul/paired.hpp:69-72`. |
| NPU_ACC_ROWS | ACC_ROWS, POTAL_HW_ACC_ROWS | **TOTAL** rows;1024/512/256; per bank = total/NPU_ACC_BANKS; bank rows divisible by D | `G/src/main/scala/gemmini/GemminiConfigs.scala:104-106,271,341`; `P/GemminiConfigs.scala:115-117`; `W/GemminiConfigs.scala:319,393`. |
| NPU_ACC_BITS | accType.getWidth, POTAL_ACC_BITS packed only; acc_t | bits per element; audited32 signed; Sat32 requires SInt32 | `G/src/main/scala/gemmini/Configs.scala:24`; `P/AccumulatorMem.scala:71-84`; `W/GemminiConfigs.scala:316`. |
| NPU_INPUT_STORAGE_BYTES | sizeof(elem_t), packed byte buffers | C scalar bytes1; packed transport bytes per element1/2 (two nibbles/byte), not an integer scalar sizeof | `S/gemmini_params.h:17`; `H/gemmini_potal_w4a4_d32_params.h:24`; `L/ggml/src/ggml-potal-w4a4/include/potal-int4.h:8-26`. |
| NPU_SPAD_ROW_BYTES | sp_width/8; POTAL_SP_ROW_BYTES packed only | bytes per row,16/32/64; packed16 | `P/GemminiConfigs.scala:110`; `W/GemminiConfigs.scala:111,318,412`; `H/gemmini_potal_w4a4_d32_params.h:20`. |
| NPU_ACC_ROW_BYTES | D*sizeof(acc_t) | bytes per full-width row64/128/256 | `G/src/main/scala/gemmini/GemminiConfigs.scala:105`; `P/GemminiConfigs.scala:116`; `H/gemmini_potal_d32_params.h:8,23`. |
| NPU_DMA_BUS_BITS | dma_buswidth; no corresponding macro in cited headers | bits128; no general runtime-programmable range demonstrated | `G/src/main/scala/gemmini/Configs.scala:66`; `P/PoTalConfigs.scala:71`; packed fixed128 `W/GemminiConfigs.scala:320`. |
| NPU_DMA_MAX_BYTES | dma_maxbytes; MAX_BYTES, POTAL_HW_MAX_BYTES | bytes64 in all audited configs; fixed generator header MAX_BYTES64 is independent from config field | `G/src/main/scala/gemmini/Configs.scala:65`; `G/src/main/scala/gemmini/GemminiConfigs.scala:343-344`; `P/PoTalConfigs.scala:70`; `W/GemminiConfigs.scala:320,406-407`. |
| NPU_MAX_BLOCK_LEN | MAX_BLOCK_LEN, MAX_BLOCK_LEN_ACC | D-column blocks, at least1;4/2/1 for INT8 D16/32/64, packed4; ACC1 | `G/src/main/scala/gemmini/GemminiConfigs.scala:346-356`; `W/GemminiConfigs.scala:415-426`; each corresponding header17-18 or packed21-22. |
| NPU_INDEX_ENTRIES | potalIndexEntries, POTAL_INDEX_ENTRIES, POTAL_HW_INDEX_ENTRIES | K-row indices **per buffer/window**; DIM*2=32/64/128. Two buffers physical total2x. IndexEntries>=D; actual runtime activeRows1..capacity | `P/GemminiConfigs.scala:205-218`; `P/PoTalConfigs.scala:14,112`; `P/ExecuteController.scala:73-89`; `P/PoTalScaleCommandIngress.scala:250-257`. |
| NPU_SCALE_ENTRIES | potalScaleEntries, POTAL_SCALE_ENTRIES, POTAL_HW_SCALE_ENTRIES; POTAL_SCALE_SLOT_BITS | slots **per output column**, configured256 (slot bits8); RTL allows power of two2..65536, checked software fixes256 | `P/GemminiConfigs.scala:221-225`; `P/PoTalConfigs.scala:111`; `P/PoTalWriteback.scala:135-139`; `L/cmake/potal-hw-params.cmake:169-176`. |
| NPU_LOCAL_ADDR_BITS | ADDR_LEN, POTAL_HW_ADDR_LEN | bits32, with geometry-dependent M as above; metadata+M<32 | `G/src/main/scala/gemmini/LocalAddr.scala:7-31`; `L/cmake/potal-hw-params.cmake:150-153`. |
| NPU_PACKING_ABI | potalPackedInt4, POTAL_PACKING_ABI | packed INT4 ABI1, exact D32 config only; N/A for INT8 | `W/GemminiConfigs.scala:313-323,408-412`; `L/ggml/src/ggml-potal-w4a4/include/potal-int4.h:8-26`. |

The CMake integer parser accepts plain decimal macros with at most9 digits; that is a parser limit, not an RTL capacity declaration. Its positive-power-of-two bank/row validation, half-memory capacity check, and MAX_BYTES multiple-of-D check describe the INT8 adapter; do not apply them blindly to packed INT4 row sizes (`L/cmake/potal-hw-params.cmake:19-39,124-125,150-176`; packed dedicated assertions `L/ggml/src/ggml-potal-w4a4/potal.c:38-42`).

Paired execution's four SP banks and two ACC banks are real constraints of these layouts: one A/B bank per pair, two concurrent pairs, and Main/Residual ACC halves. Four SP banks are checked by paired SW; resident/observation and packed RTL enforce the exact geometry. This does not prove every possible non-paired Gemmini configuration requires4/2. Sources: `L/ggml/src/ggml-potal/matmul/paired.hpp:64-76`; `P/PoTalResidentView.scala:24-33,49-60`; `P/LoopMatmul.scala:1578-1588,1623-1629`; `W/GemminiConfigs.scala:317-319`.

## Software consumers and hardcoded policies

The source search covered DIM, BANK_NUM, BANK_ROWS, ACC_ROWS, MAX_BYTES, MAX_BLOCK_LEN and POTAL_* in the llama.cpp source/CMake tree. The following are actionable consumer locations, grouped by parameter rather than thousands of repeated loop bounds. SDK consumers are external to that tree and listed separately. Artifact/build/cache contents are outside this audit.

| Parameter/macro family | llama.cpp consumers and role | Hardware constraint versus software policy |
|---|---|---|
| DIM / POTAL_HW_DIM | `L/ggml/src/ggml-gemmini/ggml-gemmini-args.h:52-54,266-285`; `L/ggml/src/ggml-gemmini/matmul/dense.cpp:248-270`; `L/ggml/src/ggml-potal/potal.c:31,183-214,247`; `L/ggml/src/ggml-potal/matmul/execution.cpp:42,247-258` | Tile geometry is HW. INT8 backend's DIM16/32 validated scope is SW restriction, despite RTL D64 config (`L/ggml/src/ggml-potal/include/potal.h:139-143`; `P/PoTalConfigs.scala:129,142-143`). |
| BANK_NUM / BANK_ROWS / POTAL_HW_BANK_* | `L/ggml/src/ggml-potal/potal.c:33-34,210-214,391-411`; `L/ggml/src/ggml-potal/matmul/tiling.cpp:35,191`; `L/ggml/src/ggml-potal/matmul/paired-auto.cpp:108-112`; `L/ggml/src/ggml-gemmini/fp32/rocc-model.cpp:14,41,59` | Memory capacity and physical pair placement are HW. FP32 model is a different architecture, not evidence for INT8 dimensions. |
| ACC_ROWS / POTAL_HW_ACC_ROWS | `L/ggml/src/ggml-potal/potal.c:35,211-214,518,566`; `L/ggml/src/ggml-potal/matmul/tiling.cpp:39`; `L/ggml/src/ggml-potal/matmul/paired.hpp:69-72`; `L/ggml/src/ggml-gemmini/fp32/rocc-model.cpp:15,74,97` | Total/2 assumes actual two banks. Exporting NPU_ACC_BANKS would remove that hidden binding; this audit makes no code change. |
| MAX_BYTES / POTAL_HW_MAX_BYTES | `L/cmake/potal-hw-params.cmake:120-123,157-159,196`; `L/ggml/src/ggml-potal/potal.c:36,164-169`; `L/ggml/src/ggml-potal/matmul/dense.cpp:27-29`; `L/ggml/src/ggml-potal/matmul/stripe.cpp:238` | DMA maximum transfer size is HW config. Requiring every input base64-byte aligned is checked API policy; RTL DMA packet selection handles offsets, and indexed B bases are local rows, not DMA bytes (`G/src/main/scala/gemmini/DMA.scala:183-203`; `P/PoTalScaleCommandIngress.scala:250-257`). |
| MAX_BLOCK_LEN / MAX_BLOCK_LEN_ACC | Parser does not export these as POTAL_HW_*; Gemmini simulation precision adapter rewrites MAX_BLOCK_LEN for16-bit input (`L/cmake/ggml-gemmini-geometry.cmake:47-52`); external SDK consumers `S/gemmini.h:459-461` and RTL-side SDK `G/software/gemmini-rocc-tests/include/gemmini.h:458-460` | Generated transfer grouping, not an independent memory capacity. ACC clamp1 permits a full row larger than64 bytes split across DMA transactions (`W/GemminiConfigs.scala:423-426`; `G/src/main/scala/gemmini/DMA.scala:424-480`). |
| POTAL_INDEX_ENTRIES / POTAL_HW_INDEX_ENTRIES | `L/cmake/potal-hw-params.cmake:120-123,175-176,198`; `L/ggml/src/ggml-potal/potal.c:403,658-660,1851-1868`; `L/ggml/src/ggml-potal/matmul/paired-auto.cpp:112` | Capacity per buffer is HW. Queue installation's DIM32/count64/budget restrictions are explicit experimental protocol/SW scope, not universal index-table capacity. |
| POTAL_SCALE_ENTRIES / POTAL_HW_SCALE_ENTRIES / POTAL_SCALE_SLOT_BITS | `L/cmake/potal-hw-params.cmake:169-174,197`; `L/ggml/src/ggml-potal/include/potal.h:34-38`; `L/ggml/src/ggml-potal/potal.c:139-148,203-207` | Config256 and eight-bit keys agree; RTL family permits larger power-of-two caches, SW does not. POTAL_SCALE_SLOT_BITS is in generated headers (`H/gemmini_potal_d32_params.h:14`); the checked adapter uses fixed key shifts rather than exporting this macro. |
| POTAL_INPUT_BITS / POTAL_ACC_BITS / POTAL_PACKING_ABI / POTAL_SP_ROW_BYTES | `L/ggml/src/ggml-potal-w4a4/potal.c:38-42`; `L/ggml/src/ggml-potal-w4a4/include/potal-int4.h:8-26`; `L/ggml/src/ggml-potal-w4a4/matmul/execution.cpp:258-259`; `L/ggml/src/ggml-potal-w4a4/matmul/stripe.cpp:239` | Exact packed ABI requirements mirror RTL. Retained comments describing DIM int8 rows are stale transport descriptions, not authority (`L/ggml/src/ggml-potal-w4a4/matmul/paired-auto.hpp:20-37`). |
| POTAL_GENERATION_MAX / POTAL_WORK_IDS_PER_LOOP / POTAL_CARRIER_* | `L/ggml/src/ggml-potal/include/potal.h:34-38`; `L/ggml/src/ggml-potal/potal.c:135-148,178-189,212`; `L/ggml/src/ggml-potal/matmul/paired-plan.cpp:24` | Generation8 bits,64 work ids per loop,carrier0..32767/zero sentinel are protocol constraints (`P/PoTalCommandMetadata.scala:11-15`; `P/PoTalSCU.scala:5-16`). |
| POTAL_FUNCT_* / POTAL_HW_XCUSTOM / POTAL_HW_ADDR_LEN | `L/ggml/src/ggml-potal/include/potal.h:358-381`; `L/ggml/src/ggml-potal/include/potal_rocc.h:27-51`; `L/ggml/src/ggml-potal/potal.c:30-32,67-92` | ISA encoding, not tuning variables (`P/GemminiISA.scala:8-51`; `P/LocalAddr.scala:7`). |
| POTAL_HW_PARAMS / POTAL_EXPECTED_DIM / POTAL_TARGET_* / POTAL_HW_CONFIG | `L/cmake/potal-hw-params.cmake:83-111,131-149,184-199,215-239` | Build provenance/consistency checks. The header hash identifies the header, not all RTL. |

A legacy software geometry helper hardcodes D32, ACC512/2 and `spad=32768/bits`; those constants describe its own quantization profile and must **not** be promoted to RTL-confirmed Gemmini D32 parameters (`L/ggml/src/ggml-gemmini/ggml-gemmini-geometry.hpp:120-132`). Host/cycle-sim CMake can replace DIM and selected bank/row macros; a generated simulation header is not evidence of physical geometry (`L/cmake/ggml-gemmini-geometry.cmake:38-77`; `L/cmake/ggml-gemmini-im2p.cmake:29-32,130-131`).

`paired_block_k=32` is HP1 quantization policy, with divisibility assertion; it is **not** a fixed RTL scale BlockSize. RTL accepts whole-D scale blocks via passesPerBlock (`L/ggml/src/ggml-potal/matmul/paired.hpp:73-76`; `P/PoTalConfigs.scala:122-124`; `P/PoTalCommandMetadata.scala:21-37`).

## Actual Gemmini software SDK versus RTL-side header

The supplied software SDK is `RISC-V-DynDNN-gemmini-include`, develop revision `fcfcf5ce7c4670bd789365f11810177f97016f81`; local checkout is detached at that revision and origin/develop points there (`S/.git:1`; resolved gitdir `/Users/zerogod/aisa-lab/DynDNN/reference/RISC-V-DynDNN-gemmini-include/.git/worktrees/gemmini-include-develop/HEAD:1`; `/Users/zerogod/aisa-lab/DynDNN/reference/RISC-V-DynDNN-gemmini-include/.git/packed-refs:5`). The source build resolves the SDK only from explicit GEMMINI_SW_PATH, so use in a particular compiled binary remains user-reported provenance, **UNVERIFIED by compilation here** (`L/cmake/ggml-gemmini-geometry.cmake:1-25`).

All geometry/type/scaling macros in `S/gemmini_params.h` agree with `G`'s parameter header except that **S lacks ACC_READ_FULL_WIDTH**, while G defines it and RTL enables full-width reads. This is a SW feature-advertisement discrepancy, not a bank-count discrepancy. Sources: `S/gemmini_params.h:7-38,76-84`; `G/software/gemmini-rocc-tests/include/gemmini_params.h:7-38,76-85`; `G/src/main/scala/gemmini/GemminiConfigs.scala:62-63`. Core issue macro bodies agree byte-for-byte across `S/gemmini.h:247-346` and `G/software/gemmini-rocc-tests/include/gemmini.h:246-345`; loop field agreement is at `S/gemmini.h:413-421` and `G/software/gemmini-rocc-tests/include/gemmini.h:412-420`.

The expected reference geometry in the request is confirmed for every listed configuration. Additional source discrepancies/caveats: MAX_BYTES emission is hardcoded64 rather than tied to dma_maxbytes; currently those values agree. PoTAL's enablePoTalWriteback=false/stale “not yet connected” comments contradict actual Controller integration through enablePoTalSat32Acc. Gemmini CONFIG_NORM exists but default has_normalizations=false, so its presence is not safe default support. Sources: `G/src/main/scala/gemmini/GemminiConfigs.scala:50,89,343-344`; `P/PoTalConfigs.scala:20-21,110-115`; `P/Controller.scala:125-165,177-189`; `G/src/main/scala/gemmini/StoreController.scala:133-135`. Gemmini's counter hardware is XLEN-wide, whereas counter_read narrows its result to uint32_t; do not confuse that SW return type with a32-bit RTL counter (`G/src/main/scala/gemmini/Controller.scala:58-60`; `G/src/main/scala/gemmini/CounterFile.scala:159-170`; `S/gemmini.h:356-361`).

## Header SHA-256 inventory

Hashes are measurements of exact complete file bytes, not claims made by an RTL source line. Each cited byte range below identifies the measured file's parameter/ABI content; machine-readable paths, byte counts and verification statuses are repeated in `baseline/rtl.json`.

| Header | Bytes | SHA-256 | Content reference |
|---|---:|---|---|
| Actual SDK `S/gemmini_params.h` | 2686 | `d6db18f8edbfeb854780286d8c2afb81ae3b9103224522478a9047d5550272e5` | `S/gemmini_params.h:7-38,76-84` |
| Actual SDK `S/gemmini.h` | 208174 | `a6d7207ddee555f0fbcd00b9952a12507b458ee43a899c7c1e60a131af53b15f` | `S/gemmini.h:247-346,413-421` |
| RTL-side `G/software/gemmini-rocc-tests/include/gemmini_params.h` | 2714 | `3758ae967af3a179497660970201093a7fb624be00173990ce33d3f5c38da924` | `G/software/gemmini-rocc-tests/include/gemmini_params.h:7-38,76-85` |
| RTL-side `G/software/gemmini-rocc-tests/include/gemmini.h` | 204951 | `9c872cdabb865dd3aee77937f090855b51120178fc181a66f0b1cb50bbf9fcf4` | `G/software/gemmini-rocc-tests/include/gemmini.h:245-345,412-420` |
| `H/gemmini_potal_params.h` | 2526 | `b4cd8b9efef9f3cd9b598ae3cc84af6478f83e628ed73d94a77fb1c452df1fc7` | `H/gemmini_potal_params.h:7-24` |
| `H/gemmini_potal_d32_params.h` | 2507 | `3580b84f15ed8de9f9d311edea64eaa7c1f9bb308b1552a8f79453ec2c05ea33` | `H/gemmini_potal_d32_params.h:7-24` |
| `H/gemmini_potal_d64_params.h` | 2508 | `5c3b625cf21748322dab3ef32112f0edf43aee89e0c17e763c097ddda484a506` | `H/gemmini_potal_d64_params.h:7-24` |
| `H/gemmini_potal_w4a4_d32_params.h` | 2596 | `4b29d84eff0bd4a01c6db0519c79904b808ba0f2dc9bc3727cb395ca9e30eaf0` | `H/gemmini_potal_w4a4_d32_params.h:7-28` |

## Unverified boundaries

Generated netlists, active deployed images, ASIC/FPGA SRAM realization, cycle timings, actual runtime max supported problem shapes, and physical-system DMA address width are **UNVERIFIED**. Parameter tables certify the specified source/config/header pairing, not arbitrary copies with changed fields. In particular, the separate highPerfInferenceConfig changes memory capacities as well as D; it must be audited independently before adding a Gemmini D32 row (`G/src/main/scala/gemmini/CustomConfigs.scala:20-28`).
