# Gemmini SDK provenance

Until Phase 1 the Gemmini backend compiled against an external SDK header
passed with `-DGEMMINI_SW_PATH`. The header that matches this tree is
`gemmini.h` of `RISC-V-DynDNN-gemmini-include`, branch `develop` (= `m-r`):

| Item | Value |
|---|---|
| Repository | https://github.com/ajou-aisa/RISC-V-DynDNN-gemmini-include |
| Commit | `fcfcf5ce7c4670bd789365f11810177f97016f81` (2026-10-06) |
| `gemmini.h` | 208174 bytes, sha256 `a6d7207ddee555f0fbcd00b9952a12507b458ee43a899c7c1e60a131af53b15f`, git blob `cd15b5479a59c62e549e14103ae2386440ffeb60` |
| `gemmini_params.h` | sha256 `d6db18f8edbfeb854780286d8c2afb81ae3b9103224522478a9047d5550272e5`, blob `768ebf218afc60f5f28c2ecf378bd03eb96001fd` |
| `gemmini_counter.h` | blob `6050ed742186dda54e05263422c0b40f9e76bb03` |
| Upstream | gemmini-rocc-tests (`gemmini-aisa` `185293e8`, `software/gemmini-rocc-tests/include`) |
| License | BSD-3-Clause: `LICENSE.gemmini-rocc-tests` (Boston University), `LICENSE.gemmini` (UC Regents) |

The develop repository carries no LICENSE file of its own; the two upstream
licenses are kept here verbatim.


Derived code: `baseline.cpp`, `block-quantized.cpp`, `execution.cpp`,
`tiling.cpp`, `reference.hpp`, and `../reference/cpu.cpp`.
The original arithmetic and WS tiling order are preserved; command encoders
and transports live in `../api/`. Phase 2A caller and verification records
are in `docs/socmulator/phase2/gemmini-legacy-removal.md`.
