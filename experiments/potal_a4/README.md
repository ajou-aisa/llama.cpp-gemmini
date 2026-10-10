# PoTal A4 제한 조건 실험

Nano / Gemmini IM2P에서 이어갈 때는 저장소 루트의 [POTAL_NANO_HANDOFF.md](../../POTAL_NANO_HANDOFF.md)를 먼저 읽는다. [원래 알고리즘](ORIGINAL_SPEC.md)과 [전체 구현 계획](IMPLEMENTATION_PLAN.md)도 이 브랜치에 보관했다.

이 디렉터리는 `hotfix/potal-attn`의 **실험 구현**이다. CPU에서 선형·Q direct-limb 정책과 K/P/V 양자화를 적용한 GPT-2 forward를 실행하고, 저장 버퍼와 정확도를 비교한다. 제품의 Gemmini attention dispatch와 fused upper-digit API는 아직 연결하지 않았다.

실측 결론은 [RESULTS.md](RESULTS.md), 조건별 원자료는 [results-v3](results-v3/)에 있다.

원문 정책에서 attention의 limb 수, 이상치 선택, 패킷 생성, integer 보정 실행을 분리한 후속 실측은 [ATTENTION_PROFILE.md](ATTENTION_PROFILE.md)에 있다.

최신 채택 조건은 **처음 제공된 selective-linear+a4nks 정책 대비 정확도 저하 금지**다. 원래 선형 corr/sat, Q 전체 보정, P 8비트와 원래 stripe, K/V 정책을 유지한다. Direct-all 선형, limb 제한, P 8행 분할은 채택하지 않은 실험으로 남긴다. 저장 최적화는 원래 유효 코드·scale·복원값을 정확히 보존하고, 동일 실행 조건의 logits/NLL/PPL까지 같아야 통과한다. 아래 측정은 이 제품 통과 조건을 완료한 증거가 아니다.

## 비교 범위

- 고정: n=4, BK=DIM=32, tau=2, distinct E2, 참조 stripe geometry `(4,4096,512)`.
- 선형: 체크포인트에 실제 Q4_HP1으로 저장된 48개 행렬만 입력을 양자화한다. FP16 embedding과 tied output head, normalization과 bias는 보존한다.
- 기준: 동일 Q4 체크포인트의 activation FP32, 원문 selective-linear+a4nks, direct-linear+a4nks. Direct와 selective의 차이를 develop 전체 구현의 차이로 해석하지 않는다.
- 제한: P 4/6/8 bits, P scale 공유 범위 geometry/32/8/1행, 상위 limb 0/1/8개. Limb 제한은 상위 digit을 단순히 버리지 않고 표현 가능 범위로 folded code를 포화시킨다.
- Q/K/V 양자화 결과로 QK를 다시 계산하고 causal softmax를 적용한다. 원본 P를 재사용하지 않는다.
- 검증: WikiText-2 valid 앞 512 tokens로 12조건을 비교한다. 사전에 고른 8조건을 별도 test 앞 4096 tokens, context=512에서 평가한다. 각 chunk 뒤 절반의 next-token 255개, 총 2,040개를 채점한다. 추가로 context=128/1024를 비교한다.

## 측정값의 의미

Python 경로는 실제 int8 main/compact digit 배열을 할당하며 `nbytes`를 합산한다. Row/lane·K index와 scale도 별도로 센다. 이것은 실험 packet의 배열 크기이며, C++ 객체 전체의 allocator 비용을 뜻하지 않는다. Model forward는 복원한 FP32 operand의 matmul이다. Fragment는 DIM 패딩 공식으로 산출한 작업량이고 accelerator 실행 횟수나 지연시간 실측이 아니다. P scale을 더 작은 행 묶음으로 바꾸면 main의 패딩도 늘어난다. 모든 비교에서 같은 원래 main 작업량을 분모로 사용한다.

`native_probe.cpp`는 기존 C++ `QuantizedActivationBuffer`, balanced decomposition, `RmdBitmapBuilder`를 실제로 실행한다. 동일한 bounded folded code에 대해 clipped main+완전 보정+dense int32 plane과 low-digit main+plane 생략을 비교한다. Stripe=32, rows=512, K=4096; 각 경우 새 프로세스로 3회 실행한다. 유지 payload/metadata와 process peak RSS를 별도로 기록한다. FP32 원본·weight·KV·전체 모델은 이 microbenchmark에 없다. 생성된 Gemmini header의 DIM32를 사용하지만 RTL이나 실제 GEMM을 실행하지 않는다.

Native probe에는 기존 scalar emitter로 연결하는 bridge가 남아 있다. `u-d0`가 int32를 벗어나면 명시적으로 실패한다. 따라서 전체 int32 범위의 fused emitter 완성이나 준비시간 최적화를 주장하지 않는다. Python packet roundtrip은 INT32_MIN/MAX와 9-digit carry까지 별도로 검사한다.

## 재현

작업 디렉터리: `/Users/chan/Projects/AISA/worktrees/potal-attn`. Python은 기존 환경을 사용한다. 결과 경로는 새 디렉터리여야 하며 기존 결과를 덮어쓰거나 완료된 것처럼 재사용하지 않는다.

```sh
rtk proxy /Users/chan/Projects/AISA/llama.cpp-gemmini/.venv-convert311/bin/python -m experiments.potal_a4.verify
rtk proxy /Users/chan/Projects/AISA/llama.cpp-gemmini/.venv-convert311/bin/python -m experiments.potal_a4.prepare /Users/chan/Projects/AISA/llama.cpp-gemmini experiments/potal_a4/results-new
rtk proxy env PYTHONPATH=gguf-py:. OPENBLAS_NUM_THREADS=2 /Users/chan/Projects/AISA/llama.cpp-gemmini/.venv-convert311/bin/python -m experiments.potal_a4.matrix /Users/chan/Projects/AISA/llama.cpp-gemmini experiments/potal_a4/results-new all
```

`matrix`에 `valid`, `holdout`, `context` 인자를 주면 각 단계를 따로 실행할 수 있다.

Native parameter header 생성과 probe 빌드:

```sh
rtk proxy cmake -S . -B build-potal-lab -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_COMMON=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DGGML_GEMMINI=ON -DGGML_METAL=OFF -DGGML_CUDA=OFF -DGGML_BACKEND_DL=OFF -DGGML_GEMMINI_OPTION=CPU -DGGML_GEMMINI_EXECUTION_BACKEND=HARDWARE -DGGML_GEMMINI_ACTIVATION_BITS=4 -DGGML_GEMMINI_WEIGHT_BITS=4 -DGGML_GEMMINI_DIM=32 -DGGML_GEMMINI_BLOCK_SIZE=32 -DGEMMINI_SW_PATH=/Users/chan/Projects/AISA/RISC-V-DynDNN-gemmini-include
rtk proxy cmake -S experiments/potal_a4 -B build-potal-probes -DCMAKE_BUILD_TYPE=Release -DLLAMA_LIB_DIR=/Users/chan/Projects/AISA/llama.cpp-gemmini/build-cpu-only-audit/bin
rtk proxy cmake --build build-potal-probes -j 2
rtk proxy ctest --test-dir build-potal-probes --output-on-failure
rtk proxy /Users/chan/Projects/AISA/llama.cpp-gemmini/.venv-convert311/bin/python -m experiments.potal_a4.native_measure build-potal-probes/potal-native-probe experiments/potal_a4/results-new/native-packets.json
```

`stress`는 synthetic 128조건을 실행한다. `weight_check`는 체크포인트의 Q4_HP1 값과 native C decoder를 비교한다. `native_model`은 별도 llama CPU 라이브러리로 FP16 baseline NLL을 계산한다. 그 CPU 빌드는 Q4_HP1 inference를 지원하지 않아 Q4 모델의 native NLL 대조는 완료하지 못했다. FP16 native와 FP32 NumPy forward도 F16 GELU table 등 구현 차이가 있어 bitwise parity로 취급하지 않는다.

```sh
rtk proxy /Users/chan/Projects/AISA/llama.cpp-gemmini/.venv-convert311/bin/python -m experiments.potal_a4.stress experiments/potal_a4/results-new/stress.json
rtk proxy env PYTHONPATH=gguf-py:. /Users/chan/Projects/AISA/llama.cpp-gemmini/.venv-convert311/bin/python -m experiments.potal_a4.weight_check /Users/chan/Projects/AISA/llama.cpp-gemmini/models/gpt2.Q4_HP1.gguf /Users/chan/Projects/AISA/llama.cpp-gemmini/build-cpu-only-audit/bin/libggml-base.dylib experiments/potal_a4/results-new/weight-parity.json
rtk proxy build-potal-probes/potal-native-model /Users/chan/Projects/AISA/llama.cpp-gemmini/models/gpt2.fp16.gguf experiments/potal_a4/results-new/valid.i32 256
rtk proxy /Users/chan/Projects/AISA/llama.cpp-gemmini/.venv-convert311/bin/python -m experiments.potal_a4.report experiments/potal_a4/results-new experiments/potal_a4/results-new/native-packets.json
```

## 파일과 증거

- `stream.py`, `attention.py`: 양자화 정책.
- `packet.py`: 실제 byte packet 생성·복원·fragment 산출.
- `model.py`, `run.py`, `matrix.py`: GPT-2 실행, NLL 및 메모리 기록.
- `verify.py`, `stress.py`, native probe/decoder: 경계·독립 검증.
- `results-v3/`: 최종 Q4 대상 비교. 이전 `results/`, `results-v2/` PPL은 FP16 head 입력까지 양자화했던 탐색 결과이므로 최종 표에서 제외한다. Native packet·decoder 검증은 이 출력층 정책과 무관하다.

모델 weight, tokenizer와 텍스트 hash, mode, context, 채점 개수 및 source hash를 기록한다. 수치 오차 감소가 PPL 개선을 보장하지 않으며, 8개 chunk의 결과를 다른 모델·전체 데이터셋으로 일반화하지 않는다. 실험 branch를 기본 inference 경로에 연결하거나 merge하지 않았다.
