# Native packet 공간 실측

512×4096 folded codes, stripe=32, DIM32. MiB=2^20 bytes.

| 조건 | clipped 유지 MiB | direct 유지 MiB | 유지 공간 감소 | clipped peak RSS MiB | direct peak RSS MiB |
|---|---:|---:|---:|---:|---:|
| narrow | 10.000 | 2.000 | 80.0% | 11.984 | 3.953 |
| sparse | 12.090 | 4.084 | 66.2% | 15.469 | 7.438 |
| dense | 17.088 | 6.961 | 59.3% | 21.422 | 11.766 |
| wide | 17.088 | 4.834 | 71.7% | 21.625 | 8.797 |
| carry | 27.722 | 19.722 | 28.9% | 33.625 | 25.922 |

유지는 container size와 C++ metadata 크기이며 allocator 여유 capacity는 제외한다. Peak RSS는 새 프로세스 3회 중앙값이며 원자료에 각 반복값이 있다.
이 값에는 입력 생성·packet 복원 검증 scratch도 포함된다. 모델 전체 peak나 accelerator SRAM은 측정하지 않았다.
