from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys

import numpy as np


def report(root: Path) -> str:
    baseline = json.loads((root / "test512-original_a4nks.json").read_text())
    base_losses = np.asarray(baseline["nll_sums"]) / 255
    base_p_main = baseline["metrics"]["p"]["main_fragments"]
    picks = np.random.default_rng(9).integers(0, 8, (10000, 8))
    lines = ["# 측정 표", "", "Context 512, 8 chunks, 2,040 scored tokens. PPL은 낮을수록 좋다.", "",
             "| 조건 | PPL | 원문 대비 PPL 변화 | paired chunk bootstrap 95% | 선형 B/값 | Q B/값 | P B/값 | PV fragment / 원래 main |",
             "|---|---:|---:|---|---:|---:|---:|---:|"]
    csv_rows = []
    for path in sorted(root.glob("test512-*.json")):
        data = json.loads(path.read_text())
        assert data["tokens_sha256"] == baseline["tokens_sha256"]
        assert data["scored_tokens"] == 2040
        assert data["metrics"]["linear"]["calls"] in (0, 384)
        delta = np.asarray(data["nll_sums"]) / 255 - base_losses
        interval = 100 * np.expm1(np.percentile(delta[picks].mean(axis=1), [2.5, 97.5]))
        relative = float(100 * np.expm1(delta.mean()))
        per_value = []
        for role in ("linear", "q", "p"):
            values = data["metrics"][role]
            per_value.append(sum(values[key] for key in ("main_bytes", "digit_bytes", "metadata_bytes")) / max(values["values"], 1))
        p = data["metrics"]["p"]
        work = (p["main_fragments"] + p["upper_fragments"]) / base_p_main
        name = data["mode"]["name"]
        lines.append(f"| {name} | {data['ppl']:.4f} | {relative:+.2f}% | [{interval[0]:+.2f}%, {interval[1]:+.2f}%] | "
                     + " | ".join(f"{b:.3f}" for b in per_value) + f" | {work:.3f} |")
        csv_rows.append([name, data["ppl"], relative, *interval, *per_value, work, data["peak_rss_bytes"]])
    with (root / "summary.csv").open("w", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(["mode", "ppl", "relative_ppl_pct", "ci_low_pct", "ci_high_pct", "linear_bytes_per_value", "q_bytes_per_value", "p_bytes_per_value", "pv_fragments_over_fixed_main", "process_peak_rss_bytes"])
        writer.writerows(csv_rows)
    lines += ["", "B/값은 실험 packet의 payload+index+scale이며 누적 값 수로 정규화했다. KV와 원본 FP32를 대체한 실측이 아니다.",
              "PV 분모는 모든 조건에 동일한 원래 main이다. 작은 P 행 묶음의 증가한 main 패딩을 포함한다.",
              "95% 구간은 8개 chunk를 재추출한 탐색적 추정이다. 전체 corpus의 신뢰구간으로 해석하지 않는다.", "",
              "## 개발용 제약 비교", "", "| 조건 | PPL | P B/값 | P zero rows | P row-sum 최대 오차 |", "|---|---:|---:|---:|---:|"]
    for path in sorted(root.glob("valid256-*.json")):
        data = json.loads(path.read_text())
        p = data["metrics"]["p"]
        size = sum(p[k] for k in ("main_bytes", "digit_bytes", "metadata_bytes")) / max(p["values"], 1)
        lines.append(f"| {data['mode']['name']} | {data['ppl']:.4f} | {size:.3f} | {p['zero_rows']} | {p['row_sum_error']:.4f} |")
    lines += ["", "## Context 제한", "", "각 context 안에서만 직접 비교한다. 길이에 따라 채점 위치와 token 수가 다르다.", "",
              "| Context | 조건 | 예측 token 수 | PPL |", "|---:|---|---:|---:|"]
    for context in (128, 512, 1024):
        for name in ("weights_only", "original_a4nks", "direct_p8"):
            data = json.loads((root / f"test{context}-{name}.json").read_text())
            lines.append(f"| {context} | {name} | {data['scored_tokens']} | {data['ppl']:.4f} |")
    return "\n".join(lines) + "\n"


def native_table(path: Path) -> str:
    records = json.loads(path.read_text())
    lines = ["# Native packet 공간 실측", "", "512×4096 folded codes, stripe=32, DIM32. MiB=2^20 bytes.", "",
             "| 조건 | clipped 유지 MiB | direct 유지 MiB | 유지 공간 감소 | clipped peak RSS MiB | direct peak RSS MiB |",
             "|---|---:|---:|---:|---:|---:|"]
    for case in ("narrow", "sparse", "dense", "wide", "carry"):
        sizes, peaks = [], []
        for mode in ("clipped", "direct"):
            runs = [r for r in records if r["fixture"] == case and r["mode"] == mode]
            assert len(runs) == 3
            sizes.append(sum(runs[0][key] for key in ("main_bytes", "dense_residual_bytes", "packet_digit_bytes", "packet_metadata_bytes")) / 2 ** 20)
            peaks.append(statistics.median(r["peak_rss_bytes"] for r in runs) / 2 ** 20)
        lines.append(f"| {case} | {sizes[0]:.3f} | {sizes[1]:.3f} | {100 * (1-sizes[1]/sizes[0]):.1f}% | {peaks[0]:.3f} | {peaks[1]:.3f} |")
    lines += ["", "유지는 container size와 C++ metadata 크기이며 allocator 여유 capacity는 제외한다. Peak RSS는 새 프로세스 3회 중앙값이며 원자료에 각 반복값이 있다.",
              "이 값에는 입력 생성·packet 복원 검증 scratch도 포함된다. 모델 전체 peak나 accelerator SRAM은 측정하지 않았다."]
    return "\n".join(lines) + "\n"


def main() -> None:
    root = Path(sys.argv[1])
    native = Path(sys.argv[2])
    (root / "TABLES.md").write_text(report(root))
    (root / "NATIVE.md").write_text(native_table(native))
    files = [*Path(__file__).parent.glob("*.py"), *Path(__file__).parent.glob("*.cpp"),
             *(path for path in root.glob("*.json") if path.name != "artifact-sha256.json"), native]
    manifest = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    (root / "artifact-sha256.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(root / "TABLES.md")


if __name__ == "__main__":
    main()
