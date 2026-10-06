#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["pydantic>=2,<3", "typer>=0.16,<1"]
# ///
# ─── How to run ───
# 1. Install uv: curl -LsSf https://astral.sh/uv/install.sh | sh
# 2. Run: uv run scripts/eval/compaction_abc/summarize.py RESULTS_DIR
# 3. Or make executable: chmod +x summarize.py && ./summarize.py RESULTS_DIR
# ──────────────────
"""Reduce paired residual replay measurements without inferring application latency."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import ClassVar, Final, Literal, assert_never

import typer
from pydantic import BaseModel, ConfigDict, Field

CONFIGURATIONS: Final = 8


class Row(BaseModel):
    """Parse the replay CSV at its file boundary."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    case: int
    phase: Literal["prefill", "decode"]
    bits: int
    dim: int
    source_m: int
    n: int
    original_k: int
    lanes: int
    variant: Literal["A", "B", "C"]
    m: int
    k: int
    runs: int
    residual_nnz: int
    digit_nnz: int
    logical_macs: int
    padded_macs: int
    decompose_ns: int
    pack_ns: int
    gather_ns: int
    restore_ns: int
    total_ns: int
    total_p10_ns: int
    total_p90_ns: int
    production_prepare_ns: int
    device_cycles: int = Field(gt=0)
    fragments: int
    numeric_mismatches: int
    numeric_max_error: int
    saturations: int
    all_column_bound: int


def reduce_file(path: Path) -> list[Row]:
    """Load all paired records from one completed replay."""
    with path.open(newline="") as stream:
        rows = [Row.model_validate(row) for row in csv.DictReader(stream)]
    groups: defaultdict[int, list[Row]] = defaultdict(list)
    for row in rows:
        groups[row.case].append(row)
    for case, paired in groups.items():
        if sorted(row.variant for row in paired) != ["A", "B", "C"]:
            message = f"Incomplete paired case {case}: {path}"
            raise RuntimeError(message)
        controls = {
            (
                row.source_m,
                row.n,
                row.original_k,
                row.lanes,
                row.residual_nnz,
                row.digit_nnz,
            )
            for row in paired
        }
        if len(controls) != 1:
            message = f"Mismatched controls for case {case}: {path}"
            raise RuntimeError(message)
    return rows


def main(result_dir: Path) -> None:
    """Write summary CSVs and tables from the eight completed configurations."""
    paths = sorted(result_dir.glob("*-a[48]-d*.csv"))
    if len(paths) != CONFIGURATIONS:
        message = f"Expected eight configuration files, found {len(paths)}"
        raise RuntimeError(message)
    summary_path = result_dir / "summary.csv"
    lines = [
        "# Residual compaction A/B/C measured tables",
        "",
        "CPU는 Apple M5 실측이다. GEMM은 Gemmini cycle model의 isolated-request 추정이다.",
        "CPU 합계는 stripe별 7회 중앙값의 합이며 application 경과 시간이 아니다.",
        "A는 현재 production packet builder + run-aware request 준비 시간과 결과 복원 시간이다.",
        "A*는 B/C와 같은 단순 replay builder에 행·K 압축을 적용한 대조군이다.",
        "DIM64는 동일 DIM16 수집 residual/stripe에 device geometry를 바꾼 조건이다.",
        "",
        "## CPU와 NPU 비용",
        "",
        "| 모델 | 정밀도 | DIM | 구간 | stripe 수 | 경로 | CPU 합계 ms | NPU Mcycles | 1GHz 가정 순차 합계 ms |",
        "|---|---|---:|---|---:|---|---:|---:|---:|",
    ]
    benefit_lines = [
        "",
        "## K 압축과 행 압축의 효과",
        "",
        "| 모델 | bits | DIM | 구간 | A/B K 감소 % | A/B padded MAC 감소 % | A/B cycle 감소 % | B/C row 감소 % | B/C cycle 감소 % |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    headers = [
        "model",
        "bits",
        "dim",
        "phase",
        "variant",
        "cases",
        "host_ms",
        "device_cycles",
        "sequential_1ghz_ms",
        "logical_macs",
        "padded_macs",
        "sum_rows",
        "sum_k",
        "gather_ms",
        "resident_host_ms",
        "numeric_mismatches",
        "saturations",
        "max_all_column_bound",
    ]
    with summary_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(headers)
        for path in paths:
            rows = reduce_file(path)
            model = path.stem.split("-")[0]
            phases = ("prefill", "decode")
            for phase in phases:
                grouped: dict[Literal["A", "B", "C"], list[Row]] = {
                    v: [r for r in rows if r.phase == phase and r.variant == v]
                    for v in ("A", "B", "C")
                }
                a, b, c = grouped["A"], grouped["B"], grouped["C"]
                for variant, selected in grouped.items():
                    first = selected[0]
                    match variant:
                        case "A":
                            host_ns = sum(
                                r.production_prepare_ns + r.restore_ns for r in selected
                            )
                        case "B" | "C":
                            host_ns = sum(r.total_ns for r in selected)
                        case unreachable:
                            assert_never(unreachable)
                    cycles = sum(r.device_cycles for r in selected)
                    gather_ns = sum(r.gather_ns for r in selected)
                    resident_ns = sum(
                        r.decompose_ns + r.pack_ns + r.restore_ns for r in selected
                    )
                    mismatches = sum(r.numeric_mismatches for r in selected)
                    saturations = sum(r.saturations for r in selected)
                    bound = max(r.all_column_bound for r in selected)
                    writer.writerow(
                        [
                            model,
                            first.bits,
                            first.dim,
                            phase,
                            variant,
                            len(selected),
                            host_ns / 1e6,
                            cycles,
                            (host_ns + cycles) / 1e6,
                            sum(r.logical_macs for r in selected),
                            sum(r.padded_macs for r in selected),
                            sum(r.m for r in selected),
                            sum(r.k for r in selected),
                            gather_ns / 1e6,
                            resident_ns / 1e6,
                            mismatches,
                            saturations,
                            bound,
                        ]
                    )
                    lines.append(
                        f"| {model} | W{first.bits}A{first.bits} | {first.dim} | {phase} | {len(selected)} | {variant} | "
                        f"{host_ns / 1e6:.3f} | {cycles / 1e6:.3f} | {(host_ns + cycles) / 1e6:.3f} |"
                    )
                    match variant:
                        case "A":
                            simple_ns = sum(r.total_ns for r in selected)
                            writer.writerow(
                                [
                                    model,
                                    first.bits,
                                    first.dim,
                                    phase,
                                    "A*",
                                    len(selected),
                                    simple_ns / 1e6,
                                    cycles,
                                    (simple_ns + cycles) / 1e6,
                                    sum(r.logical_macs for r in selected),
                                    sum(r.padded_macs for r in selected),
                                    sum(r.m for r in selected),
                                    sum(r.k for r in selected),
                                    gather_ns / 1e6,
                                    resident_ns / 1e6,
                                    mismatches,
                                    saturations,
                                    bound,
                                ]
                            )
                            lines.append(
                                f"| {model} | W{first.bits}A{first.bits} | {first.dim} | {phase} | {len(selected)} | A* | "
                                f"{simple_ns / 1e6:.3f} | {cycles / 1e6:.3f} | {(simple_ns + cycles) / 1e6:.3f} |"
                            )
                        case "B" | "C":
                            pass
                        case unreachable:
                            assert_never(unreachable)
                first = a[0]
                k_fraction = 100 * (1 - sum(r.k for r in a) / sum(r.k for r in b))
                mac_fraction = 100 * (
                    1 - sum(r.padded_macs for r in a) / sum(r.padded_macs for r in b)
                )
                cycle_fraction = 100 * (
                    1
                    - sum(r.device_cycles for r in a) / sum(r.device_cycles for r in b)
                )
                row_fraction = 100 * (1 - sum(r.m for r in b) / sum(r.m for r in c))
                row_cycle_fraction = 100 * (
                    1
                    - sum(r.device_cycles for r in b) / sum(r.device_cycles for r in c)
                )
                benefit_lines.append(
                    f"| {model} | {first.bits} | {first.dim} | {phase} | {k_fraction:.2f} | {mac_fraction:.2f} | "
                    f"{cycle_fraction:.2f} | {row_fraction:.2f} | {row_cycle_fraction:.2f} |"
                )
    lines.extend(benefit_lines)
    lines.extend(
        [
            "",
            "## 해석 범위",
            "",
            "1GHz는 시간 단위를 읽기 위한 가정이며 실제 운용 클록이 아니다. 순차 합계 = CPU_ms + cycles / (GHz × 10⁶).",
            "production A의 호스트 함수는 DIM16으로 컴파일했다. DIM64의 production-A CPU 값은 그 구현을 사용한 참고값이다.",
            "resident_host_ms는 단순 builder의 decomposition·pack·restore 합이다. A*의 K copy는 cached-weight-summary.csv에 별도 실측하여 더한다. B/C는 준비된 호스트 weight buffer를 직접 참조하는 조건이다.",
            "numeric_mismatches는 각 stripe에서 최대 32개 균등 선택 output column과 모든 source row를 독립 sparse residual 식과 비교한 값이다.",
            "all_column_bound < INT32_MAX이면 모든 output column에서 fragment/누산 포화가 없음을 보수적 L1 bound로 증명한다.",
            "prefill/decode별 1개 prompt의 결과다. 보드 실측, 전 모델 latency, 품질/PPL, CPU/NPU overlap, 여러 prompt에 대한 일반화를 주장하지 않는다.",
        ]
    )
    report = result_dir / "measured-tables.md"
    _ = report.write_text("\n".join(lines) + "\n")
    typer.echo(f"Wrote {summary_path} and {report}")


if __name__ == "__main__":
    typer.run(main)
