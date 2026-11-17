# /// script
# requires-python = ">=3.12"
# dependencies = ["pytest"]
# ///
# How to run (Linux): uv run --with pytest python -m pytest scripts/potal-timeline/test_binary_to_csv.py -q
"""Cross-language integration: real W4/W8 C++ writers versus the existing CSV writer."""

import csv
import io
import os
from pathlib import Path
import subprocess
from typing import BinaryIO, Final, TextIO

import pytest

import binary_to_csv
from binary_format import FormatError
from binary_to_csv import convert, transcode

ROOT: Final = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module", params=["ggml-potal", "ggml-potal-w4a4"])
def trace(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> Path:
    folder = tmp_path_factory.mktemp(request.param)
    backend = ROOT / "ggml/src" / request.param
    executable = folder / "writer"
    subprocess.run([os.environ.get("CXX", "g++"), "-std=c++17", "-O2", "-pthread", "-I", str(backend),
                    str(ROOT / "ggml/src/ggml-potal/tests/potal-timeline-binary-fixture.cpp"),
                    str(backend / "trace/timeline.cpp"), "-o", str(executable)], check=True)
    path = folder / "trace"
    env = dict(os.environ, POTAL_TIMELINE=str(path), POTAL_TIMELINE_EVENTS="20000", POTAL_TIMELINE_CPU="1")
    subprocess.run([str(executable), "8193"], env=env, check=True)
    return path


def test_restores_events_metadata_and_cpu_samples(trace: Path, tmp_path: Path) -> None:
    destination = tmp_path / "restored.csv"
    assert convert(Path(str(trace) + ".binary"), destination) == 16386
    with Path(str(trace) + ".csv").open() as original, destination.open() as restored:
        old = list(csv.reader(original))
        new = list(csv.reader(restored))
    # flush() adds a clock anchor per call: metadata outside those anchors is identical.
    def stable(rows: list[list[str]]) -> list[list[str]]:
        result = []
        for row in rows:
            if row[0] == "A" and row[3] == "flush":
                continue
            if row[0] == "T":
                row = row[:15] + ["anchor-count"] + row[16:]
            result.append(row)
        return result
    assert stable(new) == stable(old)
    assert Path(str(destination) + ".cpu.csv").read_bytes() == Path(str(trace) + ".csv.cpu.csv").read_bytes()


@pytest.mark.parametrize("damage", ["truncate", "count", "layout", "slot", "trailing"])
def test_rejects_corrupt_trace_without_publishing(trace: Path, tmp_path: Path, damage: str) -> None:
    payload = Path(str(trace) + ".binary").read_bytes()
    mutations = {
        "truncate": payload[:len(payload) // 2],
        "count": payload.replace(b"#END,events=16386", b"#END,events=16387"),
        "layout": payload.replace(b"event_bytes=64", b"event_bytes=63"),
        "slot": payload.replace(b"#BLOCK,slot=0", b"#BLOCK,slot=7", 1),
        "trailing": payload + b"x",
    }
    source = tmp_path / "broken.bin"
    source.write_bytes(mutations[damage])
    destination = tmp_path / "result.csv"
    with pytest.raises(FormatError):
        convert(source, destination)
    assert not destination.exists()
    assert not Path(str(destination) + ".cpu.csv").exists()


def test_preserves_existing_output(trace: Path, tmp_path: Path) -> None:
    destination = tmp_path / "existing.csv"
    destination.write_text("original measurement\n")
    with pytest.raises(FileExistsError):
        convert(Path(str(trace) + ".binary"), destination)
    assert destination.read_text() == "original measurement\n"


def test_writer_without_cpu_creates_only_timeline(trace: Path, tmp_path: Path) -> None:
    path = tmp_path / "trace"
    env = dict(os.environ, POTAL_TIMELINE=str(path), POTAL_TIMELINE_EVENTS="64", POTAL_TIMELINE_CPU="0")
    subprocess.run([str(trace.parent / "writer"), "3"], env=env, check=True)
    destination = tmp_path / "restored.csv"
    assert convert(Path(str(path) + ".binary"), destination) == 6
    assert not Path(str(destination) + ".cpu.csv").exists()


def test_preserves_output_created_during_conversion(trace: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    destination = tmp_path / "result.csv"
    def concurrent_writer(stream: BinaryIO, output: TextIO, cpu: TextIO | None) -> int:
        count = transcode(stream, output, cpu)
        destination.write_text("another completed measurement\n")
        return count
    monkeypatch.setattr(binary_to_csv, "transcode", concurrent_writer)
    with pytest.raises(FileExistsError):
        convert(Path(str(trace) + ".binary"), destination)
    assert destination.read_text() == "another completed measurement\n"


def test_accepts_empty_trace_without_cpu() -> None:
    source = io.BytesIO(b"# potal_timeline_binary v1\n"
                       b"#BINARY,endian=little,event_bytes=64,cpu_bytes=56,cpu=0,counter_source=unavailable\n"
                       b"# potal_timeline v1\n#EVENT,t_local\n#END,events=0\n")
    output = io.StringIO()
    assert transcode(source, output, None) == 0
    assert output.getvalue() == "# potal_timeline v1\n#EVENT,t_local\n#END,events=0\n"
