# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==2.5.3", "matplotlib==3.11.2"]
# ///
# Run: rtk proxy uv run --script analysis/model/analyze_weights.py gpt2
from __future__ import annotations

from contextlib import ExitStack
import csv
import ctypes
import hashlib
import json
import math
import mmap
import os
from pathlib import Path
import subprocess
import sys
import time

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
from weight_io import ROOT, FloatStream, GGUFReader, InputError, group_name, quant_values
from weight_stats import EDGES, FINE_EDGES, Moments


def main() -> None:
    model = sys.argv[1]
    output = Path(__file__).resolve().parent
    config = json.loads((output / "inputs.json").read_text())[model]
    started = time.monotonic()
    readers = {label: GGUFReader(str(ROOT / config[key]), "r") for label, key in (("INT4", "int4"), ("INT8", "int8"))}
    tensors = {label: {tensor.name: tensor for tensor in reader.tensors} for label, reader in readers.items()}
    first = readers["INT4"]
    metadata = {key: field.contents() for key, field in first.fields.items()
                if key.startswith(("general.", "gpt2.", "llama."))}
    hashes: dict[str, str] = {}
    expected = {Path(name).name: digest for digest, name in
                (line.split() for line in (ROOT / "scripts/experiment/default-ppl-models.sha256").read_text().splitlines())}
    for key in ("int4", "int8"):
        path = ROOT / config[key]
        with path.open("rb") as stream:
            hashes[key] = hashlib.file_digest(stream, "sha256").hexdigest()
        if hashes[key] != expected[path.name]:
            raise InputError(f"Quantized model differs from the PPL manifest: {path}")
    library = ctypes.CDLL(str(ROOT / "build-metal-llama-full/quality-potal4-d16/bin/libggml-base.dylib"))
    groups = {label: {} for label in ("FP16", "INT4", "INT8")}
    totals = {label: Moments() for label in groups}
    linear = {label: Moments() for label in groups}
    records: list[dict] = []
    checked = 0
    process = None
    with ExitStack() as stack:
        if config["reference"].startswith("https://"):
            process = stack.enter_context(subprocess.Popen(
                ["curl", "-fsSL", "--connect-timeout", "30", config["reference"]], stdout=subprocess.PIPE))
            source = process.stdout
            if source is None:
                raise InputError("Missing reference download stream")
        else:
            source = stack.enter_context((ROOT / config["reference"]).open("rb"))
        reader = FloatStream(source)
        entries = reader.header()
        data_offset = reader.position
        names = {entry.name for entry in entries}
        if any(set(index) != names for index in tensors.values()):
            raise InputError("Reference and HP1 tensor inventories differ")
        for ordinal, entry in enumerate(entries):
            for index in tensors.values():
                if tuple(index[entry.name].shape) != entry.shape:
                    raise InputError(f"Tensor shape differs: {entry.name}")
            reader.skip_to(data_offset + entry.offset)
            columns = entry.shape[0]
            rows = math.prod(entry.shape) // columns
            chunk_rows = max(1, 262144 // columns)
            moments = {label: Moments() for label in groups}
            for start in range(0, rows, chunk_rows):
                end = min(rows, start + chunk_rows)
                raw = reader.read((end - start) * columns * entry.dtype.itemsize)
                reference = np.frombuffer(raw, dtype=entry.dtype).astype(np.float32)
                moments["FP16"].add(reference, reference, None)
                for label, index in tensors.items():
                    tensor = index[entry.name]
                    values, codes = quant_values(tensor, start, end)
                    if start == 0 and codes is not None:
                        bits = 4 if label == "INT4" else 8
                        function = getattr(library, f"dequantize_row_q{bits}_hp1")
                        function.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
                        function.restype = None
                        count = min(columns, 256)
                        packed = np.ascontiguousarray(tensor.data.reshape(rows, -1)[0])
                        native = np.empty(count, dtype=np.float32)
                        function(packed.ctypes.data, native.ctypes.data, count)
                        if not np.array_equal(native.view(np.uint32), values[:count].view(np.uint32)):
                            raise InputError(f"Native HP1 decoder mismatch: {label} {entry.name}")
                        checked += 1
                    moments[label].add(values, reference, codes)
            group = group_name(entry)
            for label, accumulator in moments.items():
                totals[label].merge(accumulator)
                groups[label].setdefault(group, Moments()).merge(accumulator)
                if group != "auxiliary":
                    linear[label].merge(accumulator)
                storage = str(entry.dtype) if label == "FP16" else tensors[label][entry.name].tensor_type.name
                records.append(dict(precision=label, tensor=entry.name, group=group, storage=storage,
                                    shape="x".join(map(str, reversed(entry.shape))), **accumulator.summary(),
                                    code_histogram=accumulator.code_histogram.tolist()))
            for quant_reader in readers.values():
                quant_reader.data._mmap.madvise(mmap.MADV_DONTNEED)
            print(f"{model}: {ordinal + 1}/{len(entries)} {entry.name}", flush=True)
        hashes["reference"] = reader.finish()
        if process is not None and process.wait() != 0:
            raise InputError("FP16 reference download failed")
        if hashes["reference"] != config["reference_sha256"]:
            raise InputError(f"FP16 reference SHA-256 mismatch: {hashes['reference']}")
    group_data = {label: {name: dict(**value.summary(), histogram=value.histogram.tolist(),
                                    fine_histogram=value.fine_histogram.tolist(),
                                    zeroed_reference_histogram=value.zeroed_reference_histogram.tolist(),
                                    code_histogram=value.code_histogram.tolist())
                          for name, value in mapping.items()} for label, mapping in groups.items()}
    for label, mapping in groups.items():
        for name, value in mapping.items():
            if np.any(value.zeroed_reference_histogram > groups["FP16"][name].fine_histogram):
                raise InputError(f"Zeroed reference count exceeds FP16 population: {label} {name}")
            if int(value.zeroed_reference_histogram.sum()) > value.zeros:
                raise InputError(f"New zero count exceeds stored zero count: {label} {name}")
    result = dict(model=model, title=config["title"], input=config, sha256=hashes, metadata=metadata,
                  reference_bytes=reader.position, native_decoder_checks=checked,
                  seconds=time.monotonic() - started, histogram_log2_edges=EDGES.tolist(),
                  fine_histogram_log2_edges=FINE_EDGES.tolist(),
                  summary={label: value.summary() for label, value in totals.items()},
                  linear_summary={label: value.summary() for label, value in linear.items()},
                  groups=group_data, tensors=records)
    destination = output / "data"
    destination.mkdir(exist_ok=True)
    (destination / f"{model}.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    fields = [key for key in records[0] if key != "code_histogram"]
    with (destination / f"{model}-tensors.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    with (destination / f"{model}-layers.csv").open("w", newline="") as stream:
        fields = ["precision", "layer", *totals["FP16"].summary().keys()]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for label, mapping in groups.items():
            for name, value in mapping.items():
                writer.writerow(dict(precision=label, layer=name, **value.summary()))
    with (destination / f"{model}-magnitude-fine.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["precision", "layer", "kind", "magnitude_lower_inclusive",
                         "magnitude_upper_exclusive", "count", "percent_of_layer"])
        for label, mapping in groups.items():
            for name, value in sorted(mapping.items()):
                writer.writerow([label, name, "zero", 0, 0, value.zeros, 100 * value.zeros / value.count])
                for index, count in enumerate(value.fine_histogram):
                    writer.writerow([label, name, "bin", 2.0 ** FINE_EDGES[index],
                                     2.0 ** FINE_EDGES[index + 1], int(count), 100 * int(count) / value.count])
    with (destination / f"{model}-zeroed-reference-fine.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["precision", "layer", "fp_magnitude_lower_inclusive", "fp_magnitude_upper_exclusive",
                         "fp_nonzero_count", "zeroed_count", "zeroed_percent_of_fp_bin", "zeroed_percent_of_layer"])
        for label in ("INT4", "INT8"):
            for name, value in sorted(groups[label].items()):
                reference_histogram = groups["FP16"][name].fine_histogram
                for index, count in enumerate(value.zeroed_reference_histogram):
                    population = int(reference_histogram[index])
                    writer.writerow([label, name, 2.0 ** FINE_EDGES[index], 2.0 ** FINE_EDGES[index + 1],
                                     population, int(count), 100 * int(count) / population if population else "",
                                     100 * int(count) / value.count])
    print(json.dumps(dict(model=model, seconds=result["seconds"], sha256=hashes,
                         parameters=totals["FP16"].count, native_decoder_checks=checked)), flush=True)


if __name__ == "__main__":
    main()
