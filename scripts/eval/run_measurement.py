#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# How to run: python3 scripts/eval/run_measurement.py describe
"""Canonical entry point of the PoTal measurement interface: performance, timeline, metric.

  run_measurement.py performance [options]                 TTFT/TPOT, CPU-NPU timing (+ --timeline compact)
  run_measurement.py timeline --from-run RUN [options]     export view of a completed run's stored schedule
  run_measurement.py metric activation|residual|scu [...]  activation / residual / SCU metrics
  run_measurement.py metric all [...]                      the three metrics over models x precisions x DIMs,
                                                           aggregated (orchestration, not a fourth metric)

  run_measurement.py identity RUN [RUN ...]                shared model/configuration identity of finished runs
  run_measurement.py describe [--json | --markdown SECTION | --update-readme]

This file only selects the delegate script and a default output directory; every option after the domain is passed
through unchanged (`run_measurement.py performance --help` shows the delegate's options). All logic stays in
run_cycle_evaluation.py (performance, timeline), campaign.py (metric) and metric_sweep.py (metric all).
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from measurement_domains import (
    DOMAINS,
    EVAL,
    METRIC_ALL,
    METRIC_KINDS,
    SECTIONS,
    SWEEP,
    description,
    markdown,
    measurement,
    update_readme,
)

# Read-only helpers next to the measurement domains: (script, fixed arguments).
HELPERS = {"identity": ("measurement_identity.py",)}
# The wrapper chooses the delegate's mode; a passthrough option may not choose another one.
MODE_OPTIONS = ("--run",)


class UsageError(ValueError):
    pass


def option(arguments: list[str], name: str, default: str | None = None) -> str | None:
    """Value of `name VALUE` or `name=VALUE` in a passthrough argument list."""
    for index, value in enumerate(arguments):
        if value == name and index + 1 < len(arguments):
            return arguments[index + 1]
        if value.startswith(name + "="):
            return value.split("=", 1)[1]
    return default


def present(arguments: list[str], name: str) -> bool:
    return any(value == name or value.startswith(name + "=") for value in arguments)


def default_output(command: tuple[str, ...], arguments: list[str], stamp: str) -> Path:
    precision, dim = option(arguments, "--precision", "a8w8"), option(arguments, "--dim", "32")
    if command == ("performance",):
        model = Path(option(arguments, "--model", "model") or "model").stem
        suffix = "-smoke" if present(arguments, "--smoke") else ""
        return Path("runs/performance") / f"{stamp}-{model}-{precision}-d{dim}-hp1{suffix}"
    if command == ("timeline",):
        source = Path(option(arguments, "--from-run", "run") or "run").name
        return Path("runs/timeline") / f"{stamp}-{source}"
    if command == ("metric", METRIC_ALL):
        return Path("runs/metrics") / f"sweep-{stamp}"
    model = option(arguments, "--model", "model")
    return Path("runs/metrics") / command[1] / f"{stamp}-{model}-{precision}-d{dim}"


def delegate(argv: list[str], stamp: str | None = None) -> tuple[Path, list[str]]:
    """(delegate script, its argument list) for `run_measurement.py ARGV`; no logic beyond the selection."""
    if not argv or argv[0] not in DOMAINS:
        raise UsageError("choose a domain: performance, timeline, metric activation|residual|scu|all "
                         "(or identity, describe)")
    if argv[0] == "metric":
        if len(argv) < 2 or argv[1] not in (*METRIC_KINDS, METRIC_ALL):
            raise UsageError("metric requires one of: " + ", ".join((*METRIC_KINDS, METRIC_ALL)) +
                             " (the legacy cycle replay adapter is `campaign.py cycle`)")
        command, rest = (argv[0], argv[1]), argv[2:]
    else:
        command, rest = (argv[0],), argv[1:]
    chosen = [name for name in MODE_OPTIONS if present(rest, name)]
    if chosen:
        raise UsageError(f"{' '.join(command)} selects the mode; remove {', '.join(chosen)}")
    if command == ("timeline",) and not present(rest, "--from-run") and not present(rest, "--help") and "-h" not in rest:
        raise UsageError("timeline requires --from-run COMPLETED_PERFORMANCE_RUN")
    script, *fixed = (SWEEP,) if command == ("metric", METRIC_ALL) else measurement(command)[1].delegate
    arguments = [*fixed, *rest]
    # Help and dry runs write nothing; a resumed sweep writes into its recorded directory.
    unwritten = present(rest, "--help") or "-h" in rest or present(rest, "--dry-run") or present(rest, "--resume")
    if not present(rest, "--output") and not unwritten:
        moment = stamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        arguments += ["--output", str(default_output(command, rest, moment))]
    return EVAL / script, arguments


def describe(arguments: list[str]) -> int:
    if arguments[:1] == ["--json"]:
        print(json.dumps(description(), indent=2, sort_keys=True))
    elif arguments[:1] == ["--markdown"]:
        sections = SECTIONS if arguments[1:2] in ([], ["all"]) else (arguments[1],)
        print("\n".join(markdown(section) for section in sections))
    elif arguments[:1] == ["--update-readme"]:
        print("README.md " + ("updated" if update_readme(EVAL / "README.md") else "already current"))
    else:
        print(__doc__)
        print(markdown("matrix"))
    return 0


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    if arguments[0] == "describe":
        return describe(arguments[1:])
    try:
        if arguments[0] in HELPERS:
            script, forwarded = EVAL / HELPERS[arguments[0]][0], [*HELPERS[arguments[0]][1:], *arguments[1:]]
        else:
            script, forwarded = delegate(arguments)
    except (UsageError, KeyError) as error:
        print(f"run_measurement: {error}", file=sys.stderr)
        return 2
    os.execv(sys.executable, [sys.executable, "-B", str(script), *forwarded])
    return 0  # unreachable: execv replaces the process


if __name__ == "__main__":
    raise SystemExit(main())
