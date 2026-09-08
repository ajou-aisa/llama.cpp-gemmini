from __future__ import annotations

import json
from typing import Sequence

try:
    from .cycle_schema import JsonValue
    from .residual_profile import HostTiming, Profile
except ImportError:
    from cycle_schema import JsonValue
    from residual_profile import HostTiming, Profile


def trace(profiles: Sequence[Profile]) -> str:
    events: list[dict[str, JsonValue]] = []
    for pid, execution in enumerate(dict.fromkeys(profile.host.execution_id for profile in profiles), 1):
        group = [profile for profile in profiles if profile.host.execution_id == execution]
        origin = min(profile.host.start_ns for profile in group)
        events.append({"ph": "M", "name": "process_name", "pid": pid, "tid": 0,
                       "args": {"name": f"{execution} (independent origin {origin} ns)"}})
        tids = {profile.host.start_tid for profile in group} | {worker.tid for profile in group for worker in profile.workers}
        for tid in sorted(tids):
            kind = group[0].host.thread_id_kind
            events.append({"ph": "M", "name": "thread_name", "pid": pid, "tid": tid,
                           "args": {"name": f"{kind} {tid}"}})
        spans: list[tuple[HostTiming, str, dict[str, JsonValue]]] = []
        for profile in group:
            identity: dict[str, JsonValue] = {"execution_id": execution, "run_id": profile.run_id,
                                              "layer": profile.layer, "stripe_id": profile.stripe_id}
            spans.append((profile.host, "CPU_DIRECT", identity))
            spans.extend((timing.host, name, {**identity, "thread_cpu_ns": str(timing.cpu_ns)
                                               if timing.cpu_ns is not None else None})
                         for name, timing in profile.phases.items())
            for worker in profile.workers:
                args: dict[str, JsonValue] = {**identity, "worker_id": worker.worker_id,
                    "thread_cpu_ns": str(worker.timing.cpu_ns) if worker.timing.cpu_ns is not None else None}
                spans.append((worker.timing.host, "worker work", args))
                if worker.barrier is not None:
                    spans.append((worker.barrier.host, "barrier", {**args, "thread_cpu_ns":
                        str(worker.barrier.cpu_ns) if worker.barrier.cpu_ns is not None else None}))
            for tile in profile.tiles:
                args = {**identity, "worker_id": tile.worker_id, "node_id": tile.node_id,
                        "j_begin": tile.j_begin, "j_end": tile.j_end,
                        "thread_cpu_ns": str(tile.timing.cpu_ns) if tile.timing.cpu_ns is not None else None}
                spans.append((tile.timing.host, "J-tile compute", args))
                if tile.log is not None:
                    spans.append((tile.log.host, "legacy logging", {**args, "thread_cpu_ns":
                        str(tile.log.cpu_ns) if tile.log.cpu_ns is not None else None}))
        for host, name, args in sorted(spans, key=lambda span: (span[0].start_ns, -span[0].duration_ns)):
            events.append({"ph": "X", "name": name, "cat": "residual", "pid": pid, "tid": host.start_tid,
                           "ts": (host.start_ns - origin) / 1000, "dur": host.duration_ns / 1000,
                           "args": {**args, "start_ns": str(host.start_ns), "end_ns": str(host.end_ns),
                                    "wall_ns": str(host.duration_ns)}})
    return json.dumps({"traceEvents": events, "displayTimeUnit": "ns"}, ensure_ascii=False) + "\n"
