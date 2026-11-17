# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
# How to run: imported by cpu_parse.py and binary_to_csv.py.
"""Interpret command-window counters without calling mixed API brackets pure CPU work."""


def counter_only_stage(stage: int) -> bool:
    """Stages whose recording path deliberately omits clock and scheduler calls."""
    return 26 <= stage <= 39 or 45 <= stage <= 49 or 121 <= stage <= 127


def measurement_kind(stage: int) -> str:
    """Keep CPU-only descriptor work distinct from inclusive or blocking brackets."""
    if 121 <= stage <= 125:
        return "cpu_descriptor"
    if stage == 35:
        return "fence_api_mixed"
    if stage in (26, 39, 45, 46, 47, 48, 49):
        return "issuer_window_inclusive"
    if stage in (27, 28):
        return "cpu_setup"
    if counter_only_stage(stage):
        return "command_api_mixed"
    return "stage_inclusive"
