from collections.abc import Mapping
from typing import Final

METRIC_OPTIONS: Final = (
    "GGML_GEMMINI_ACT_METRICS",
    "GGML_GEMMINI_ACT_QUANT_METRICS",
    "GGML_GEMMINI_RESIDUAL_METRICS",
    "GGML_GEMMINI_SCALE_METRICS",
)


class InvalidEvaluationOption(ValueError):
    def __init__(self, name: str, value: str) -> None:
        self.name = name
        self.value = value
        super().__init__(f"{name} must be 0 or 1; got {value!r}")


def validate_evaluation_options(options: Mapping[str, str]) -> None:
    for name in (*METRIC_OPTIONS, "CYCLE_SIM", "CYCLE_DETAIL", "LOG_DEBUG",
                 "LOG_CYCLE", "LOG_DUMP", "LOG_DUMP_SCALE"):
        value = options.get(name, "0")
        if value not in ("0", "1"):
            raise InvalidEvaluationOption(name, value)
