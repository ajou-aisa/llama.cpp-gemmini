from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
from gguf import GGUFReader, GGMLQuantizationType

from .attention import p_quant, softmax, stationary
from .metrics import Measurements
from .stream import stream_quant
from .types import F32, MainRule, Mode, PPolicy, QuantError, StreamPolicy


class Weights:
    def __init__(self, path: Path) -> None:
        self.values: dict[str, F32] = {}
        self.quantized_names: set[str] = set()
        for tensor in GGUFReader(path).tensors:
            shape = tuple(int(v) for v in tensor.shape[::-1])
            if tensor.tensor_type in (GGMLQuantizationType.F32, GGMLQuantizationType.F16):
                value = tensor.data.astype(np.float32).reshape(shape)
            elif tensor.tensor_type == GGMLQuantizationType.Q4_HP1:
                self.quantized_names.add(tensor.name)
                blocks = tensor.data.reshape(-1, 24)
                exponent = blocks[:, 16:18].copy().view("<i2").reshape(-1)
                scale = blocks[:, 20:24].copy().view("<f4").reshape(-1)
                codes = np.concatenate((blocks[:, :16] & 15, blocks[:, :16] >> 4), axis=1).astype(np.float32) - 8
                value = np.ldexp(codes * scale[:, None],
                                 np.where(exponent == -32768, 0, exponent).astype(np.int32)[:, None])
                value[exponent == -32768] = 0
                value = value.reshape(shape)
            else:
                raise QuantError(f"Unsupported weight {tensor.name}: {tensor.tensor_type}")
            if not np.isfinite(value).all():
                raise QuantError(f"Nonfinite weight {tensor.name}")
            self.values[tensor.name] = value

    def norm(self, x: F32, prefix: str) -> F32:
        centered = x - x.mean(axis=-1, keepdims=True, dtype=np.float64).astype(np.float32)
        variance = np.square(centered).mean(axis=-1, keepdims=True, dtype=np.float64).astype(np.float32)
        return centered / np.sqrt(variance + np.float32(1e-5)) * self.values[prefix + ".weight"] + self.values[prefix + ".bias"]


class Forward:
    def __init__(self, weights: Weights, mode: Mode) -> None:
        self.weights = weights
        self.mode = mode
        self.metrics = Measurements.empty()

    def linear(self, x: F32, prefix: str) -> F32:
        w = self.weights.values[prefix + ".weight"]
        if self.mode.linear is not None and prefix + ".weight" in self.weights.quantized_names:
            result = stream_quant(x, StreamPolicy(len(w), 2, self.mode.linear, self.mode.upper_limbs))
            self.metrics = replace(self.metrics, linear=self.metrics.linear.observe(x, result))
            x = result.values
        return x @ w.T + self.weights.values[prefix + ".bias"]

    def attention(self, operands: tuple[F32, F32, F32]) -> F32:
        q, k, v = operands
        rows = len(q)
        heads = [u.reshape(rows, 12, 64).transpose(1, 0, 2) for u in (q, k, v)]
        mask = np.triu(np.full((rows, rows), -np.inf, dtype=np.float32), 1)
        output = np.empty_like(heads[0])
        for h in range(12):
            qh, kh, vh = (u[h] for u in heads)
            if self.mode.attention:
                result = stream_quant(qh, StreamPolicy(rows, 6, MainRule.DIRECT, self.mode.upper_limbs))
                self.metrics = replace(self.metrics, q=self.metrics.q.observe(qh, result),
                    k_main_fragments=self.metrics.k_main_fragments + sum(p.main_fragments for p in result.packets),
                    k_upper_fragments=self.metrics.k_upper_fragments + sum(p.upper_fragments for p in result.packets),
                    stationary_bytes=self.metrics.stationary_bytes + kh.size * 2 + vh.size)
                qh = result.values
                kh = stationary(kh.T, 8).T
                vh = stationary(vh, 4)
            p = softmax((qh @ kh.T) * np.float32(0.125) + mask)
            if self.mode.attention:
                result = p_quant(p, PPolicy(64, self.mode.p_rows, self.mode.p_bits))
                self.metrics = replace(self.metrics, p=self.metrics.p.observe(p, result).probability(result.values))
                p = result.values
            output[h] = p @ vh
        return output.transpose(1, 0, 2).reshape(rows, 768)

    def run(self, tokens: np.ndarray[tuple[int, ...], np.dtype[np.int32]]) -> F32:
        x = self.weights.values["token_embd.weight"][tokens] + self.weights.values["position_embd.weight"][:len(tokens)]
        for layer in range(12):
            prefix = f"blk.{layer}"
            a = self.weights.norm(x, prefix + ".attn_norm")
            q, k, v = np.split(self.linear(a, prefix + ".attn_qkv"), 3, axis=1)
            x = x + self.linear(self.attention((q, k, v)), prefix + ".attn_output")
            f = self.linear(self.weights.norm(x, prefix + ".ffn_norm"), prefix + ".ffn_up")
            f = np.float32(0.5) * f * (1 + np.tanh(np.float32(0.7978845608028654) * f * (1 + np.float32(0.044715) * f * f)))
            x = x + self.linear(f, prefix + ".ffn_down")
        return self.weights.norm(x, "output_norm")

    def logits(self, hidden: F32) -> F32:
        name = "output.weight" if "output.weight" in self.weights.values else "token_embd.weight"
        head = self.weights.values[name]
        if self.mode.linear is not None and name in self.weights.quantized_names:
            result = stream_quant(hidden, StreamPolicy(len(head), 2, self.mode.linear, self.mode.upper_limbs))
            self.metrics = replace(self.metrics, linear=self.metrics.linear.observe(hidden, result))
            hidden = result.values
        return hidden @ head.T
