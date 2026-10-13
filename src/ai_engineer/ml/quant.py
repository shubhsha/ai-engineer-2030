"""Chapter 23: symmetric integer quantization and bfloat16 rounding.

A b-bit symmetric quantizer maps [-max|w|, +max|w|] onto the integers
[-(2**(b-1) - 1), 2**(b-1) - 1] with one floating-point scale per block
of values. Smaller blocks follow the data more closely, at the price of
storing more scales.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ai_engineer.ml.numerics import Array

SCALE_BYTES = 2  # scales are commonly stored as 16-bit floats


@dataclass(frozen=True)
class Quantized:
    codes: NDArray[np.int8]  # one small integer per weight
    scale: Array  # broadcasts against ``codes``
    bits: int
    shape: tuple[int, ...]  # the original weight's shape

    def dequantize(self) -> Array:
        return (self.codes * self.scale).reshape(self.shape)

    @property
    def nbytes(self) -> float:
        """Storage if codes are packed tightly, plus the scales."""
        return self.codes.size * self.bits / 8 + self.scale.size * SCALE_BYTES


def quantize(
    w: Array, bits: int = 8, *, shared: tuple[int, ...] | None = None
) -> Quantized:
    """Quantize ``w``; values along the ``shared`` axes share one scale.

    ``shared=None`` gives one scale for the whole tensor. For a weight of
    shape (d_in, d_out), ``shared=(0,)`` gives one scale per output
    channel.
    """
    qmax = 2 ** (bits - 1) - 1
    axes = tuple(range(w.ndim)) if shared is None else shared
    amax: Array = np.max(np.abs(w), axis=axes, keepdims=True)
    scale = np.where(amax > 0, amax / qmax, 1.0)
    codes = np.clip(np.round(w / scale), -qmax, qmax).astype(np.int8)
    return Quantized(codes, scale, bits, w.shape)


def quantize_groups(w: Array, bits: int = 4, group: int = 32) -> Quantized:
    """One scale for every ``group`` consecutive values along each row."""
    rows, cols = w.shape
    if cols % group:
        raise ValueError(f"{cols} columns do not split into groups of {group}")
    q = quantize(w.reshape(rows, cols // group, group), bits, shared=(2,))
    return Quantized(q.codes, q.scale, bits, w.shape)


def relative_rmse(w: Array, w_hat: Array) -> float:
    """Root-mean-square error relative to the weights' own size."""
    return float(np.linalg.norm(w - w_hat) / np.linalg.norm(w))


def to_bfloat16(x: Array) -> Array:
    """Round to the nearest bfloat16 value (ties to even), as float64.

    bfloat16 keeps float32's 8 exponent bits, so its range is the same,
    but only 7 stored mantissa bits: about two to three decimal digits.
    """
    bits = np.asarray(x, dtype=np.float32).view(np.uint32)
    round_up = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    kept = (bits + round_up) & np.uint32(0xFFFF0000)
    return kept.view(np.float32).astype(np.float64)


def quantize_asymmetric(
    w: Array, bits: int = 8
) -> tuple[NDArray[np.uint8], float, int]:
    """Map [min, max] onto 0..2**bits-1; return codes, scale, zero point."""
    levels = 2**bits - 1
    lo, hi = float(w.min()), float(w.max())
    scale = (hi - lo) / levels or 1.0
    zero = int(np.clip(round(-lo / scale), 0, levels))
    codes = np.clip(np.round(w / scale) + zero, 0, levels).astype(np.uint8)
    return codes, scale, zero


def snr_db(w: Array, w_hat: Array) -> float:
    """Signal-to-noise ratio in decibels: 10 log10(signal / noise power)."""
    noise = float(np.sum((w - w_hat) ** 2))
    return float(10 * np.log10(np.sum(w**2) / noise))
