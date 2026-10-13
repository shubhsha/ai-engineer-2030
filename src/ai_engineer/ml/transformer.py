"""Chapter 18: a pre-norm transformer decoder, positions and RoPE.

The decoder is small enough to read in one sitting and complete enough
to check the properties that matter: it is causal, its parameter count
matches the formula, and RoPE makes attention depend on relative
position.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ai_engineer.ml.attention import MultiHeadAttention
from ai_engineer.ml.numerics import Array, IntArray


def rms_norm(x: Array, gain: Array, eps: float = 1e-6) -> Array:
    """Rescale each vector to unit root-mean-square, then apply ``gain``."""
    ms: Array = np.mean(x * x, axis=-1, keepdims=True)
    return gain * x / np.sqrt(ms + eps)


def gelu(x: Array) -> Array:
    """GELU activation, tanh approximation (Hendrycks and Gimpel)."""
    inner = math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)
    return 0.5 * x * (1.0 + np.tanh(inner))


def sinusoidal_positions(n: int, d: int) -> Array:
    """The fixed position signals of the original transformer paper."""
    pos = np.arange(n)[:, None]
    freqs = 10_000.0 ** (-np.arange(0, d, 2) / d)
    pe = np.zeros((n, d))
    pe[:, 0::2] = np.sin(pos * freqs)
    pe[:, 1::2] = np.cos(pos * freqs)
    return pe


def apply_rope(x: Array, positions: IntArray, base: float = 10_000.0) -> Array:
    """Rotary position embedding: rotate feature pairs by position.

    Pair i of the vector at position m turns by angle m * base**(-2i/d).
    After rotation, the dot product of a query at position m and a key
    at position n depends only on m - n, so attention sees relative
    position without any learned position table.
    """
    d = x.shape[-1]
    angles = positions[:, None] * base ** (-np.arange(0, d, 2) / d)
    cos, sin = np.cos(angles), np.sin(angles)
    even, odd = x[..., 0::2], x[..., 1::2]
    out = np.empty_like(x)
    out[..., 0::2] = even * cos - odd * sin
    out[..., 1::2] = even * sin + odd * cos
    return out


@dataclass
class Block:
    """Pre-norm decoder block: attention mixes positions, the MLP
    transforms each position, and both add to the residual stream."""

    attn: MultiHeadAttention
    w_up: Array
    w_down: Array
    g_attn: Array
    g_mlp: Array

    @classmethod
    def create(
        cls, d_model: int, heads: int, rng: np.random.Generator
    ) -> Block:
        d_ff = 4 * d_model
        return cls(
            MultiHeadAttention(d_model, heads, rng),
            rng.normal(0.0, d_model**-0.5, (d_model, d_ff)),
            rng.normal(0.0, d_ff**-0.5, (d_ff, d_model)),
            np.ones(d_model),
            np.ones(d_model),
        )

    def __call__(self, x: Array) -> Array:
        x = x + self.attn.forward(rms_norm(x, self.g_attn))
        return x + gelu(rms_norm(x, self.g_mlp) @ self.w_up) @ self.w_down


@dataclass
class TinyDecoder:
    """Embeddings, a stack of blocks and a tied output projection."""

    embed: Array
    blocks: list[Block]
    g_final: Array

    @classmethod
    def create(
        cls, *, vocab: int, d_model: int, heads: int, layers: int, seed: int
    ) -> TinyDecoder:
        rng = np.random.default_rng(seed)
        return cls(
            rng.normal(0.0, d_model**-0.5, (vocab, d_model)),
            [Block.create(d_model, heads, rng) for _ in range(layers)],
            np.ones(d_model),
        )

    def logits(self, tokens: IntArray) -> Array:
        """Next-token scores at every position, shape (tokens, vocab)."""
        d = self.embed.shape[1]
        x = self.embed[tokens] + sinusoidal_positions(len(tokens), d)
        for block in self.blocks:
            x = block(x)
        return rms_norm(x, self.g_final) @ self.embed.T

    def parameter_count(self) -> int:
        weights = [self.embed, self.g_final]
        for b in self.blocks:
            a = b.attn
            weights += [a.w_q, a.w_k, a.w_v, a.w_o, b.w_up, b.w_down]
            weights += [b.g_attn, b.g_mlp]
        return sum(w.size for w in weights)


def layer_norm(x: Array, gain: Array, bias: Array, eps: float = 1e-5) -> Array:
    """Subtract each vector's mean, divide by its standard deviation,
    then scale by ``gain`` and shift by ``bias`` (Ba, Kiros, Hinton)."""
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    out: Array = gain * (x - mean) / np.sqrt(var + eps) + bias
    return out


def silu(x: Array) -> Array:
    """x times its own sigmoid: a smooth bend, also called swish."""
    out: Array = x / (1.0 + np.exp(-x))
    return out


def swiglu(x: Array, w_gate: Array, w_up: Array, w_down: Array) -> Array:
    """Gated MLP: a SiLU-bent branch multiplies a linear one (Shazeer)."""
    out: Array = (silu(x @ w_gate) * (x @ w_up)) @ w_down
    return out


def alibi_slopes(heads: int) -> Array:
    """The slope of each head: powers of 2**(-8/heads) (Press et al.)."""
    ratio = 2.0 ** (-8.0 / heads)
    slopes: Array = ratio ** np.arange(1, heads + 1, dtype=np.float64)
    return slopes


def alibi_bias(n: int, slopes: Array) -> Array:
    """Add to the scores before the softmax: -slope * distance back.

    Shape (heads, n, n). Entries above the diagonal are zero here; the
    causal mask removes those positions.
    """
    pos = np.arange(n)
    back = np.maximum(pos[:, None] - pos[None, :], 0)
    bias: Array = -slopes[:, None, None] * back
    return bias
