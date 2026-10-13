"""Chapter 17: convolution and recurrence, two ways to share parameters.

A convolution reuses one small kernel at every position. A recurrent
network reuses one cell at every time step. Each builds in an assumption
about the data, and each assumption has a cost that attention avoids.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from ai_engineer.ml.linear import sigmoid
from ai_engineer.ml.numerics import Array


def conv_output_size(n: int, k: int, pad: int = 0, stride: int = 1) -> int:
    """Positions a size-``k`` window visits on ``n`` inputs.

    ``pad`` zeros are added on both sides first, and the window moves
    ``stride`` steps at a time: ``(n + 2 * pad - k) // stride + 1``.
    """
    return (n + 2 * pad - k) // stride + 1


def conv1d(x: Array, kernel: Array, stride: int = 1) -> Array:
    """Slide ``kernel`` along the last axis of ``x`` ("valid" padding).

    Like every deep-learning library, this computes cross-correlation:
    the kernel is not flipped. Output length is
    ``(len(x) - len(kernel)) // stride + 1``.
    """
    windows = sliding_window_view(x, kernel.shape[-1], axis=-1)
    return windows[..., ::stride, :] @ kernel


def conv1d_as_matrix(kernel: Array, n: int) -> Array:
    """The dense weight matrix that computes ``conv1d`` on ``n`` inputs.

    Row ``i`` holds the kernel at columns ``i`` to ``i + len(kernel) - 1``
    and zeros elsewhere: a dense layer with locality (the zeros) and
    sharing (the repeated kernel) forced on it.
    """
    k = kernel.shape[-1]
    matrix = np.zeros((n - k + 1, n))
    for i in range(matrix.shape[0]):
        matrix[i, i : i + k] = kernel
    return matrix


def conv2d(
    image: Array, kernel: Array, pad: int = 0, stride: int = 1
) -> Array:
    """2-D cross-correlation of an (H, W) image with a small kernel.

    ``pad`` rings of zeros go around the image first, and the window
    moves ``stride`` pixels at a time. With the defaults the result is
    the "valid" one, smaller than the image by ``kernel - 1``.
    """
    windows = sliding_window_view(np.pad(image, pad), kernel.shape)
    out: Array = np.einsum("ijkl,kl->ij", windows[::stride, ::stride], kernel)
    return out


def max_pool2d(image: Array, size: int = 2) -> Array:
    """Keep the largest value in each ``size`` x ``size`` tile."""
    h, w = (d - d % size for d in image.shape)
    tiles = image[:h, :w].reshape(h // size, size, w // size, size)
    pooled: Array = tiles.max(axis=(1, 3))
    return pooled


def receptive_field(layers: Sequence[tuple[int, int]]) -> int:
    """Input positions one output unit sees, for ``(kernel, stride)`` layers.

    Each layer adds ``kernel - 1`` steps, measured in input positions:
    the spacing between neighboring units, which every stride multiplies.
    """
    field, spacing = 1, 1
    for kernel, stride in layers:
        field += (kernel - 1) * spacing
        spacing *= stride
    return field


def conv_budget(
    out_size: int, k: int, pairs: int, maps_out: int
) -> tuple[int, int]:
    """Parameters and multiply-adds of a convolutional layer.

    ``pairs`` counts the (input map, output map) connections. Each carries
    one ``k`` x ``k`` kernel, and each output map has one bias. The kernels
    are reused at all ``out_size`` squared positions, so the arithmetic
    grows with the image while the parameters do not.
    """
    weights = pairs * k * k
    return weights + maps_out, weights * out_size**2


def dense_budget(n_in: int, n_out: int) -> tuple[int, int]:
    """Parameters and multiply-adds of a dense layer: one use per weight."""
    return n_out * (n_in + 1), n_out * n_in


def rnn_forward(
    xs: Array,
    w_x: Array,
    w_h: Array,
    b: Array,
    h0: Array | None = None,
) -> Array:
    """Run a simple (Elman) RNN over ``xs`` of shape (steps, d_in).

    Returns every hidden state, shape (steps, d_hidden). The loop is
    inherently sequential: step t cannot start until step t - 1 ends.
    """
    h = np.zeros(w_h.shape[0]) if h0 is None else h0
    states = []
    for x in xs:
        h = np.tanh(x @ w_x + h @ w_h + b)
        states.append(h)
    return np.stack(states)


def sensitivity(states: Array, w_h: Array) -> Array:
    """Size of d(h_last)/d(h_t) for every step t; the last entry is 1.

    Each step multiplies the Jacobian by diag(1 - h**2) @ w_h.T. When
    those factors are smaller than one, the product shrinks
    geometrically and early inputs stop influencing the gradient: the
    vanishing-gradient problem that LSTMs eased and attention removed.
    """
    jac = np.eye(w_h.shape[0])
    norms = [1.0]
    for h in states[:0:-1]:  # h_last down to h_1
        jac = jac @ ((1 - h**2)[:, None] * w_h.T)
        norms.append(float(np.linalg.norm(jac, 2)))
    return np.array(norms[::-1])


def clip_global_norm(
    grads: Sequence[Array], max_norm: float
) -> tuple[list[Array], float]:
    """Shrink gradients so their joint L2 norm is at most ``max_norm``.

    Returns the clipped gradients and the norm before clipping. Only the
    length of the step changes, never its direction (Pascanu et al.).
    """
    total = float(np.sqrt(sum(float(np.sum(g**2)) for g in grads)))
    scale = min(1.0, max_norm / total) if total > 0 else 1.0
    return [g * scale for g in grads], total


def lstm_step(
    x: Array, h: Array, c: Array, w_x: Array, w_h: Array, b: Array
) -> tuple[Array, Array]:
    """One LSTM step; ``w_x``, ``w_h`` and ``b`` hold 4 blocks of width d.

    The blocks are the input gate, forget gate, candidate and output
    gate. The cell state changes by addition, ``f * c + i * g``, so
    its gradient is multiplied by the forget gate and nothing else.
    """
    d = h.shape[0]
    z = x @ w_x + h @ w_h + b
    i, f = sigmoid(z[:d]), sigmoid(z[d : 2 * d])
    g, o = np.tanh(z[2 * d : 3 * d]), sigmoid(z[3 * d :])
    c_new = f * c + i * g
    return o * np.tanh(c_new), c_new


def gru_step(x: Array, h: Array, w_x: Array, w_h: Array, b: Array) -> Array:
    """One GRU step (Cho et al. 2014); 3 blocks of width d.

    The blocks are the update gate z, the reset gate r and the candidate.
    The new state is ``z * h + (1 - z) * candidate``.
    """
    d = h.shape[0]
    zx, zh = x @ w_x, h @ w_h[:, : 2 * d]
    z = sigmoid(zx[:d] + zh[:d] + b[:d])
    r = sigmoid(zx[d : 2 * d] + zh[d : 2 * d] + b[d : 2 * d])
    cand = np.tanh(zx[2 * d :] + (r * h) @ w_h[:, 2 * d :] + b[2 * d :])
    return z * h + (1 - z) * cand
