from __future__ import annotations

import numpy as np
from tinygrad import Tensor, dtypes

_DTYPES = {"float32": dtypes.float32, "float16": dtypes.float16, "half": dtypes.float16}


def _logsumexp(x: Tensor, axis: int) -> Tensor:
    m = x.max(axis=axis, keepdim=True)
    shape = [s for i, s in enumerate(x.shape) if i != axis]
    return (x - m).exp().sum(axis=axis).log() + m.reshape(shape)


def _pairwise_l1(h: int, w: int, dtype) -> Tensor:
    n = h * w
    ys = Tensor.arange(h).cast(dtype).reshape(h, 1).expand(h, w).reshape(n, 1)
    xs = Tensor.arange(w).cast(dtype).reshape(1, w).expand(h, w).reshape(n, 1)
    xy = ys.cat(xs, dim=1)
    return (xy.reshape(n, 1, 2) - xy.reshape(1, n, 2)).abs().sum(2)


def _mass(x: np.ndarray | Tensor, h: int, w: int, c: int, dtype) -> Tensor:
    if isinstance(x, Tensor):
        return x.reshape(h * w, c).permute(1, 0).cast(dtype)
    return Tensor(x.reshape(h * w, c).astype(np.float32)).permute(1, 0).cast(dtype)


def plan(
    A: np.ndarray | Tensor,
    B: np.ndarray | Tensor,
    X: float,
    Y: float,
    epsilon: float = 1.0,
    num_iters: int = 1000,
    dtype: str = "float32",
) -> Tensor:
    dt = _DTYPES.get(dtype)
    if dt is None:
        raise ValueError(f"dtype must be one of {list(_DTYPES)}; got {dtype!r}")

    h, w, c = A.shape
    n, log_k = h * w, -X * _pairwise_l1(h, w, dt) / epsilon
    eps = Tensor([1e-7]).cast(dt)
    a, b = _mass(B, h, w, c, dt).maximum(eps).log(), _mass(A, h, w, c, dt).maximum(eps).log()
    tau = float(Y / (Y + epsilon))
    u, v = Tensor.zeros(c, n).cast(dt), Tensor.zeros(c, n).cast(dt)
    for _ in range(num_iters):
        u = (tau * (a - _logsumexp(log_k.reshape(1, n, n) + v.reshape(c, 1, n), 2))).realize()
        v = (tau * (b - _logsumexp(log_k.T.reshape(1, n, n) + u.reshape(c, 1, n), 2))).realize()
    return (u.reshape(c, n, 1) + log_k.reshape(1, n, n) + v.reshape(c, 1, n)).exp().float()
