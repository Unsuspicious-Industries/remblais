from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from tinygrad import Tensor

from .ops import flow_to_ops
from .plan import plan
from .render import render


@dataclass
class MorphResult:
    frames: Tensor
    col_err_pct: Tensor
    row_err_pct: Tensor

    def frames_np(self) -> list[np.ndarray]:
        data = self.frames.numpy()
        return [data[i] for i in range(data.shape[0])]


def morph(
    A: np.ndarray,
    B: np.ndarray,
    X: float,
    Y: float,
    epsilon: float = 1.0,
    n_frames: int | None = None,
    normalize: str = "clip",
    num_iters: int = 300,
    dtype: str = "float32",
) -> MorphResult:
    if A.shape != B.shape:
        raise ValueError(f"shape mismatch: A={A.shape} B={B.shape}")

    h, w, c = A.shape
    P = plan(A, B, X, Y, epsilon, num_iters=num_iters, dtype=dtype)
    channels, col_errs, row_errs = [], [], []
    for channel in range(c):
        ops = flow_to_ops(P[channel], A[..., channel], B[..., channel])
        channels.append(render(A[..., channel], ops, n_frames, normalize))
        col_errs.append(ops.col_err_pct)
        row_errs.append(ops.row_err_pct)

    n = max(frame.shape[0] for frame in channels)
    for channel, frame in enumerate(channels):
        if frame.shape[0] < n:
            channels[channel] = frame.cat(frame[-1:].expand(n - frame.shape[0], -1, -1), dim=0)

    out = channels[0].reshape(n, h, w, 1)
    for frame in channels[1:]:
        out = out.cat(frame.reshape(n, h, w, 1), dim=3)
    return MorphResult(out, Tensor(col_errs), Tensor(row_errs))
