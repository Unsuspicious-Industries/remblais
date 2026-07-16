from __future__ import annotations

import numpy as np
from tinygrad import Tensor

from .ops import OpsBundle


def _normalize(out: np.ndarray, mode: str) -> np.ndarray:
    if mode == "clip":
        return np.clip(out, 0, 255).astype(np.uint8)
    if mode == "global":
        lo, hi = out.min(), out.max()
        return np.clip((out - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
    if mode == "tonemap":
        out = np.maximum(out, 0)
        return (255 * out / (255 + out)).astype(np.uint8)
    if mode == "none":
        return out
    raise ValueError(f"unknown normalize mode: {mode!r}")


def render(
    A: np.ndarray,
    ops: OpsBundle,
    n_frames: int | None = None,
    normalize: str = "clip",
) -> Tensor:
    h, w = A.shape
    state = A.flatten().astype(np.float64)
    k, m = len(ops.pass_ops), len(ops.shift_ops)
    if k == 0 and m == 0:
        return Tensor(np.broadcast_to(A.reshape(1, h, w), (1, h, w)))

    order = np.argsort(ops.pass_ops[:, 2]) if k else slice(None)
    source = ops.pass_ops[order, 0].astype(np.intp) if k else np.empty(0, np.intp)
    dst = ops.pass_ops[order, 1].astype(np.intp) if k else np.empty(0, np.intp)
    counts = ops.pass_counts[order].astype(np.float64) if k else np.empty(0)
    steps = ops.pass_ops[order, 2] if k else np.empty(0, np.int32)
    record_at = (
        None if n_frames is None else set(np.round(np.linspace(0, k + m, n_frames)).astype(int))
    )
    if record_at is not None:
        record_at.discard(0)
    frames = [state.copy()]

    if k:
        current_step = steps[0]
        for i in range(k):
            state[source[i]] -= counts[i]
            state[dst[i]] += counts[i]
            if record_at is None and (i == k - 1 or steps[i + 1] != current_step):
                frames.append(state.copy())
                if i < k - 1:
                    current_step = steps[i + 1]
            elif record_at is not None and i + 1 in record_at:
                frames.append(state.copy())

    if m:
        for px, delta in ops.shift_ops:
            state[px] += delta
        if record_at is None or k + m in record_at:
            frames.append(state.copy())

    out = np.stack(frames).reshape(len(frames), h, w)
    return Tensor(_normalize(out, normalize) if normalize != "none" else out)
