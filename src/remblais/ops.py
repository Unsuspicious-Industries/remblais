from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from tinygrad import Tensor, dtypes


@dataclass
class OpsBundle:
    pass_ops: np.ndarray
    pass_counts: np.ndarray
    shift_ops: np.ndarray
    n_pass_steps: int
    col_err_pct: float
    row_err_pct: float


def _manhattan_path(sy: int, sx: int, dy: int, dx: int, w: int, rng) -> list[int]:
    path, y, x = [sy * w + sx], sy, sx
    while y != dy or x != dx:
        ry, rx = abs(dy - y), abs(dx - x)
        vertical = rx == 0 or (ry > 0 and rng.random() < ry / (ry + rx))
        y, x = (y + (1 if dy > y else -1), x) if vertical else (y, x + (1 if dx > x else -1))
        path.append(y * w + x)
    return path


def flow_to_ops(P: Tensor, A: np.ndarray, B: np.ndarray, interleave: bool = True) -> OpsBundle:
    h, w = A.shape
    n, rng = h * w, np.random.default_rng()
    src = A.flatten().astype(np.int64)
    p = np.maximum(P.cast(dtypes.float32).numpy().astype(np.float64), 0)
    flow, frac = np.floor(p).astype(np.int64), p % 1

    for q in range(n):
        budget = int(src[q] - flow[:, q].sum())
        if budget > 0:
            flow[np.argpartition(frac[:, q], -budget)[-budget:], q] += 1
        elif budget < 0:
            for _ in range(-budget):
                dst = int(np.argmax(flow[:, q]))
                if flow[dst, q] == 0:
                    break
                flow[dst, q] -= 1

    rows = []
    for dst, source in zip(*np.nonzero(flow)):
        if dst == source:
            continue
        path = _manhattan_path(source // w, source % w, dst // w, dst % w, w, rng)
        rows.extend(
            (a, b, step, flow[dst, source]) for step, (a, b) in enumerate(zip(path, path[1:]))
        )

    if rows:
        data = np.array(rows, dtype=np.int32)
        passes, counts, steps = data[:, :3], data[:, 3], int(data[:, 2].max()) + 1
    else:
        passes, counts, steps = np.empty((0, 3), np.int32), np.empty(0, np.int32), 0

    state = src.astype(np.float64)
    if steps:
        source, dst = passes[:, 0].astype(np.intp), passes[:, 1].astype(np.intp)
        if interleave:
            for step in range(steps):
                mask = passes[:, 2] == step
                np.add.at(state, source[mask], -counts[mask])
                np.add.at(state, dst[mask], counts[mask])
        else:
            np.add.at(state, source, -counts)
            np.add.at(state, dst, counts)

    diff = B.flatten().astype(np.int64) - np.round(state).astype(np.int64)
    px = np.flatnonzero(diff)
    shifts = np.stack((px, diff[px]), 1).astype(np.int32).reshape(-1, 2)
    mass = max(float(src.sum()), 1.0)
    col_err = float(np.abs(flow.sum(0) - src).sum() / mass * 100)
    row_err = float(np.abs(flow.sum(1) - B.flatten()).sum() / mass * 100)
    return OpsBundle(passes, counts, shifts, steps, col_err, row_err)
