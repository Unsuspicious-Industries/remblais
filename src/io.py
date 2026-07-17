from pathlib import Path

import numpy as np
from PIL import Image


def load_image(path: str | Path, size: tuple[int, int] | None = None) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    return np.array(image.resize(size) if size else image, dtype=np.uint8)


def mass_delta(A: np.ndarray, B: np.ndarray) -> float:
    return float(np.abs(A.sum() - B.sum()))


def mass_delta_rgb(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.array([mass_delta(A[..., c], B[..., c]) for c in range(3)])


def save_video(frames: list[np.ndarray], path: str | Path, fps: int = 24) -> None:
    path = Path(path)
    if path.suffix.lower() in {".gif", ".webp"}:
        images = [Image.fromarray(frame) for frame in frames]
        images[0].save(
            path,
            save_all=True,
            append_images=images[1:],
            duration=max(1, 1000 // fps),
            loop=0,
            lossless=path.suffix.lower() == ".webp",
        )
    elif path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}:
        import imageio

        imageio.mimsave(str(path), frames, fps=fps)
    else:
        raise ValueError("output must be GIF, WebP, MP4, AVI, MOV, or MKV")
