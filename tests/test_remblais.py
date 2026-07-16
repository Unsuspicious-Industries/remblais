import numpy as np

from remblais import OpsBundle, morph, render
from remblais.cli import parser


def bundle(moves: np.ndarray, counts: np.ndarray, shifts: np.ndarray) -> OpsBundle:
    steps = int(moves[:, 2].max()) + 1 if len(moves) else 0
    return OpsBundle(moves, counts, shifts, steps, 0, 0)


def test_render_reaches_endpoint() -> None:
    source = np.array([[2, 0]], dtype=np.uint8)
    ops = bundle(
        np.array([[0, 1, 0]], dtype=np.int32),
        np.array([2], dtype=np.int32),
        np.empty((0, 2), dtype=np.int32),
    )

    frames = render(source, ops, n_frames=5).numpy()

    np.testing.assert_array_equal(frames[0], source)
    np.testing.assert_array_equal(frames[-1], [[0, 2]])


def test_unchanged_image_has_one_frame() -> None:
    source = np.array([[2]], dtype=np.uint8)
    ops = bundle(
        np.empty((0, 3), dtype=np.int32),
        np.empty(0, dtype=np.int32),
        np.empty((0, 2), dtype=np.int32),
    )

    assert render(source, ops, n_frames=4).shape == (1, 1, 1)


def test_cli_exposes_morph_command() -> None:
    args = parser().parse_args(["morph", "a.png", "b.png"])

    assert args.command == "morph"
    assert args.frames is None


def test_morph_reaches_target() -> None:
    source = np.zeros((2, 2, 1), dtype=np.uint8)
    target = np.zeros_like(source)
    source[0, 0] = target[1, 1] = 255

    frames = morph(source, target, X=1, Y=1e10, n_frames=3, num_iters=2).frames.numpy()

    np.testing.assert_array_equal(frames[0], source)
    np.testing.assert_array_equal(frames[-1], target)
