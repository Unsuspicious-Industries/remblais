<div align="center">

# REMBLAIS

**Do not cross-fade the pixels. Move the mass.**

<img src="morph_big.gif" width="560" alt="A white block transported into three colored blocks" />

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Unsuspicious-Industries/remblais/blob/main/demo.ipynb)
[![tinygrad](https://img.shields.io/badge/compute-tinygrad-black)](https://github.com/tinygrad/tinygrad)
[![MIT](https://img.shields.io/badge/license-MIT-8a2be2)](LICENSE)

</div>

Remblais turns an optimal transport plan into an image morph. Brightness is mass. Pixels are sites. Color moves through the grid instead of dissolving through a cross-fade.

In earthworks, *remblai* is the material placed to fill a void. This experiment applies the same cut-and-fill view to images.

## Run it

Directly from GitHub:

```bash
uvx --from git+https://github.com/Unsuspicious-Industries/remblais remblais morph source.png target.png --size 64 64
```

From a checkout, with the reproducible Nix environment:

```bash
nix develop
uv sync
uv run remblais morph source.png target.png --out morph.gif
```

The output can be GIF, animated WebP, or a common video format. Keep inputs small. The dense transport plan uses quadratic memory in the pixel count.

```python
from remblais import load_image, morph, save_video

a = load_image("source.png", size=(64, 64))
b = load_image("target.png", size=(64, 64))
result = morph(a, b, X=1.0, Y=1e10, n_frames=24)
frames = result.frames_np()
save_video(frames, "morph.gif")
```

The [Colab notebook](https://colab.research.google.com/github/Unsuspicious-Industries/remblais/blob/main/demo.ipynb) builds the example from scratch and exposes the transport parameters.

## The idea

An OT coupling says how much mass should move between every pair of pixels. It does not say how that movement should unfold in time. Remblais supplies that missing dynamics.

For each color channel, it computes the entropic unbalanced transport plan

```math
\begin{aligned}
P^\star = \underset{P \geq 0}{\operatorname{arg\,min}} \quad
& \lambda \langle C, P \rangle
+ \varepsilon \sum_{ij} P_{ij}(\log P_{ij} - 1) \\
& + \rho\,\mathrm{KL}(P\mathbf{1} \mathbin\| b)
+ \rho\,\mathrm{KL}(P^\top\mathbf{1} \mathbin\| a).
\end{aligned}
```

where $C_{ij}$ is Manhattan distance on the image grid. A log-domain generalized Sinkhorn iteration runs in tinygrad.

The coupling is then lifted into an animation:

1. **Conservative discretization.** Largest-remainder rounding reconciles each plan column with its source pixel's integer mass.
2. **Local path decomposition.** Every nonzero coupling edge is factored along a random shortest Manhattan path.
3. **Conservative dynamics.** A routed step has the form $x \leftarrow x + m(e_j-e_i)$, so transported mass is preserved at every local move.
4. **Explicit residual.** Unbalanced marginal error is applied as the final cut-and-fill field $r=b-\tilde b$. Under the default clipped render, the last latent state reaches the target rather than an approximate barycenter.

This coupling-to-dynamics construction is the experiment. Sinkhorn finds global intent. Grid routing gives it local motion. The residual makes creation and destruction visible instead of hiding it inside a blend.

<div align="center">
  <img src="morph.gif" width="300" alt="32 by 32 Remblais morph" />
  <br />
  <sub>The same construction on a 32 x 32 grid. Independent RGB plans split white mass into color.</sub>
</div>

## Controls

| Option | Symbol | Effect |
|---|---:|---|
| `--X`, `--distance-weight` | $\lambda$ | Higher values resist long motion and favor cut-and-fill |
| `--Y`, `--mass-penalty` | $\rho$ | Higher values push the plan toward balanced marginals |
| `--epsilon` | $\varepsilon$ | Higher values spread the coupling |
| `--iters` | | More generalized Sinkhorn updates |

```bash
uv run remblais morph --help
```

## Scope

Remblais is a small experiment, not a production image model. Inputs are uint8 RGB images and channels are transported independently. The dense plan scales as $O((HW)^2)$. Downsample first, then upscale the animation if needed.

## Development

```bash
nix develop -c uv sync
nix develop -c uv run ruff check .
nix develop -c uv run pytest
```

MIT licensed.
