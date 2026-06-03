# Results Summary

This document consolidates every experimental result in this internship in one place. Each row corresponds to a notebook in `notebooks/`. For training curves, qualitative panels, and raw `results.json` files, see `results/phaseN/` once they have been downloaded from Kaggle.

## Headline comparison (official RFMiD test split)

| Model | Notebook | Light PSNR | Medium PSNR | **Heavy PSNR** | Light SSIM | Medium SSIM | **Heavy SSIM** | Best val PSNR | Learned c |
|---|---|---|---|---|---|---|---|---|---|
| **Phase 2.1** — Euclidean baseline | `03_*` | 40.98 | 39.61 | **38.39** | 0.954 | 0.939 | **0.925** | ~39.6 | — |
| **Phase 3.1** — Optimized Poincaré | `05_*` | 40.84 | 39.47 | **38.03** | 0.954 | 0.940 | **0.922** | 39.10 | **0.6931** |
| **Phase 4** — Hyperbolic encoder | `06_*` | _running_ | _running_ | _running_ | _running_ | _running_ | _running_ | _running_ | _running_ |

All test sets are n=640 fundus images per noise level. Deterministic noise (fixed RNG seed) ensures every model sees the same noise realizations for the same image.

## Per-noise-level Δ vs Euclidean baseline (Phase 2.1)

| Phase | Light ΔPSNR | Medium ΔPSNR | Heavy ΔPSNR | Mean |
|---|---|---|---|---|
| Phase 3.1 (optimized Poincaré) | −0.14 | −0.14 | −0.36 | −0.21 |
| Phase 4 (hyperbolic encoder) | _running_ | _running_ | _running_ | _running_ |

## Earlier experiments on random 80/10/10 splits (not directly comparable)

These were run before adopting the official RFMiD splits. They are not on the same test set as Phase 2.1+ above, so their absolute numbers should not be compared directly to the post-2.1 results.

| Model | Notebook | Light PSNR | Medium PSNR | Heavy PSNR | Best val PSNR |
|---|---|---|---|---|---|
| Phase 2 — Euclidean (random split) | `02_*` | 41.29 | 39.81 | 38.39 | 39.74 |
| Phase 3 — Poincaré bottleneck (random split, fixed c=0.1) | `04_*` | 40.86 | 39.47 | 37.91 | — |

Phase 3 deltas vs Phase 2: −0.43 / −0.34 / −0.48 dB across the three noise levels.

## Findings and interpretation

### 1. The Euclidean baseline is strong

Phase 2.1 reaches PSNR 38.4 dB at heavy noise, SSIM 0.93. For reference, published retinal-denoising baselines on similar Poisson-Gaussian setups (Cherukuri et al. TMI 2020 and others) sit in the 33–36 dB band — though those use slightly different noise levels and splits, so the comparison isn't apples-to-apples. The key takeaway is that the bar for "interesting result" is set high.

### 2. Partial Poincaré interventions consistently underperform

Both Phase 3 (bottleneck-only) and Phase 3.1 (one level up, residual skip, trainable curvature) lost to their Euclidean counterparts. The optimizations cut the deficit roughly in half (from a mean −0.42 dB to a mean −0.21 dB) but did not reverse the sign. The residual skip in particular *should* have given Phase 3.1 a guaranteed floor at Phase 2.1's numbers — the fact that it didn't suggests the optimizer doesn't drive the hyperbolic contribution to zero. It actively chooses to engage the manifold.

### 3. The curvature trajectory is the most novel observation

In Phase 3.1, the trainable curvature parameter migrates from initialization `c = 0.1` to `c ≈ 0.6931` (≈ ln 2). This is consistent across the training run, not noise. Two interpretations:

- **The Mishra et al. default is suboptimal for our task.** They tuned `c = 0.1` for segmentation; restoration may prefer a more aggressively curved ball.
- **The network is exploiting some structural feature of the curvature parameter space** that we don't fully understand. The proximity to ln 2 is suggestive but probably coincidental — gradient descent landing exactly on a mathematical constant would be surprising given the loss landscape.

This finding is worth a paragraph in the manuscript regardless of which model "wins" overall.

### 4. Vessel softening is the visible failure mode

In all phases, qualitative inspection of denoised outputs reveals that fine vessels are softer than in the ground-truth clean image. This is the well-known pixel-loss-→-blur phenomenon (Johnson et al. ECCV 2016, Ledig et al. CVPR 2017): L1 minimization at uncertain locations returns the conditional median, which is smoother than any individual plausible reconstruction. PSNR and SSIM under-penalize this because they average over all pixels and SSIM's 11×11 window is much larger than a single-pixel-wide vessel.

A perceptual loss component (VGG-based, à la Johnson 2016) or an LPIPS evaluation would surface this gap quantitatively. Not yet implemented; would be the natural Phase 5 follow-up if the architectural search converges.

## Reproducibility checklist

For each entry in the headline table, the following are reproducible from the corresponding notebook:

- ✓ Identical train/val/test split (official RFMiD folders from `andrewmvd/retinal-disease-classification`).
- ✓ Identical noise model (Poisson-Gaussian, three preset levels α=0.5/1.0/2.0, σ=3/5/8).
- ✓ Identical training recipe (L1 loss, AdamW or RiemannianAdam, lr=2e-4, cosine schedule, 30 epochs).
- ✓ Deterministic test-time noise injection (`deterministic_seed=7` per noise level).
- ✓ All hyperparameters frozen in the `Config` dataclass at the top of each notebook.

The only sources of nondeterminism are cuDNN's nondeterministic kernel selection (we enable `benchmark=True` for speed) and AMP gradient scaling for Phases 2 and 2.1. Both are documented sources of small (<0.05 dB) per-run variance.
