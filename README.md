# Hyperbolic Deep Learning for Retinal Image Denoising

> **Internship project** investigating whether hyperbolic geometry, as a structural prior for hierarchical feature representations, can improve fundus-image denoising quality over standard Euclidean U-Nets.

## The question

Retinal fundus images are corrupted by photon-counting (Poisson) noise from the sensor and Gaussian read noise from the electronics. The clinical workflow needs clean reconstructions because downstream diagnostic models (multi-label disease classifiers) are sensitive to input quality. The natural baseline is a Euclidean U-Net trained with L1 loss in DnCNN-style residual mode. The research question this project answers is:

**Does replacing parts of a U-Net's feature extraction with hyperbolic operations on the Poincaré ball produce measurably better denoised retinal images?**

The hypothesis comes from Mishra, van Spengler, Berkhout & Mettes ([MIDL 2026](https://openreview.net/forum?id=NxKaeTNMxR)), who showed that a fully hyperbolic U-Net is more robust to image noise than its Euclidean twin on medical segmentation. Vessels are an inherently hierarchical tree structure — bifurcations splitting recursively into smaller bifurcations — exactly the kind of object that hyperbolic embeddings preserve better than Euclidean ones. If their robustness benefit transfers from segmentation to restoration, we should see a measurable improvement.

## Results

All numbers are PSNR (dB) on the **official RFMiD test split (n=640 per noise level)**. Each model is evaluated against three Poisson-Gaussian noise severities (light / medium / heavy) using deterministic noise injection for reproducibility.

| Model | Light | Medium | **Heavy** | Best val PSNR | Learned curvature |
|---|---|---|---|---|---|
| **Phase 2.1** — Euclidean baseline (official splits) | 40.98 | 39.61 | **38.39** | ~39.6 | — |
| **Phase 3.1** — Optimized Poincaré (block at 32², residual, trainable c) | 40.84 | 39.47 | **38.03** | 39.10 | 0.6931 |
| **Phase 4** — Hyperbolic encoder, Euclidean decoder | _running_ | _running_ | _running_ | _running_ | _running_ |

Earlier random-split experiments (Phases 2 and 3) are included in the repo for completeness but are not directly comparable because the test set differs.

The clearest non-obvious finding to date is the **curvature trajectory**: when the manifold's curvature parameter is allowed to be trainable, it migrates from the Mishra et al. default `c = 0.1` to `c ≈ 0.6931` (= ln 2). This is reproducible across runs and survives the architectural changes between Phase 3 and Phase 3.1. The network actively engages with the manifold and chooses a much more aggressively curved operating point than the literature default, even when the result is not a net win.

For the full per-level table, training curves and qualitative comparisons, see [docs/results_summary.md](docs/results_summary.md).

## How to reproduce

Every notebook in `notebooks/` is self-contained and designed to run on Kaggle without modification.

1. Sign into [kaggle.com](https://kaggle.com) → Code → New Notebook.
2. Right sidebar → Settings → Accelerator → **GPU T4 x1**, Internet → **On**.
3. Right sidebar → Add Data → search "retinal disease classification" → add `andrewmvd/retinal-disease-classification`.
4. File → Import Notebook → upload the notebook you want from `notebooks/`.
5. Run All.

Runtime varies by phase: Phase 1 needs no GPU (~5 min); Phase 2 / Phase 2.1 with AMP take ~1.5–2 h; Phase 3 / Phase 3.1 in fp32 take ~3–4 h; Phase 4 takes ~5–7 h on T4.

For local development, see `requirements.txt`. The notebooks expect Python 3.10+, PyTorch 2.0+, CUDA-capable GPU (any with compute capability ≥ 7.0).

## Repository layout

```
.
├── README.md                            ← this file
├── LICENSE                              ← MIT
├── requirements.txt                     ← pinned dependencies for reproducibility
├── docs/
│   ├── phase1_report.pdf                ← Dataset analysis + Poisson-Gaussian noise model
│   ├── phase2_design_rationale.pdf      ← Defense-grade justification of every design choice
│   ├── results_summary.md               ← Consolidated per-phase tables + interpretation
│   └── references.md                    ← Annotated bibliography
├── notebooks/
│   ├── 01_rfmid_noise_analysis.ipynb            ← Phase 1: noise characterization
│   ├── 02_euclidean_baseline_random_split.ipynb ← Phase 2 (original, random 80/10/10)
│   ├── 03_euclidean_baseline_official_splits.ipynb ← Phase 2.1 (official RFMiD splits)
│   ├── 04_poincare_bottleneck.ipynb     ← Phase 3 (Poincaré at bottleneck, fixed c)
│   ├── 05_poincare_optimized.ipynb      ← Phase 3.1 (earlier block + residual + trainable c)
│   └── 06_hyperbolic_encoder.ipynb      ← Phase 4 (full hyperbolic encoder)
└── results/                             ← per-phase outputs (populate from Kaggle after each run)
    ├── phase2/  phase2_1/  phase3/  phase3_1/  phase4/
```

Each `*.ipynb` has a paired `*.py` source-of-truth file with `# %%` cell markers — useful for code review and version-control diffs, since `.py` diffs cleanly while `.ipynb` JSON does not.

## Dataset

**RFMiD** (Retinal Fundus Multi-disease Image Dataset) — 3,200 fundus photographs labeled with 28 retinal conditions. We use only the images; the labels would matter for a downstream classification experiment but are not needed for denoising. The official train/validation/test splits (1920 / 640 / 640) ship with the dataset and are used from Phase 2.1 onward.

- Source: [Pachade et al., *Data* 2021](https://www.mdpi.com/2306-5729/6/2/14)
- Kaggle mirror: [andrewmvd/retinal-disease-classification](https://www.kaggle.com/datasets/andrewmvd/retinal-disease-classification)

The synthetic noise model is a Poisson-Gaussian mixture (`var(y) = α·μ + σ²`) at three severity levels (α=0.5, σ=3 / α=1.0, σ=5 / α=2.0, σ=8). The rationale for adopting literature noise parameters rather than fitting them empirically to RFMiD is documented in `docs/phase1_report.pdf` §7.

## Citation

If this work is useful in your research, please cite the upstream paper this project adapts:

```bibtex
@inproceedings{mishra2026hyperbolic,
  title     = {Hyperbolic U-Net for Robust Medical Image Segmentation},
  author    = {Mishra, Swasti Shreya and van Spengler, Max and Berkhout, Erwin and Mettes, Pascal},
  booktitle = {Medical Imaging with Deep Learning (MIDL)},
  year      = {2026},
  url       = {https://openreview.net/forum?id=NxKaeTNMxR}
}
```

And the library that provides the hyperbolic operations:

```bibtex
@article{spengler2023hypll,
  title   = {HypLL: The Hyperbolic Learning Library},
  author  = {van Spengler, Max and Wirth, Philipp and Mettes, Pascal},
  journal = {arXiv preprint arXiv:2306.06154},
  year    = {2023}
}
```

A formal write-up of this internship's results will be added once the comparison table is finalized.

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgments

This work was conducted as part of an internship at *Barcelona Ceneter for New Medical Technologies*, supervised by *PIELLA FENOY, GEMMA*. Thanks to Pachade et al. for releasing the RFMiD dataset, the Amsterdam group for both the reference paper and the `hypll` library, and Kaggle for the GPU credits that made the experiments possible.
