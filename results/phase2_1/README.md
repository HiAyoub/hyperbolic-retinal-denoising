# Phase 2.1 — results

Per-phase outputs from the corresponding notebook.

After running [`notebooks/03_euclidean_baseline_official_splits.ipynb`](../../notebooks/03_euclidean_baseline_official_splits.ipynb) on Kaggle, download the following artifacts from `/kaggle/working/` into this folder:

- `results.json` (or `results_v2.json` / `results_hyp_v2.json` / `results_henc.json` depending on phase)
- `history.json`
- `training_curves*.png`
- `final_qualitative*.png`
- Selected `viz_epoch*.png` if you want to track training progression

Do **not** commit the `unet_denoiser_best*.pt` checkpoint — those are large and regenerable. Hyperlink to a cloud-hosted copy in the parent results table if you need to share trained weights.
