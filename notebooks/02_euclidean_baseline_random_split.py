# %% [markdown]
# # RFMiD Denoising U-Net — Phase 2 (Euclidean baseline)
#
# **Goal.** Train an Euclidean U-Net to denoise RFMiD fundus images corrupted
# with Poisson-Gaussian noise. This is the **Phase 2** deliverable of the
# "Hyperbolic Deep Learning for Retinal Image Denoising" internship — a clean
# baseline that will later be compared head-to-head with a hyperbolic variant
# (Phase 3, `--model hyp`).
#
# **Why a Euclidean baseline first?** Without an apples-to-apples Euclidean
# reference trained on the same data, with the same architecture, same loss,
# and the same training recipe, any later hyperbolic improvement would be
# uninterpretable — we wouldn't know whether the gain came from geometry or
# from some unrelated implementation difference. See `phase2_design_rationale.pdf`
# for the full justification of every design decision in this notebook.
#
# **Design summary (all defended in the rationale PDF).**
# - **Architecture:** 4-level U-Net, channel widths 32 / 64 / 128 / 256 / 512,
#   GroupNorm + ReLU, bilinear upsampling. Mirrors the upstream
#   `swastishreya/Hyperbolic-U-Net` (MIDL 2026) reference.
# - **Output head:** linear 3-channel head predicting the *residual noise*
#   (DnCNN-style). Forward returns `noisy − noise_pred`.
# - **Loss:** L1 (MAE) — sharper than MSE for image restoration (Zhao 2017).
# - **Patches:** 256×256 random crops.
# - **Optimizer:** AdamW, cosine annealing, mixed precision (AMP).
# - **Noise:** sampled per-image at training time from three levels
#   (light / medium / heavy) — "blind" denoising regime.
# - **Metrics:** PSNR + SSIM, broken down by noise level.
#
# **How to run this on Kaggle.**
# 1. kaggle.com → Code → New Notebook → switch the accelerator to **GPU T4**.
# 2. Right sidebar: Settings → toggle **Internet ON** (needed for `pip install`).
# 3. Right sidebar: Add Data → search "retinal disease classification" → add
#    `andrewmvd/retinal-disease-classification`.
# 4. Copy each cell of this script into a notebook cell (cells are separated by
#    `# %%` markers — Kaggle, Jupyter and VS Code all recognize them) — or
#    upload the `.ipynb` directly via File → Import Notebook.
# 5. Run all cells. End-to-end training is ~1.5–2 h on a single T4 at the
#    default settings (30 epochs, batch size 16, 256² patches).

# %% [markdown]
# ## 0. Install extra packages

# %% Install
# torchmetrics gives us a clean, GPU-friendly PSNR/SSIM. Everything else is
# already on the Kaggle image. We don't install hypll here — that's a Phase 3
# concern.
import subprocess, sys
def _pip(*pkgs):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])

try:
    import torchmetrics  # noqa: F401
except ImportError:
    _pip("torchmetrics==1.4.0")

# %% [markdown]
# ## 1. Imports & global configuration

# %% Imports
import os
import json
import math
import random
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler

from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

# Reproducibility (within the limits of cuDNN nondeterminism)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
# cuDNN benchmark: ON for speed; we accept the small variance for ~10-15% speedup.
torch.backends.cudnn.benchmark = True

plt.rcParams["figure.dpi"] = 100

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch {torch.__version__}  |  device = {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}  |  "
          f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# %% [markdown]
# ## 2. Configuration

# %% Config dataclass
@dataclass
class Config:
    # ---- Data ----
    data_root: str = "/kaggle/input/retinal-disease-classification"
    out_dir:   str = "/kaggle/working"
    image_size: int = 384         # all images resized to this on load (keeps memory bounded)
    patch_size: int = 256         # random crop size for training
    train_frac: float = 0.80      # train / val / test = 80 / 10 / 10
    val_frac:   float = 0.10
    num_workers: int = 2          # Kaggle is happiest with 2

    # ---- Model ----
    base_channels: int = 32       # gives widths 32/64/128/256/512 across 4 down/ups
    depth: int = 4

    # ---- Optimization ----
    batch_size: int = 16
    epochs: int = 30
    lr: float = 2e-4              # AdamW default-ish; cosine-annealed to 0
    weight_decay: float = 1e-4
    warmup_epochs: int = 1        # short linear warmup before cosine kicks in
    grad_clip: float = 1.0
    amp: bool = True              # mixed precision — ~2× speed on T4

    # ---- Eval / I/O ----
    eval_every: int = 1           # validate every epoch
    viz_every: int = 5            # save a before/after PNG every N epochs
    save_best_only: bool = True   # only keep the best-val checkpoint

cfg = Config()

# Ensure output directory exists (Kaggle has /kaggle/working ready, but harmless)
Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
print("Config:")
for k, v in asdict(cfg).items():
    print(f"  {k:>16}: {v}")

# %% [markdown]
# ## 3. Locate RFMiD images

# %% Find images
DATA_ROOT = Path(cfg.data_root)
if not DATA_ROOT.exists():
    # Fallback for local runs
    DATA_ROOT = Path("./retinal-disease-classification")
assert DATA_ROOT.exists(), f"Dataset not found at {cfg.data_root}"

EXTS = (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG")
all_images = sorted(p for p in DATA_ROOT.rglob("*") if p.suffix in EXTS)
print(f"Found {len(all_images)} images under {DATA_ROOT}")
assert len(all_images) > 100, "Suspiciously few images — check dataset attach."

# Quick look at one image to confirm we can decode it
with Image.open(all_images[0]) as im:
    print(f"Example: {all_images[0].name}  size={im.size}  mode={im.mode}")

# %% [markdown]
# ## 4. Synthetic noise model
#
# The Poisson-Gaussian model and parameters here come directly from Phase 1
# (`rfmid_noise_analysis.ipynb`). We use three preset noise levels — light,
# medium, heavy — and at training time we sample one of them uniformly per
# image. This is the "blind denoising" regime: the model never gets told which
# level it is facing, so it has to learn to handle the whole range.

# %% Noise levels + injector
NOISE_LEVELS = {
    "light":  {"alpha": 0.5, "sigma": 3.0},   # noise std ≈  8  at μ = 128
    "medium": {"alpha": 1.0, "sigma": 5.0},   # noise std ≈ 12  at μ = 128
    "heavy":  {"alpha": 2.0, "sigma": 8.0},   # noise std ≈ 18  at μ = 128
}

def add_poisson_gaussian_noise(clean_uint8: np.ndarray,
                               alpha: float,
                               sigma: float,
                               rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Mixed Poisson-Gaussian sensor noise model.

      var(y) = alpha * mean(y) + sigma**2

    The Poisson (shot-noise) component is approximated by a signal-dependent
    Gaussian with variance alpha*x — this is the standard linearization used
    by Foi et al. (TIP 2008) and is exact at non-trivial intensities.

    clean_uint8 : (H, W, C) uint8, [0, 255]
    Returns noisy image as uint8.
    """
    if rng is None:
        rng = np.random.default_rng()
    x = clean_uint8.astype(np.float64)
    shot = rng.standard_normal(x.shape) * np.sqrt(np.clip(alpha * x, 0.0, None))
    read = rng.standard_normal(x.shape) * sigma
    y = x + shot + read
    return np.clip(y, 0, 255).astype(np.uint8)

# Quick visual sanity check
_demo = np.array(Image.open(all_images[0]).convert("RGB").resize((384, 384)))
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
axes[0].imshow(_demo); axes[0].set_title("Clean (pseudo-GT)"); axes[0].axis("off")
for i, (name, p) in enumerate(NOISE_LEVELS.items(), start=1):
    n = add_poisson_gaussian_noise(_demo, p["alpha"], p["sigma"],
                                   rng=np.random.default_rng(i))
    axes[i].imshow(n)
    axes[i].set_title(f"{name}  α={p['alpha']}, σ={p['sigma']}")
    axes[i].axis("off")
plt.tight_layout()
plt.savefig(Path(cfg.out_dir) / "noise_levels_preview.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 5. Dataset
#
# `RFMiDDataset` returns `(noisy, clean)` tensor pairs in `[0, 1]`, shape `(3, H, W)`.
# The flow per item:
#   1. Open image, convert to RGB, resize to `image_size` (keeps memory bounded
#      while preserving enough detail at 384×384).
#   2. If training: random 256×256 crop + random dihedral flip/rotation.
#   3. Sample a noise level uniformly, apply Poisson-Gaussian noise.
#   4. Return both tensors. The clean tensor is the regression target.
#
# Why the augmentations are flips/rotations only (no color jitter / blur /
# anything that changes pixel statistics): we are explicitly modelling pixel
# noise. Color jitter would shift the clean distribution, blur would smooth
# noise — both would teach the network a different inverse problem than the
# one it will face at test time. See rationale PDF §7.

# %% Dataset class
class RFMiDDataset(Dataset):
    def __init__(self,
                 paths,
                 image_size: int = 384,
                 patch_size: int = 256,
                 noise_levels: dict = None,
                 fixed_noise: Optional[str] = None,
                 augment: bool = True,
                 deterministic_seed: Optional[int] = None):
        """
        paths              : list of Path to image files
        image_size         : resize images to (image_size, image_size) on load
        patch_size         : crop size at training time. If <= 0, no cropping.
        noise_levels       : dict {name: {alpha, sigma}} — see NOISE_LEVELS
        fixed_noise        : if set (e.g., "medium"), always use that level
                             (used by the per-level test split).
        augment            : enable random crop + dihedral flips/rots
        deterministic_seed : if set, derive a deterministic noise RNG per
                             item (used at validation/test).
        """
        self.paths = list(paths)
        self.image_size = image_size
        self.patch_size = patch_size
        self.noise_levels = noise_levels or NOISE_LEVELS
        self.fixed_noise = fixed_noise
        self.augment = augment
        self.deterministic_seed = deterministic_seed

        self._level_names = list(self.noise_levels.keys())

    def __len__(self):
        return len(self.paths)

    def _load(self, path):
        with Image.open(path) as im:
            im = im.convert("RGB").resize((self.image_size, self.image_size),
                                          Image.BILINEAR)
            return np.array(im, dtype=np.uint8)

    def _sample_noise_params(self, item_seed):
        if self.fixed_noise is not None:
            return self.fixed_noise, self.noise_levels[self.fixed_noise]
        # Use the same seed for picking the level as for the noise itself,
        # so val/test are fully reproducible.
        rng = np.random.default_rng(item_seed)
        name = self._level_names[rng.integers(0, len(self._level_names))]
        return name, self.noise_levels[name]

    def _augment(self, img, rng):
        # Random crop
        if self.patch_size > 0 and img.shape[0] > self.patch_size:
            H, W = img.shape[:2]
            top  = int(rng.integers(0, H - self.patch_size + 1))
            left = int(rng.integers(0, W - self.patch_size + 1))
            img = img[top:top + self.patch_size, left:left + self.patch_size]
        if self.augment:
            # Random dihedral flip / 90° rotation (8 symmetries of a square).
            # These preserve pixel-noise statistics; color jitter would not.
            if rng.random() < 0.5:
                img = np.fliplr(img).copy()
            if rng.random() < 0.5:
                img = np.flipud(img).copy()
            k = int(rng.integers(0, 4))
            if k > 0:
                img = np.rot90(img, k=k).copy()
        return img

    def __getitem__(self, idx):
        path = self.paths[idx]
        # Per-item seed: deterministic for val/test, fresh-random for train
        if self.deterministic_seed is not None:
            item_seed = self.deterministic_seed * 10_000 + idx
        else:
            item_seed = random.randint(0, 2**31 - 1)
        rng = np.random.default_rng(item_seed)

        clean = self._load(path)
        clean = self._augment(clean, rng)

        level_name, params = self._sample_noise_params(item_seed)
        noisy = add_poisson_gaussian_noise(clean, params["alpha"], params["sigma"],
                                           rng=np.random.default_rng(item_seed + 1))

        # to tensors in [0, 1], (C, H, W)
        clean_t = torch.from_numpy(clean).permute(2, 0, 1).float() / 255.0
        noisy_t = torch.from_numpy(noisy).permute(2, 0, 1).float() / 255.0
        return noisy_t, clean_t, level_name

# Sanity check the dataset
_ds = RFMiDDataset(all_images[:8], image_size=cfg.image_size, patch_size=cfg.patch_size)
_n, _c, _lvl = _ds[0]
print(f"Dataset OK. noisy={_n.shape} dtype={_n.dtype} "
      f"range=[{_n.min():.3f}, {_n.max():.3f}]  level={_lvl}")
assert _n.shape == (3, cfg.patch_size, cfg.patch_size)
assert _c.shape == (3, cfg.patch_size, cfg.patch_size)

# %% [markdown]
# ## 6. Splits and DataLoaders
#
# 80 / 10 / 10 train / val / test split, seeded so it's reproducible across
# restarts. The val set uses deterministic noise (same noise level + same RNG
# seed for the same image index every epoch) so the validation curve is a
# clean signal of *model* improvement, not noise-realization variance.

# %% Build splits
rng_split = np.random.default_rng(SEED)
shuffled = list(all_images)
rng_split.shuffle(shuffled)

n_total = len(shuffled)
n_train = int(cfg.train_frac * n_total)
n_val   = int(cfg.val_frac   * n_total)
n_test  = n_total - n_train - n_val

train_paths = shuffled[:n_train]
val_paths   = shuffled[n_train:n_train + n_val]
test_paths  = shuffled[n_train + n_val:]

print(f"Splits: train={len(train_paths)}  val={len(val_paths)}  test={len(test_paths)}")

train_ds = RFMiDDataset(train_paths, cfg.image_size, cfg.patch_size, augment=True)
val_ds   = RFMiDDataset(val_paths,   cfg.image_size, cfg.patch_size,
                        augment=False, deterministic_seed=1)
# For the per-level test, we'll instantiate one dataset per noise level below.

train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                          num_workers=cfg.num_workers, pin_memory=True,
                          drop_last=True, persistent_workers=cfg.num_workers > 0)
val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False,
                          num_workers=cfg.num_workers, pin_memory=True,
                          persistent_workers=cfg.num_workers > 0)

print(f"Batches/epoch: train={len(train_loader)}  val={len(val_loader)}")

# %% [markdown]
# ## 7. U-Net architecture
#
# Self-contained implementation matching the upstream `swastishreya/Hyperbolic-U-Net`
# (MIDL 2026) Euclidean variant, with three deliberate adaptations for denoising
# (each defended in the rationale PDF):
#
# 1. **Output head:** linear `1×1` conv → 3 channels (residual noise), no
#    softmax / sigmoid. The forward pass returns `noisy − noise_pred`.
# 2. **Normalization:** GroupNorm (groups=8) instead of BatchNorm. Denoising
#    cares about per-image statistics; BatchNorm's running stats mix across
#    images, which is harmful when noise level varies per item (EDSR precedent).
# 3. **Upsampling:** `bilinear + 1×1 conv` instead of transposed conv —
#    avoids checkerboard artifacts that are very visible in denoised output.
#
# Depth = 4 means 4 down/up steps. With `base_channels=32`, the width schedule
# is 32 / 64 / 128 / 256 / 512. This is half of "classic" U-Net (64/128/256/
# 512/1024) — denoising doesn't need segmentation-level capacity (the DnCNN
# precedent), and the half-size variant fits the Kaggle T4 (16 GB) at 256²
# patches with batch size 16 and AMP, with ~3 GB headroom for the Phase 3
# hyperbolic variant.

# %% U-Net building blocks
def _norm(c):
    # GroupNorm with at most 8 groups (and at least 1 channel per group)
    g = min(8, c)
    while c % g != 0:
        g -= 1
    return nn.GroupNorm(g, c)

class DoubleConv(nn.Module):
    """ (Conv -> GN -> ReLU) * 2 """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            _norm(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            _norm(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)

class Down(nn.Module):
    """ MaxPool 2x2 then DoubleConv """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x):
        return self.conv(self.pool(x))

class Up(nn.Module):
    """ Bilinear upsample + 1x1 conv (cheaper, no checkerboard) + DoubleConv """
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.reduce = nn.Conv2d(in_ch, in_ch // 2, kernel_size=1)
        self.up     = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv   = DoubleConv(in_ch // 2 + skip_ch, out_ch)
    def forward(self, x, skip):
        x = self.up(self.reduce(x))
        # Pad if odd spatial mismatch (rare for power-of-2 inputs)
        if x.shape[-2:] != skip.shape[-2:]:
            dy = skip.shape[-2] - x.shape[-2]
            dx = skip.shape[-1] - x.shape[-1]
            x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        return self.conv(torch.cat([skip, x], dim=1))

# %% U-Net model
class UNetDenoiser(nn.Module):
    """
    Euclidean U-Net for residual denoising.

    Forward(noisy) -> denoised
        noise_pred = head( UNet_features )
        denoised   = noisy - noise_pred
    """
    def __init__(self, in_ch=3, out_ch=3, base=32, depth=4):
        super().__init__()
        widths = [base * (2 ** i) for i in range(depth + 1)]  # [32, 64, 128, 256, 512]
        self.inc   = DoubleConv(in_ch, widths[0])
        self.downs = nn.ModuleList([Down(widths[i], widths[i + 1]) for i in range(depth)])
        self.ups   = nn.ModuleList([
            Up(in_ch=widths[depth - i], skip_ch=widths[depth - i - 1],
               out_ch=widths[depth - i - 1])
            for i in range(depth)
        ])
        # Linear 1x1 head — predicts residual noise in the same dynamic range as the input
        self.head = nn.Conv2d(widths[0], out_ch, kernel_size=1)
        # Zero-init the head so the model starts as the identity mapping
        # (noise_pred = 0 → denoised = noisy). This is a common stabilizer
        # for residual networks; see e.g. Fixup-init / ReZero.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        skips = []
        h = self.inc(x)
        skips.append(h)
        for down in self.downs:
            h = down(h)
            skips.append(h)
        # h is now the bottleneck; skips[-1] == bottleneck, the rest are encoder feats
        for i, up in enumerate(self.ups):
            skip = skips[-i - 2]  # walk back through encoder features
            h = up(h, skip)
        noise_pred = self.head(h)
        return x - noise_pred  # residual denoising

# Sanity check + parameter count
model = UNetDenoiser(in_ch=3, out_ch=3, base=cfg.base_channels, depth=cfg.depth).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
n_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"UNetDenoiser  total params: {n_params/1e6:.2f}M  trainable: {n_train_params/1e6:.2f}M")

with torch.no_grad():
    _x = torch.randn(1, 3, cfg.patch_size, cfg.patch_size, device=DEVICE)
    _y = model(_x)
print(f"Forward sanity: in={tuple(_x.shape)}  out={tuple(_y.shape)}  "
      f"(zero-init head → max|out-in|={(_y - _x).abs().max().item():.2e})")
del _x, _y

# %% [markdown]
# ## 8. Loss, optimizer, scheduler
#
# - **Loss:** L1 between predicted clean and ground-truth clean.
# - **Optimizer:** AdamW (decoupled weight decay), `lr=2e-4`, `wd=1e-4`.
# - **Schedule:** linear warmup for `warmup_epochs` → cosine annealing to 0
#   over the remaining epochs (Loshchilov & Hutter, ICLR 2017).
# - **AMP:** `torch.cuda.amp` for ~2× speedup on T4.

# %% Optim / sched
criterion = nn.L1Loss()
optimizer = torch.optim.AdamW(model.parameters(),
                              lr=cfg.lr,
                              weight_decay=cfg.weight_decay)

steps_per_epoch = max(1, len(train_loader))
total_steps  = cfg.epochs * steps_per_epoch
warmup_steps = cfg.warmup_epochs * steps_per_epoch

def lr_lambda(step):
    if step < warmup_steps:
        return float(step + 1) / float(max(1, warmup_steps))
    # Cosine annealing from 1 → 0 over the remaining steps
    progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return 0.5 * (1.0 + math.cos(math.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
scaler = GradScaler(enabled=(cfg.amp and DEVICE.type == "cuda"))

print(f"steps_per_epoch={steps_per_epoch}  total_steps={total_steps}  "
      f"warmup_steps={warmup_steps}")

# %% [markdown]
# ## 9. Metric helpers
#
# `torchmetrics` `PeakSignalNoiseRatio` and `StructuralSimilarityIndexMeasure`
# expect inputs in `[0, 1]` with `data_range=1.0`. We accumulate state across
# batches and `.compute()` at the end of each epoch.

# %% Metric helpers
psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)
ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)

def reset_metrics():
    psnr_metric.reset()
    ssim_metric.reset()

# %% [markdown]
# ## 10. Train and validation loops

# %% Training step
def train_one_epoch(epoch):
    model.train()
    running = 0.0
    n_seen = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{cfg.epochs} [train]", leave=False)
    for noisy, clean, _level in pbar:
        noisy = noisy.to(DEVICE, non_blocking=True)
        clean = clean.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=cfg.amp and DEVICE.type == "cuda"):
            denoised = model(noisy)
            # Clamp only for the loss target range comparison? No — we want the
            # model to learn natural [0,1] outputs; clamping at train time would
            # mask any overshoot it produces. We clamp only at eval/visualization.
            loss = criterion(denoised, clean)

        scaler.scale(loss).backward()
        if cfg.grad_clip is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        bs = noisy.size(0)
        running += loss.item() * bs
        n_seen += bs
        pbar.set_postfix(loss=f"{running / n_seen:.4f}",
                         lr=f"{scheduler.get_last_lr()[0]:.2e}")
    return running / max(1, n_seen)

@torch.no_grad()
def validate(epoch):
    model.eval()
    reset_metrics()
    running = 0.0
    n_seen = 0
    pbar = tqdm(val_loader, desc=f"Epoch {epoch:02d}/{cfg.epochs} [val]  ", leave=False)
    for noisy, clean, _level in pbar:
        noisy = noisy.to(DEVICE, non_blocking=True)
        clean = clean.to(DEVICE, non_blocking=True)
        with autocast(enabled=cfg.amp and DEVICE.type == "cuda"):
            denoised = model(noisy).clamp(0.0, 1.0)
            loss = criterion(denoised, clean)
        psnr_metric.update(denoised, clean)
        ssim_metric.update(denoised, clean)
        bs = noisy.size(0)
        running += loss.item() * bs
        n_seen  += bs
    return (running / max(1, n_seen),
            psnr_metric.compute().item(),
            ssim_metric.compute().item())

# %% [markdown]
# ## 11. Visualization helper

# %% Visualization
@torch.no_grad()
def save_qualitative(epoch, n_examples=3):
    model.eval()
    # Take a few deterministic val samples for consistent qualitative tracking
    examples = []
    for idx in range(min(n_examples, len(val_ds))):
        noisy, clean, level = val_ds[idx]
        noisy_b = noisy.unsqueeze(0).to(DEVICE)
        with autocast(enabled=cfg.amp and DEVICE.type == "cuda"):
            denoised = model(noisy_b).clamp(0.0, 1.0).squeeze(0).cpu()
        examples.append((noisy, denoised, clean, level))

    fig, axes = plt.subplots(n_examples, 3, figsize=(12, 4 * n_examples))
    if n_examples == 1:
        axes = axes[None, :]
    for i, (noisy, denoised, clean, level) in enumerate(examples):
        for ax, img, title in zip(
            axes[i],
            [noisy, denoised, clean],
            [f"noisy ({level})", "denoised", "clean"],
        ):
            ax.imshow(img.permute(1, 2, 0).numpy())
            ax.set_title(title)
            ax.axis("off")
    plt.tight_layout()
    out = Path(cfg.out_dir) / f"viz_epoch{epoch:02d}.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    return out

# %% [markdown]
# ## 12. Training driver
#
# Loops over epochs, tracks train/val loss + val PSNR + val SSIM, saves the
# best-PSNR checkpoint, dumps a qualitative panel every `viz_every` epochs.

# %% Train
history = {"epoch": [], "train_loss": [], "val_loss": [],
           "val_psnr": [], "val_ssim": [], "lr": []}
best_psnr = -1.0
best_ckpt_path = Path(cfg.out_dir) / "unet_denoiser_best.pt"
t0 = time.time()

for epoch in range(1, cfg.epochs + 1):
    train_loss = train_one_epoch(epoch)
    if epoch % cfg.eval_every == 0:
        val_loss, val_psnr, val_ssim = validate(epoch)
    else:
        val_loss = val_psnr = val_ssim = float("nan")

    history["epoch"].append(epoch)
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["val_psnr"].append(val_psnr)
    history["val_ssim"].append(val_ssim)
    history["lr"].append(scheduler.get_last_lr()[0])

    elapsed = time.time() - t0
    print(f"epoch {epoch:02d}/{cfg.epochs}  "
          f"train_L1={train_loss:.4f}  val_L1={val_loss:.4f}  "
          f"PSNR={val_psnr:.2f}  SSIM={val_ssim:.4f}  "
          f"lr={scheduler.get_last_lr()[0]:.2e}  "
          f"({elapsed/60:.1f} min total)")

    if val_psnr > best_psnr:
        best_psnr = val_psnr
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "config": asdict(cfg),
            "val_psnr": val_psnr,
            "val_ssim": val_ssim,
        }, best_ckpt_path)
        print(f"  ↳ new best PSNR={val_psnr:.2f}  saved to {best_ckpt_path.name}")

    if epoch % cfg.viz_every == 0 or epoch == cfg.epochs:
        viz = save_qualitative(epoch)
        print(f"  ↳ qualitative dump: {viz.name}")

print(f"\nTraining done in {(time.time() - t0)/60:.1f} min. Best val PSNR = {best_psnr:.2f} dB.")

# %% [markdown]
# ## 13. Training curves

# %% Plot curves
fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(history["epoch"], history["train_loss"], label="train L1", color="C0")
ax1.plot(history["epoch"], history["val_loss"],   label="val L1",   color="C1")
ax1.set_xlabel("epoch"); ax1.set_ylabel("L1 loss"); ax1.legend(loc="upper left")
ax1.grid(alpha=0.3)
ax2 = ax1.twinx()
ax2.plot(history["epoch"], history["val_psnr"], label="val PSNR", color="C2",
         linestyle="--", marker="o", markersize=3)
ax2.set_ylabel("PSNR (dB)")
ax2.legend(loc="upper right")
plt.title("Training curves")
plt.tight_layout()
curves_path = Path(cfg.out_dir) / "training_curves.png"
plt.savefig(curves_path, bbox_inches="tight")
plt.show()
print(f"Saved {curves_path}")

# %% [markdown]
# ## 14. Final evaluation — PSNR/SSIM per noise level
#
# Load the best checkpoint and run it on the held-out test split *three times*,
# once per noise level, with deterministic noise. The per-level breakdown is
# what the supervisor (and Phase 3 comparison) will care about most: where does
# the model struggle? The hyperbolic variant should help most on the heavy
# noise level if it helps at all.

# %% Per-level eval
ckpt = torch.load(best_ckpt_path, map_location=DEVICE)
model.load_state_dict(ckpt["model_state"])
model.eval()
print(f"Loaded best checkpoint from epoch {ckpt['epoch']}  (val PSNR {ckpt['val_psnr']:.2f}).")

@torch.no_grad()
def evaluate_level(level_name):
    ds = RFMiDDataset(test_paths, cfg.image_size, cfg.patch_size,
                      augment=False, deterministic_seed=7,
                      fixed_noise=level_name)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=cfg.num_workers, pin_memory=True)
    reset_metrics()
    # Also compute the "noisy baseline" PSNR/SSIM (what you'd get without
    # denoising) so we can report an improvement, not just an absolute number.
    psnr_in = PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)
    ssim_in = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)
    n_total = 0
    for noisy, clean, _ in tqdm(loader, desc=f"test [{level_name}]", leave=False):
        noisy = noisy.to(DEVICE, non_blocking=True)
        clean = clean.to(DEVICE, non_blocking=True)
        with autocast(enabled=cfg.amp and DEVICE.type == "cuda"):
            denoised = model(noisy).clamp(0.0, 1.0)
        psnr_metric.update(denoised, clean)
        ssim_metric.update(denoised, clean)
        psnr_in.update(noisy, clean)
        ssim_in.update(noisy, clean)
        n_total += noisy.size(0)
    return {
        "n": n_total,
        "psnr_in":  psnr_in.compute().item(),
        "ssim_in":  ssim_in.compute().item(),
        "psnr_out": psnr_metric.compute().item(),
        "ssim_out": ssim_metric.compute().item(),
    }

per_level = {name: evaluate_level(name) for name in NOISE_LEVELS}

print()
print(f"{'level':>8}  {'n':>5}  "
      f"{'PSNR_in':>8}  {'PSNR_out':>8}  {'ΔPSNR':>6}  "
      f"{'SSIM_in':>8}  {'SSIM_out':>8}  {'ΔSSIM':>6}")
for name, r in per_level.items():
    print(f"{name:>8}  {r['n']:>5}  "
          f"{r['psnr_in']:>8.2f}  {r['psnr_out']:>8.2f}  "
          f"{r['psnr_out']-r['psnr_in']:>6.2f}  "
          f"{r['ssim_in']:>8.4f}  {r['ssim_out']:>8.4f}  "
          f"{r['ssim_out']-r['ssim_in']:>6.4f}")

# %% [markdown]
# ## 15. Final qualitative panel (test split)
#
# One representative image per noise level — clean / noisy / denoised side by
# side. This is the figure that goes into the slide deck and manuscript.

# %% Final qualitative
@torch.no_grad()
def final_qualitative():
    fig, axes = plt.subplots(len(NOISE_LEVELS), 3, figsize=(12, 4 * len(NOISE_LEVELS)))
    for i, level in enumerate(NOISE_LEVELS):
        ds = RFMiDDataset([test_paths[0]], cfg.image_size, cfg.patch_size,
                          augment=False, deterministic_seed=99,
                          fixed_noise=level)
        noisy, clean, _ = ds[0]
        with autocast(enabled=cfg.amp and DEVICE.type == "cuda"):
            denoised = model(noisy.unsqueeze(0).to(DEVICE)).clamp(0.0, 1.0).squeeze(0).cpu()
        for ax, img, title in zip(
            axes[i],
            [noisy, denoised, clean],
            [f"noisy ({level})", "denoised", "clean"],
        ):
            ax.imshow(img.permute(1, 2, 0).numpy())
            ax.set_title(title)
            ax.axis("off")
    plt.tight_layout()
    out = Path(cfg.out_dir) / "final_qualitative.png"
    plt.savefig(out, bbox_inches="tight")
    plt.show()
    return out

final_q = final_qualitative()
print(f"Saved {final_q}")

# %% [markdown]
# ## 16. Persist results
#
# Everything goes into `/kaggle/working` so Kaggle's "Output" tab exposes it as
# the notebook's downloadable artifacts:
# - `unet_denoiser_best.pt`  — best checkpoint
# - `training_curves.png`    — loss + PSNR curves
# - `viz_epoch*.png`         — qualitative panels per N epochs
# - `final_qualitative.png`  — one row per noise level
# - `noise_levels_preview.png`
# - `results.json`           — summary: config + per-level PSNR/SSIM
# - `history.json`           — per-epoch metrics

# %% Save results
results = {
    "config":         asdict(cfg),
    "best_epoch":     int(ckpt["epoch"]),
    "best_val_psnr":  float(ckpt["val_psnr"]),
    "best_val_ssim":  float(ckpt["val_ssim"]),
    "test_per_level": per_level,
    "noise_levels":   NOISE_LEVELS,
    "n_params":       int(n_params),
    "device":         str(DEVICE),
}
with open(Path(cfg.out_dir) / "results.json", "w") as f:
    json.dump(results, f, indent=2)
with open(Path(cfg.out_dir) / "history.json", "w") as f:
    json.dump(history, f, indent=2)

print("\nArtifacts saved to /kaggle/working:")
for p in sorted(Path(cfg.out_dir).iterdir()):
    if p.is_file():
        print(f"  {p.name:<35}  {p.stat().st_size/1024:>8.1f} KB")

# %% [markdown]
# ## 17. Phase 3 — handoff notes
#
# To produce the hyperbolic variant for the apples-to-apples comparison:
#
# 1. `pip install hypll==0.1.1` at the top of a fresh notebook.
# 2. Keep this notebook's data pipeline, loss, optimizer, schedule, AMP,
#    metrics, and eval harness exactly as-is — those must not vary between
#    the two runs.
# 3. Replace `UNetDenoiser` with a hyperbolic version where:
#       - encoder/decoder convs become `hypll.nn.HConv2d`,
#       - skip concatenations happen on the Poincaré ball,
#       - the output head logs from the ball back to Euclidean space and
#         predicts the residual noise there (the input/output of the model
#         remains a Euclidean RGB image — only the internal representation
#         is hyperbolic).
# 4. Switch the optimizer to `geoopt.optim.RiemannianAdam` so the hyperbolic
#    parameters follow the manifold's geometry.
# 5. Train with the *identical* recipe and compare `results.json` side by side.
#
# All design rationale for these choices is in `phase2_design_rationale.pdf`
# §12 ("Path to Phase 3"). Until that doc is on hand, keep Phase 2 unchanged.
