# %% [markdown]
# # RFMiD Denoising U-Net — Phase 3 (Bottleneck-only Hyperbolic)
#
# **Goal.** Re-run Phase 2 with a single architectural change: the U-Net's
# bottleneck features are passed through a **hyperbolic block on the Poincaré
# ball** before continuing into the decoder. Everything else — data pipeline,
# loss, optimizer schedule, augmentations, splits, evaluation harness — is
# **identical to Phase 2**, so any difference in final metrics is attributable
# to the geometry of the bottleneck representation alone.
#
# **What "bottleneck-only" means.** Our Phase 2 U-Net has 4 down-blocks and 4
# up-blocks. After the last down-block, features have shape `[B, 512, 16, 16]`
# — this is the *bottleneck*, the most semantic, lowest-resolution
# representation. In Phase 3 we insert a `HyperbolicBottleneck` module that:
#   1. Lifts those Euclidean features onto the Poincaré ball via the exponential
#      map at the origin (`expmap`).
#   2. Applies two hyperbolic convolutions with a hyperbolic ReLU in between.
#   3. Maps the result back to the tangent space at the origin (`logmap`),
#      returning Euclidean features in the same shape.
# Encoder and decoder remain fully Euclidean.
#
# **Why bottleneck-only.** Three reasons. (a) It's the cheapest meaningful test
# of the hyperbolic hypothesis — if hyperbolic geometry has any benefit, the
# most semantic layer is where it should show up. (b) It isolates the
# numerical-stability work to one place rather than scattering it through 9
# layers. (c) It makes the apples-to-apples comparison with Phase 2 maximally
# clean: only one module differs.
#
# **Why we expect a benefit (and why it might be modest).** Retinal vessels are
# a hierarchical tree structure — bifurcations splitting into smaller
# bifurcations recursively. Hyperbolic space embeds trees with arbitrarily low
# distortion, so a hyperbolic latent representation may preserve vascular
# topology better than a Euclidean one of the same dimensionality. But the
# bottleneck is only 16×16 — most of the fine vessel detail is already encoded
# in earlier (Euclidean) feature maps via skip connections, which Phase 3 does
# NOT touch. Realistic expectation: 0.3–1 dB PSNR improvement at heavy noise,
# possibly visible vessel sharpening in qualitative panels. If we get more,
# that's a finding; if we get equal or worse, that's also a finding.
#
# **What changed vs Phase 2 (compact list):**
#   - `pip install hypll geoopt` added to the setup cell.
#   - `HyperbolicBottleneck` module + `UNetDenoiser` updated to call it.
#   - Optimizer switched from `torch.optim.AdamW` to `hypll.optim.RiemannianAdam`
#     (handles Euclidean params identically to AdamW, plus the manifold params).
#   - **AMP disabled.** fp16 lacks the precision needed for stable operations
#     near the Poincaré boundary. We train in fp32; training time roughly
#     doubles (~3-4 h on T4 vs Phase 2's 1.8 h).
#   - Output filenames suffixed with `_hyp` to keep Phase 2 artifacts intact.
#
# **How to run on Kaggle.** Same as Phase 2 (GPU T4, Internet ON, attach
# `andrewmvd/retinal-disease-classification`), but plan for the longer runtime.

# %% [markdown]
# ## 0. Install extra packages

# %% Install
import subprocess, sys
def _pip(*pkgs):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])

# torchmetrics: PSNR + SSIM
try:
    import torchmetrics  # noqa: F401
except ImportError:
    _pip("torchmetrics==1.4.0")

# hypll: hyperbolic layers, manifolds, RiemannianAdam
try:
    import hypll  # noqa: F401
except ImportError:
    _pip("hypll==0.1.1")

print("Packages OK.")

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
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

# Hyperbolic imports
from hypll.manifolds.poincare_ball import PoincareBall, Curvature
from hypll import nn as hnn
from hypll.tensors import TangentTensor
from hypll.optim import RiemannianAdam

SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True
plt.rcParams["figure.dpi"] = 100

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch {torch.__version__}  |  device = {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}  |  "
          f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# %% [markdown]
# ## 2. Configuration
#
# Same hyperparameters as Phase 2 except: `amp=False` (fp16 is unsafe near the
# Poincaré boundary), and output paths suffixed `_hyp`. The curvature `c=0.1`
# matches the Mishra et al. (MIDL 2026) default and is **not** trainable in
# this first run — fewer moving parts. A future ablation could set
# `trainable_curvature=True`.

# %% Config
@dataclass
class Config:
    # Data
    data_root: str = "/kaggle/input/retinal-disease-classification"
    out_dir:   str = "/kaggle/working"
    image_size: int = 384
    patch_size: int = 256
    train_frac: float = 0.80
    val_frac:   float = 0.10
    num_workers: int = 2

    # Model
    base_channels: int = 32
    depth: int = 4

    # Hyperbolic-specific
    curvature: float = 0.1
    trainable_curvature: bool = False
    hyperbolic_kernel: int = 3       # HConvolution2d kernel size at the bottleneck

    # Optimization
    batch_size: int = 16
    epochs: int = 30
    lr: float = 2e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 1
    grad_clip: float = 1.0
    amp: bool = False                # MUST be False for hyperbolic stability

    # Eval / I/O
    eval_every: int = 1
    viz_every: int = 5
    suffix: str = "_hyp"             # appended to all output filenames

cfg = Config()
Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
print("Config:")
for k, v in asdict(cfg).items():
    print(f"  {k:>20}: {v}")

# %% [markdown]
# ## 3. Locate RFMiD images
# (Identical to Phase 2.)

# %% Find images
DATA_ROOT = Path(cfg.data_root)
if not DATA_ROOT.exists():
    DATA_ROOT = Path("./retinal-disease-classification")
assert DATA_ROOT.exists(), f"Dataset not found at {cfg.data_root}"

EXTS = (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG")
all_images = sorted(p for p in DATA_ROOT.rglob("*") if p.suffix in EXTS)
print(f"Found {len(all_images)} images under {DATA_ROOT}")
assert len(all_images) > 100

# %% [markdown]
# ## 4. Noise model + Dataset + splits
# (All identical to Phase 2 — same NOISE_LEVELS, same `add_poisson_gaussian_noise`,
# same `RFMiDDataset`, same 80/10/10 split, same loaders. Identical pipeline
# is the whole point: only the bottleneck geometry varies between Phase 2 and 3.)

# %% Noise
NOISE_LEVELS = {
    "light":  {"alpha": 0.5, "sigma": 3.0},
    "medium": {"alpha": 1.0, "sigma": 5.0},
    "heavy":  {"alpha": 2.0, "sigma": 8.0},
}

def add_poisson_gaussian_noise(clean_uint8, alpha, sigma, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    x = clean_uint8.astype(np.float64)
    shot = rng.standard_normal(x.shape) * np.sqrt(np.clip(alpha * x, 0.0, None))
    read = rng.standard_normal(x.shape) * sigma
    return np.clip(x + shot + read, 0, 255).astype(np.uint8)

# %% Dataset
class RFMiDDataset(Dataset):
    def __init__(self, paths, image_size=384, patch_size=256,
                 noise_levels=None, fixed_noise=None,
                 augment=True, deterministic_seed=None):
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
        rng = np.random.default_rng(item_seed)
        name = self._level_names[rng.integers(0, len(self._level_names))]
        return name, self.noise_levels[name]

    def _augment(self, img, rng):
        if self.patch_size > 0 and img.shape[0] > self.patch_size:
            H, W = img.shape[:2]
            top  = int(rng.integers(0, H - self.patch_size + 1))
            left = int(rng.integers(0, W - self.patch_size + 1))
            img = img[top:top + self.patch_size, left:left + self.patch_size]
        if self.augment:
            if rng.random() < 0.5: img = np.fliplr(img).copy()
            if rng.random() < 0.5: img = np.flipud(img).copy()
            k = int(rng.integers(0, 4))
            if k > 0: img = np.rot90(img, k=k).copy()
        return img

    def __getitem__(self, idx):
        path = self.paths[idx]
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
        clean_t = torch.from_numpy(clean).permute(2, 0, 1).float() / 255.0
        noisy_t = torch.from_numpy(noisy).permute(2, 0, 1).float() / 255.0
        return noisy_t, clean_t, level_name

# %% Splits + loaders
rng_split = np.random.default_rng(SEED)
shuffled = list(all_images); rng_split.shuffle(shuffled)
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
train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                          num_workers=cfg.num_workers, pin_memory=True,
                          drop_last=True, persistent_workers=cfg.num_workers > 0)
val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False,
                          num_workers=cfg.num_workers, pin_memory=True,
                          persistent_workers=cfg.num_workers > 0)
print(f"Batches/epoch: train={len(train_loader)}  val={len(val_loader)}")

# %% [markdown]
# ## 5. Euclidean U-Net blocks
# (Identical to Phase 2 — `DoubleConv`, `Down`, `Up`. We only swap what happens
# *between* the deepest `Down` and the first `Up`.)

# %% Euclidean blocks
def _norm(c):
    g = min(8, c)
    while c % g != 0:
        g -= 1
    return nn.GroupNorm(g, c)

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            _norm(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            _norm(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2); self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x): return self.conv(self.pool(x))

class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.reduce = nn.Conv2d(in_ch, in_ch // 2, 1)
        self.up     = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv   = DoubleConv(in_ch // 2 + skip_ch, out_ch)
    def forward(self, x, skip):
        x = self.up(self.reduce(x))
        if x.shape[-2:] != skip.shape[-2:]:
            dy = skip.shape[-2] - x.shape[-2]; dx = skip.shape[-1] - x.shape[-1]
            x = F.pad(x, [dx//2, dx-dx//2, dy//2, dy-dy//2])
        return self.conv(torch.cat([skip, x], dim=1))

# %% [markdown]
# ## 6. The hyperbolic bottleneck — the only Phase 3 architectural change
#
# Three operations, in order:
#
# 1. **Exponential map at the origin.** Treats the encoder's Euclidean
#    bottleneck features as tangent vectors at the origin of the Poincaré ball,
#    and maps them onto the ball. The channel dimension becomes the manifold
#    dimension (`man_dim=1`), so each spatial position is one 512-dimensional
#    point on the ball.
# 2. **Two hyperbolic convolutions with a hyperbolic ReLU between them.** Same
#    receptive-field structure as a Euclidean ResBlock would have, but every
#    operation respects the manifold's geometry (`HConvolution2d` uses Möbius
#    operations internally; `HReLU` is a manifold-aware nonlinearity).
# 3. **Logarithmic map back to the tangent space at the origin.** Returns
#    Euclidean features in the same shape as the input, ready to feed back into
#    the Euclidean decoder. Using `logmap` (rather than just `.tensor`) ensures
#    we return unbounded tangent vectors instead of points inside the unit ball
#    — the decoder is initialized for unbounded inputs.
#
# Curvature `c=0.1` follows the Mishra et al. paper. Smaller `c` ⇒ flatter ball
# ⇒ less aggressive hyperbolic behaviour ⇒ better numerical stability. We keep
# it fixed in this run; a future ablation could make it trainable.

# %% HyperbolicBottleneck module
class HyperbolicBottleneck(nn.Module):
    """
    Wraps a small hyperbolic block. Forward signature is plain Euclidean
    in / Euclidean out so the surrounding U-Net code doesn't have to know
    anything about manifolds.
    """
    def __init__(self, channels: int, kernel_size: int = 3,
                 curvature: float = 0.1, trainable_curvature: bool = False):
        super().__init__()
        self.manifold = PoincareBall(
            c=Curvature(value=curvature, requires_grad=trainable_curvature)
        )
        # Two hyperbolic convolutions with a hyperbolic activation in between.
        # We use the same kernel size and a single intermediate width; this is
        # the minimum-viable "hyperbolic block" that still has receptive-field
        # behaviour beyond a per-pixel transformation.
        self.hconv1 = hnn.HConvolution2d(
            in_channels=channels, out_channels=channels,
            kernel_size=kernel_size, manifold=self.manifold,
        )
        self.hrelu  = hnn.HReLU(manifold=self.manifold)
        self.hconv2 = hnn.HConvolution2d(
            in_channels=channels, out_channels=channels,
            kernel_size=kernel_size, manifold=self.manifold,
        )

    def forward(self, x_euc):
        # x_euc: (B, C, H, W) Euclidean
        tangent = TangentTensor(data=x_euc, man_dim=1, manifold=self.manifold)
        h = self.manifold.expmap(tangent)        # ManifoldTensor on the ball
        h = self.hconv1(h)
        h = self.hrelu(h)
        h = self.hconv2(h)
        h_tan = self.manifold.logmap(None, h)    # back to tangent space at origin
        return h_tan.tensor                      # (B, C, H, W) Euclidean

# %% [markdown]
# ## 7. UNetDenoiser — encoder/decoder unchanged, bottleneck wrapped

# %% Modified UNet
class UNetDenoiser(nn.Module):
    """
    Same as the Phase 2 U-Net, but with a HyperbolicBottleneck inserted
    between the deepest Down and the first Up. Forward returns
    `noisy - noise_pred` (residual learning), same as Phase 2.
    """
    def __init__(self, in_ch=3, out_ch=3, base=32, depth=4,
                 hyp_kernel=3, curvature=0.1, trainable_curvature=False):
        super().__init__()
        widths = [base * (2 ** i) for i in range(depth + 1)]
        self.inc   = DoubleConv(in_ch, widths[0])
        self.downs = nn.ModuleList([Down(widths[i], widths[i + 1]) for i in range(depth)])
        # === The single Phase 3 architectural change ===
        self.hyperbolic = HyperbolicBottleneck(
            channels=widths[depth],
            kernel_size=hyp_kernel,
            curvature=curvature,
            trainable_curvature=trainable_curvature,
        )
        # ===
        self.ups   = nn.ModuleList([
            Up(in_ch=widths[depth - i], skip_ch=widths[depth - i - 1],
               out_ch=widths[depth - i - 1])
            for i in range(depth)
        ])
        self.head = nn.Conv2d(widths[0], out_ch, 1)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

    def forward(self, x):
        skips = []
        h = self.inc(x); skips.append(h)
        for down in self.downs:
            h = down(h); skips.append(h)
        # === Hyperbolic round-trip at the bottleneck ===
        h = self.hyperbolic(h)
        # ===
        for i, up in enumerate(self.ups):
            skip = skips[-i - 2]
            h = up(h, skip)
        return x - self.head(h)

# Build, count params, sanity-check forward
model = UNetDenoiser(
    in_ch=3, out_ch=3, base=cfg.base_channels, depth=cfg.depth,
    hyp_kernel=cfg.hyperbolic_kernel,
    curvature=cfg.curvature,
    trainable_curvature=cfg.trainable_curvature,
).to(DEVICE)

def _numel(p):
    # hypll's ManifoldParameter requires going through `.tensor` for torch ops
    return p.tensor.numel() if hasattr(p, 'tensor') else p.numel()

n_params       = sum(_numel(p) for p in model.parameters())
n_train_params = sum(_numel(p) for p in model.parameters() if p.requires_grad)
n_hyp_params   = sum(_numel(p) for p in model.hyperbolic.parameters())
print(f"UNetDenoiser  total params: {n_params/1e6:.2f}M  "
      f"trainable: {n_train_params/1e6:.2f}M  "
      f"hyperbolic-block: {n_hyp_params/1e6:.2f}M")

with torch.no_grad():
    _x = torch.randn(1, 3, cfg.patch_size, cfg.patch_size, device=DEVICE)
    _y = model(_x)
print(f"Forward sanity: in={tuple(_x.shape)}  out={tuple(_y.shape)}  "
      f"finite={torch.isfinite(_y).all().item()}")
del _x, _y

# %% [markdown]
# ## 8. Loss + RiemannianAdam + cosine schedule
#
# The loss and schedule are identical to Phase 2. The optimizer is switched to
# `hypll.optim.RiemannianAdam`, which:
#   - Treats Euclidean parameters (encoder, decoder, head) exactly like AdamW.
#   - Treats hyperbolic parameters (hconv1, hconv2's manifold weights) using
#     Riemannian gradient updates that respect the manifold's metric.
# We pass `weight_decay=1e-4` so the Euclidean-side regularization matches
# Phase 2; RiemannianAdam applies it to the Euclidean params only.

# %% Loss / optim / sched
criterion = nn.L1Loss()
optimizer = RiemannianAdam(model.parameters(),
                           lr=cfg.lr,
                           weight_decay=cfg.weight_decay)

steps_per_epoch = max(1, len(train_loader))
total_steps  = cfg.epochs * steps_per_epoch
warmup_steps = cfg.warmup_epochs * steps_per_epoch

def lr_lambda(step):
    if step < warmup_steps:
        return float(step + 1) / float(max(1, warmup_steps))
    progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return 0.5 * (1.0 + math.cos(math.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
print(f"steps_per_epoch={steps_per_epoch}  total_steps={total_steps}  "
      f"warmup_steps={warmup_steps}  amp=False (hyperbolic stability)")

# %% Metrics
psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)
ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)

def reset_metrics():
    psnr_metric.reset(); ssim_metric.reset()

# %% [markdown]
# ## 9. Train + validation loops
# (Identical structure to Phase 2 — no AMP wrappers since we disabled it.)

# %% Train / val
def train_one_epoch(epoch):
    model.train()
    running = 0.0; n_seen = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{cfg.epochs} [train]", leave=False)
    for noisy, clean, _level in pbar:
        noisy = noisy.to(DEVICE, non_blocking=True)
        clean = clean.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        denoised = model(noisy)
        loss = criterion(denoised, clean)
        loss.backward()
        if cfg.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        scheduler.step()
        bs = noisy.size(0)
        running += loss.item() * bs; n_seen += bs
        pbar.set_postfix(loss=f"{running / n_seen:.4f}",
                         lr=f"{scheduler.get_last_lr()[0]:.2e}")
    return running / max(1, n_seen)

@torch.no_grad()
def validate(epoch):
    model.eval()
    reset_metrics()
    running = 0.0; n_seen = 0
    pbar = tqdm(val_loader, desc=f"Epoch {epoch:02d}/{cfg.epochs} [val]  ", leave=False)
    for noisy, clean, _level in pbar:
        noisy = noisy.to(DEVICE, non_blocking=True)
        clean = clean.to(DEVICE, non_blocking=True)
        denoised = model(noisy).clamp(0.0, 1.0)
        loss = criterion(denoised, clean)
        psnr_metric.update(denoised, clean)
        ssim_metric.update(denoised, clean)
        bs = noisy.size(0)
        running += loss.item() * bs; n_seen += bs
    return (running / max(1, n_seen),
            psnr_metric.compute().item(),
            ssim_metric.compute().item())

# %% [markdown]
# ## 10. Visualization helper
# (Same panel layout as Phase 2 so the two sets of viz dumps can be compared.)

# %% Viz
@torch.no_grad()
def save_qualitative(epoch, n_examples=3):
    model.eval()
    examples = []
    for idx in range(min(n_examples, len(val_ds))):
        noisy, clean, level = val_ds[idx]
        noisy_b = noisy.unsqueeze(0).to(DEVICE)
        denoised = model(noisy_b).clamp(0.0, 1.0).squeeze(0).cpu()
        examples.append((noisy, denoised, clean, level))

    fig, axes = plt.subplots(n_examples, 3, figsize=(12, 4 * n_examples))
    if n_examples == 1:
        axes = axes[None, :]
    for i, (noisy, denoised, clean, level) in enumerate(examples):
        for ax, img, title in zip(
            axes[i],
            [noisy, denoised, clean],
            [f"noisy ({level})", "denoised (hyp)", "clean"],
        ):
            ax.imshow(img.permute(1, 2, 0).numpy())
            ax.set_title(title); ax.axis("off")
    plt.tight_layout()
    out = Path(cfg.out_dir) / f"viz_epoch{epoch:02d}{cfg.suffix}.png"
    plt.savefig(out, bbox_inches="tight"); plt.close()
    return out

# %% [markdown]
# ## 11. Training driver

# %% Train
history = {"epoch": [], "train_loss": [], "val_loss": [],
           "val_psnr": [], "val_ssim": [], "lr": []}
best_psnr = -1.0
best_ckpt_path = Path(cfg.out_dir) / f"unet_denoiser_best{cfg.suffix}.pt"
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
# ## 12. Training curves

# %% Curves
fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(history["epoch"], history["train_loss"], label="train L1", color="C0")
ax1.plot(history["epoch"], history["val_loss"],   label="val L1",   color="C1")
ax1.set_xlabel("epoch"); ax1.set_ylabel("L1 loss")
ax1.legend(loc="upper left"); ax1.grid(alpha=0.3)
ax2 = ax1.twinx()
ax2.plot(history["epoch"], history["val_psnr"], label="val PSNR", color="C2",
         linestyle="--", marker="o", markersize=3)
ax2.set_ylabel("PSNR (dB)"); ax2.legend(loc="upper right")
plt.title(f"Training curves — Phase 3 (hyperbolic bottleneck, c={cfg.curvature})")
plt.tight_layout()
curves_path = Path(cfg.out_dir) / f"training_curves{cfg.suffix}.png"
plt.savefig(curves_path, bbox_inches="tight"); plt.show()
print(f"Saved {curves_path}")

# %% [markdown]
# ## 13. Per-noise-level test eval
# Same harness as Phase 2 — the numbers are directly comparable.

# %% Per-level
ckpt = torch.load(best_ckpt_path, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt["model_state"])
model.eval()
print(f"Loaded best Phase 3 checkpoint from epoch {ckpt['epoch']}  "
      f"(val PSNR {ckpt['val_psnr']:.2f}).")

@torch.no_grad()
def evaluate_level(level_name):
    ds = RFMiDDataset(test_paths, cfg.image_size, cfg.patch_size,
                      augment=False, deterministic_seed=7,
                      fixed_noise=level_name)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=cfg.num_workers, pin_memory=True)
    reset_metrics()
    psnr_in = PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)
    ssim_in = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)
    n_total = 0
    for noisy, clean, _ in tqdm(loader, desc=f"test [{level_name}]", leave=False):
        noisy = noisy.to(DEVICE, non_blocking=True)
        clean = clean.to(DEVICE, non_blocking=True)
        denoised = model(noisy).clamp(0.0, 1.0)
        psnr_metric.update(denoised, clean); ssim_metric.update(denoised, clean)
        psnr_in.update(noisy, clean);        ssim_in.update(noisy, clean)
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
# ## 14. Head-to-head with Phase 2
#
# Hardcoded Phase 2 numbers (from our previous Kaggle run, val PSNR 39.74 dB,
# best epoch 29) so the comparison renders even when Phase 2's `results.json`
# isn't attached to this notebook. The deltas in the rightmost columns are
# the answer to "did making the bottleneck hyperbolic help?"

# %% Comparison table
PHASE2_PER_LEVEL = {
    "light":  {"psnr_out": 41.29, "ssim_out": 0.9578},
    "medium": {"psnr_out": 39.81, "ssim_out": 0.9437},
    "heavy":  {"psnr_out": 38.39, "ssim_out": 0.9288},
}
PHASE2_BEST_VAL_PSNR = 39.74

print(f"{'level':>8}  {'P2_PSNR':>8}  {'P3_PSNR':>8}  {'ΔPSNR':>7}  "
      f"{'P2_SSIM':>8}  {'P3_SSIM':>8}  {'ΔSSIM':>8}")
for name in NOISE_LEVELS:
    p2 = PHASE2_PER_LEVEL[name]; p3 = per_level[name]
    dpsnr = p3["psnr_out"] - p2["psnr_out"]
    dssim = p3["ssim_out"] - p2["ssim_out"]
    print(f"{name:>8}  "
          f"{p2['psnr_out']:>8.2f}  {p3['psnr_out']:>8.2f}  {dpsnr:>+7.2f}  "
          f"{p2['ssim_out']:>8.4f}  {p3['ssim_out']:>8.4f}  {dssim:>+8.4f}")
print(f"\nBest val PSNR: Phase 2 = {PHASE2_BEST_VAL_PSNR:.2f} dB  |  "
      f"Phase 3 = {ckpt['val_psnr']:.2f} dB  "
      f"({ckpt['val_psnr'] - PHASE2_BEST_VAL_PSNR:+.2f})")

# %% [markdown]
# ## 15. Final qualitative panel + save results

# %% Final qual
@torch.no_grad()
def final_qualitative():
    fig, axes = plt.subplots(len(NOISE_LEVELS), 3, figsize=(12, 4 * len(NOISE_LEVELS)))
    for i, level in enumerate(NOISE_LEVELS):
        ds = RFMiDDataset([test_paths[0]], cfg.image_size, cfg.patch_size,
                          augment=False, deterministic_seed=99,
                          fixed_noise=level)
        noisy, clean, _ = ds[0]
        denoised = model(noisy.unsqueeze(0).to(DEVICE)).clamp(0.0, 1.0).squeeze(0).cpu()
        for ax, img, title in zip(
            axes[i],
            [noisy, denoised, clean],
            [f"noisy ({level})", "denoised (hyp)", "clean"],
        ):
            ax.imshow(img.permute(1, 2, 0).numpy())
            ax.set_title(title); ax.axis("off")
    plt.tight_layout()
    out = Path(cfg.out_dir) / f"final_qualitative{cfg.suffix}.png"
    plt.savefig(out, bbox_inches="tight"); plt.show()
    return out

final_q = final_qualitative()
print(f"Saved {final_q}")

# %% Persist
results = {
    "phase":          "phase3_bottleneck_hyperbolic",
    "config":         asdict(cfg),
    "best_epoch":     int(ckpt["epoch"]),
    "best_val_psnr":  float(ckpt["val_psnr"]),
    "best_val_ssim":  float(ckpt["val_ssim"]),
    "test_per_level": per_level,
    "noise_levels":   NOISE_LEVELS,
    "n_params":       int(n_params),
    "n_hyperbolic_params": int(n_hyp_params),
    "device":         str(DEVICE),
    "comparison_phase2": {
        "best_val_psnr": PHASE2_BEST_VAL_PSNR,
        "per_level": PHASE2_PER_LEVEL,
    },
}
with open(Path(cfg.out_dir) / f"results{cfg.suffix}.json", "w") as f:
    json.dump(results, f, indent=2)
with open(Path(cfg.out_dir) / f"history{cfg.suffix}.json", "w") as f:
    json.dump(history, f, indent=2)

print("\nArtifacts saved to /kaggle/working:")
for p in sorted(Path(cfg.out_dir).iterdir()):
    if p.is_file() and cfg.suffix in p.name:
        print(f"  {p.name:<40}  {p.stat().st_size/1024:>8.1f} KB")

# %% [markdown]
# ## 16. Notes for the writeup
#
# **If Phase 3 ≥ Phase 2 by > 0.5 dB at heavy noise:** the bottleneck-only
# hyperbolic intervention helped. This is a publishable positive finding —
# it shows hyperbolic geometry transfers from segmentation robustness
# (Mishra et al.) to restoration tasks.
#
# **If Phase 3 ≈ Phase 2 (within ±0.3 dB):** the bottleneck alone is not
# enough; the natural next step is a full hyperbolic U-Net (matching Mishra
# et al. exactly) before concluding hyperbolic doesn't help for denoising.
#
# **If Phase 3 < Phase 2:** likely the bottleneck-only intervention created
# an information bottleneck (literally). Two possible follow-ups: (a) add a
# residual skip *around* the hyperbolic block so it can degrade to identity,
# (b) re-run with the trainable curvature so the network can flatten the
# manifold if it wants to.
#
# Either way, the head-to-head table in §14 is what goes into the manuscript.
