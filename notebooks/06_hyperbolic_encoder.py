# %% [markdown]
# # RFMiD Denoising U-Net — Phase 4 (Hyperbolic Encoder, Euclidean Decoder)
#
# **The motivating finding:** Phase 3 and Phase 3.1 showed that partial Poincaré
# interventions — one or two hyperbolic layers at or near the bottleneck — were
# consistently slightly worse than the Euclidean baseline, even with the
# residual skip and trainable curvature. The learned curvature drifted from
# `c=0.1` to `c≈0.69`, meaning the network *engaged* with the manifold but
# couldn't extract a net benefit at that limited scale.
#
# Phase 4 tests the natural follow-up hypothesis: that hyperbolic geometry
# needs to be *consistently present* throughout the feature extraction path
# to provide its hierarchical-embedding benefit. Concretely, the entire
# encoder (`inc` + four `Down` blocks) operates on the Poincaré ball; the
# decoder (`Up` blocks + output head) stays Euclidean. The image is exp-mapped
# onto the ball at the input; features are log-mapped back to Euclidean at the
# bottleneck and at each skip connection before flowing into the decoder.
#
# **Why hybrid encoder/decoder rather than full hyperbolic.** The decoder's
# job is pixel reconstruction — predicting precise Euclidean RGB values. The
# encoder's job is hierarchical feature extraction — exactly the workload
# hyperbolic embeddings are theoretically suited to. Splitting them this way
# tests the cleanest version of the hypothesis: "is hyperbolic representation
# useful for extracting features from retinal images?" without conflating it
# with "is hyperbolic representation useful for predicting pixels?"
#
# **Architecture summary.**
#
# - Image → `expmap0` → ManifoldTensor at 256×256, 3 channels on the ball.
# - `inc`: 2× `HConvolution2d(3→32, k=3, padding=1)` + `HReLU`. Output stays on
#   the manifold, shape `[B, 32, 256, 256]`.
# - `down[0..3]`: each is `HMaxPool2d` + 2× `HConvolution2d` + `HReLU`. Channel
#   widths 32/64/128/256/512, exactly matching the Euclidean baseline.
# - At each level the encoder also produces a **skip**: log-mapped back to
#   Euclidean at the moment it's saved, so the decoder receives ordinary
#   Euclidean tensors.
# - Bottleneck: log-map the deepest feature map (`[B, 512, 16, 16]`) back to
#   Euclidean. Decoder is Phase 2/3 Euclidean code, unchanged.
# - Output head: zero-initialized 1×1 conv predicting residual noise. Forward
#   returns `noisy − noise_pred`.
#
# **Other design choices.**
#
# - **Curvature `c=0.5`, trainable.** Initialize closer to where Phase 3.1's
#   trainable-c converged (≈0.69) rather than the Mishra default. The network
#   already showed it prefers more aggressive curvature than the literature
#   default; we just give it a head start.
# - **No AMP.** Same numerical-stability constraint as Phase 3 — fp16 is
#   unsafe near the Poincaré boundary, and we now have many more hyperbolic
#   ops per forward pass.
# - **RiemannianAdam.** Handles the mixed encoder (manifold params) and
#   decoder (Euclidean params) automatically.
# - **Padding.** `HConvolution2d(padding=1)` is the primary path; if hypll's
#   version doesn't support that kwarg we fall back to manual pre-padding
#   per conv (with a runtime detection in the build cell).
# - **Outputs suffixed `_henc`** so they don't collide with Phase 3.1 artifacts.
#
# **Runtime estimate.** ~5-7 h on T4 (fp32, many more hyperbolic ops than
# Phase 3.1's single block). Plan a long session.

# %% [markdown]
# ## 0. Install + 1. Imports + 2. Config

# %% Install
import subprocess, sys
def _pip(*pkgs):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])
try:
    import torchmetrics  # noqa: F401
except ImportError:
    _pip("torchmetrics==1.4.0")
try:
    import hypll  # noqa: F401
except ImportError:
    _pip("hypll==0.1.1")
print("Packages OK.")

# %% Imports
import os, json, math, random, time, inspect
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

# %% Config
@dataclass
class Config:
    out_dir:   str = "/kaggle/working"
    image_size: int = 384
    patch_size: int = 256
    num_workers: int = 2

    base_channels: int = 32
    depth: int = 4

    curvature: float = 0.5            # Phase 3.1 learned ≈0.69; start closer
    trainable_curvature: bool = True
    hyperbolic_kernel: int = 3

    batch_size: int = 4         # Phase 4 encoder is ~4-5x more memory-hungry than Euclidean;
                                # batch=16 OOMs on T4 (14.5 GB). lr=2e-4 handles batch=4 fine.
    epochs: int = 30
    lr: float = 2e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 1
    grad_clip: float = 1.0
    amp: bool = False

    eval_every: int = 1
    viz_every: int = 5
    suffix: str = "_henc"

cfg = Config()
Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
print("Config:")
for k, v in asdict(cfg).items():
    print(f"  {k:>22}: {v}")

# %% [markdown]
# ## 3. Locate official RFMiD splits
# (Identical path-finder to Phase 2.1 / 3.1.)

# %% Find split folders
EXTS = (".png", ".jpg", ".jpeg")
DATA_ROOT_CANDIDATES = [
    Path("/kaggle/input/datasets/andrewmvd/retinal-disease-classification"),
    Path("/kaggle/input/retinal-disease-classification"),
    Path("./retinal-disease-classification"),
]
DATA_ROOT = next((p for p in DATA_ROOT_CANDIDATES if p.exists()), None)
assert DATA_ROOT is not None, "RFMiD dataset not found"
print(f"Data root: {DATA_ROOT}")

def find_split_images(root, set_dir, inner_dir):
    for c in [root / set_dir / set_dir / inner_dir,
              root / set_dir / inner_dir,
              root / inner_dir]:
        if c.is_dir():
            imgs = sorted(p for p in c.iterdir() if p.suffix.lower() in EXTS)
            if len(imgs) > 0:
                print(f"  {inner_dir:>12}: {len(imgs)} images @ {c.relative_to(root)}")
                return imgs
    raise FileNotFoundError(f"No images for {set_dir}/{inner_dir}")

train_paths = find_split_images(DATA_ROOT, "Training_Set",   "Training")
val_paths   = find_split_images(DATA_ROOT, "Evaluation_Set", "Validation")
test_paths  = find_split_images(DATA_ROOT, "Test_Set",       "Test")
print(f"\nTotal: train={len(train_paths)}  val={len(val_paths)}  test={len(test_paths)}")

# %% [markdown]
# ## 4. Noise + Dataset + loaders
# (Identical to Phase 2.1 / 3.1.)

# %% Noise + dataset
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

class RFMiDDataset(Dataset):
    def __init__(self, paths, image_size=384, patch_size=256,
                 noise_levels=None, fixed_noise=None,
                 augment=True, deterministic_seed=None):
        self.paths = list(paths); self.image_size = image_size; self.patch_size = patch_size
        self.noise_levels = noise_levels or NOISE_LEVELS
        self.fixed_noise = fixed_noise
        self.augment = augment; self.deterministic_seed = deterministic_seed
        self._level_names = list(self.noise_levels.keys())
    def __len__(self): return len(self.paths)
    def _load(self, path):
        with Image.open(path) as im:
            im = im.convert("RGB").resize((self.image_size, self.image_size), Image.BILINEAR)
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
        clean = self._load(path); clean = self._augment(clean, rng)
        level_name, params = self._sample_noise_params(item_seed)
        noisy = add_poisson_gaussian_noise(clean, params["alpha"], params["sigma"],
                                           rng=np.random.default_rng(item_seed + 1))
        clean_t = torch.from_numpy(clean).permute(2, 0, 1).float() / 255.0
        noisy_t = torch.from_numpy(noisy).permute(2, 0, 1).float() / 255.0
        return noisy_t, clean_t, level_name

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
# ## 5. Padding-support probe + helper
#
# `HConvolution2d` may or may not accept a `padding` argument depending on the
# hypll version. We probe its signature once and dispatch accordingly. If
# padding isn't supported, we'll fall back to manual logmap → F.pad → expmap
# around each conv (more expensive but reliable).

# %% Probe + helper
def _conv_supports_padding():
    """Check whether hypll's HConvolution2d accepts a `padding` kwarg."""
    try:
        sig = inspect.signature(hnn.HConvolution2d.__init__)
        if "padding" in sig.parameters:
            return True
    except Exception:
        pass
    return False

HCONV_HAS_PADDING = _conv_supports_padding()
print(f"hypll HConvolution2d supports padding kwarg: {HCONV_HAS_PADDING}")

def make_hconv(in_ch, out_ch, kernel_size, manifold):
    """Build a hyperbolic conv that preserves spatial dims with kernel_size=3."""
    if HCONV_HAS_PADDING:
        return hnn.HConvolution2d(
            in_channels=in_ch, out_channels=out_ch,
            kernel_size=kernel_size, padding=kernel_size // 2,
            manifold=manifold,
        )
    # Fallback: vanilla hconv, caller must pre-pad
    return hnn.HConvolution2d(
        in_channels=in_ch, out_channels=out_ch,
        kernel_size=kernel_size, manifold=manifold,
    )

# %% [markdown]
# ## 6. Hyperbolic encoder building blocks
#
# - `HyperbolicDoubleConv`: two hyperbolic 3×3 convs with `HReLU` in between.
#   In-place spatial preservation via `padding=1` when supported, or via the
#   logmap-pad-expmap dance otherwise.
# - `HyperbolicDown`: `HMaxPool2d(2)` then `HyperbolicDoubleConv` widening
#   channels by 2.

# %% Encoder blocks
class HyperbolicDoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, manifold, kernel_size=3):
        super().__init__()
        self.manifold = manifold
        self.hconv1 = make_hconv(in_ch, out_ch, kernel_size, manifold)
        self.hrelu  = hnn.HReLU(manifold=manifold)
        self.hconv2 = make_hconv(out_ch, out_ch, kernel_size, manifold)
        self.kernel_size = kernel_size
        self._needs_manual_pad = not HCONV_HAS_PADDING
        # If we need manual padding, the spatial loss per conv is (k - 1)
        # symmetric. We'll pad before each conv via Euclidean F.pad after
        # logmap → expmap round-trip.
        self._pad = (kernel_size - 1) // 2

    def _padded_conv(self, x_manifold, hconv):
        # x_manifold is a ManifoldTensor. Logmap → pad → expmap → hconv.
        # Only called when HCONV_HAS_PADDING is False.
        x_tan_t = self.manifold.logmap(None, x_manifold)
        x_euc = x_tan_t.tensor
        x_pad = F.pad(x_euc, [self._pad] * 4, mode='replicate')
        tan = TangentTensor(data=x_pad, man_dim=1, manifold=self.manifold)
        x_pad_m = self.manifold.expmap(tan)
        return hconv(x_pad_m)

    def forward(self, x_manifold):
        if self._needs_manual_pad:
            h = self._padded_conv(x_manifold, self.hconv1)
            h = self.hrelu(h)
            h = self._padded_conv(h, self.hconv2)
        else:
            h = self.hconv1(x_manifold)
            h = self.hrelu(h)
            h = self.hconv2(h)
        return h


class HyperbolicDown(nn.Module):
    """HMaxPool2d(2) then HyperbolicDoubleConv. Manifold-tensor in, manifold-tensor out."""
    def __init__(self, in_ch, out_ch, manifold, kernel_size=3):
        super().__init__()
        self.pool = hnn.HMaxPool2d(kernel_size=2, manifold=manifold, stride=2)
        self.conv = HyperbolicDoubleConv(in_ch, out_ch, manifold=manifold,
                                         kernel_size=kernel_size)
    def forward(self, x_manifold):
        return self.conv(self.pool(x_manifold))


# %% [markdown]
# ## 7. Euclidean decoder building blocks (unchanged from Phase 2.1)

# %% Decoder blocks
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
# ## 8. UNetDenoiser — hyperbolic encoder + Euclidean decoder
#
# Forward semantics:
#
# 1. Image (Euclidean) → expmap → ManifoldTensor at input.
# 2. `inc`, then four `HyperbolicDown` blocks. After each, **logmap that
#    feature map to Euclidean** and append it to `skips`. The decoder uses
#    Euclidean skips.
# 3. After the deepest encoder block, logmap once more to feed the Euclidean
#    decoder.
# 4. Decoder: standard `Up` blocks consume Euclidean skips and the
#    logmapped bottleneck.
# 5. Head: zero-initialized 1×1 conv predicts residual noise.
# 6. Forward returns `noisy − noise_pred` (residual denoising, same as
#    Phase 2 / 3 / 3.1).

# %% UNet
class UNetDenoiser(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, base=32, depth=4,
                 hyp_kernel=3, curvature=0.5, trainable_curvature=True):
        super().__init__()
        widths = [base * (2 ** i) for i in range(depth + 1)]
        self.depth = depth

        # One shared manifold across the whole encoder; curvature can be
        # trainable. Using a single curvature parameter (rather than per-layer)
        # matches Mishra et al.
        self.manifold = PoincareBall(
            c=Curvature(value=curvature, requires_grad=trainable_curvature)
        )

        # Hyperbolic encoder
        self.inc   = HyperbolicDoubleConv(in_ch, widths[0], manifold=self.manifold,
                                          kernel_size=hyp_kernel)
        self.downs = nn.ModuleList([
            HyperbolicDown(widths[i], widths[i + 1], manifold=self.manifold,
                           kernel_size=hyp_kernel)
            for i in range(depth)
        ])

        # Euclidean decoder (unchanged from Phase 2.1)
        self.ups   = nn.ModuleList([
            Up(in_ch=widths[depth - i], skip_ch=widths[depth - i - 1],
               out_ch=widths[depth - i - 1])
            for i in range(depth)
        ])
        self.head = nn.Conv2d(widths[0], out_ch, 1)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

    def _to_manifold(self, x_euc):
        """Euclidean (B, C, H, W) → ManifoldTensor on the Poincaré ball."""
        tan = TangentTensor(data=x_euc, man_dim=1, manifold=self.manifold)
        return self.manifold.expmap(tan)

    def _to_euclidean(self, x_manifold):
        """ManifoldTensor → Euclidean (B, C, H, W) via logmap at origin."""
        tan = self.manifold.logmap(None, x_manifold)
        return tan.tensor

    def forward(self, x):
        x_in = x  # save Euclidean input for the residual subtraction at the end

        # Lift onto the ball
        h_m = self._to_manifold(x)

        # Encoder, with Euclidean skips
        h_m = self.inc(h_m)
        skips = [self._to_euclidean(h_m)]
        for down in self.downs:
            h_m = down(h_m)
            skips.append(self._to_euclidean(h_m))

        # Bottleneck → Euclidean
        h = self._to_euclidean(h_m)

        # Decoder
        for i, up in enumerate(self.ups):
            skip = skips[-i - 2]
            h = up(h, skip)

        return x_in - self.head(h)

# Build + count + sanity
model = UNetDenoiser(
    in_ch=3, out_ch=3, base=cfg.base_channels, depth=cfg.depth,
    hyp_kernel=cfg.hyperbolic_kernel,
    curvature=cfg.curvature,
    trainable_curvature=cfg.trainable_curvature,
).to(DEVICE)

def _numel(p):
    return p.tensor.numel() if hasattr(p, 'tensor') else p.numel()

n_params       = sum(_numel(p) for p in model.parameters())
n_train_params = sum(_numel(p) for p in model.parameters() if p.requires_grad)
n_enc_params   = sum(_numel(p) for p in model.inc.parameters()) \
                 + sum(_numel(p) for p in model.downs.parameters())
n_dec_params   = sum(_numel(p) for p in model.ups.parameters()) \
                 + sum(_numel(p) for p in model.head.parameters())
print(f"UNetDenoiser  total: {n_params/1e6:.2f}M  "
      f"trainable: {n_train_params/1e6:.2f}M  "
      f"encoder(hyp): {n_enc_params/1e6:.2f}M  decoder(euc): {n_dec_params/1e6:.2f}M")

with torch.no_grad():
    _x = torch.randn(1, 3, cfg.patch_size, cfg.patch_size, device=DEVICE)
    _y = model(_x)
print(f"Forward sanity: in={tuple(_x.shape)}  out={tuple(_y.shape)}  "
      f"finite={torch.isfinite(_y).all().item()}")
del _x, _y

# %% [markdown]
# ## 9. Loss + RiemannianAdam + cosine schedule + metrics
# (Same as Phase 3.1; `RiemannianAdam` handles the encoder's manifold params
# and the decoder's Euclidean params automatically.)

# %% Optim + metrics
criterion = nn.L1Loss()
optimizer = RiemannianAdam(model.parameters(),
                           lr=cfg.lr, weight_decay=cfg.weight_decay)

steps_per_epoch = max(1, len(train_loader))
total_steps  = cfg.epochs * steps_per_epoch
warmup_steps = cfg.warmup_epochs * steps_per_epoch
def lr_lambda(step):
    if step < warmup_steps:
        return float(step + 1) / float(max(1, warmup_steps))
    progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return 0.5 * (1.0 + math.cos(math.pi * progress))
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)
ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)
def reset_metrics(): psnr_metric.reset(); ssim_metric.reset()
print(f"steps_per_epoch={steps_per_epoch}  total_steps={total_steps}  "
      f"warmup_steps={warmup_steps}  amp=False (fp32 for hyperbolic stability)")

# %% [markdown]
# ## 10. Train / val / viz

# %% Train / val / viz
def train_one_epoch(epoch):
    model.train()
    running = 0.0; n_seen = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{cfg.epochs} [train]", leave=False)
    for noisy, clean, _ in pbar:
        noisy = noisy.to(DEVICE, non_blocking=True); clean = clean.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        denoised = model(noisy); loss = criterion(denoised, clean)
        loss.backward()
        if cfg.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                [p.tensor if hasattr(p, 'tensor') else p for p in model.parameters()],
                cfg.grad_clip,
            )
        optimizer.step(); scheduler.step()
        bs = noisy.size(0); running += loss.item() * bs; n_seen += bs
        pbar.set_postfix(loss=f"{running/n_seen:.4f}",
                         lr=f"{scheduler.get_last_lr()[0]:.2e}")
    return running / max(1, n_seen)

@torch.no_grad()
def validate(epoch):
    model.eval(); reset_metrics()
    running = 0.0; n_seen = 0
    pbar = tqdm(val_loader, desc=f"Epoch {epoch:02d}/{cfg.epochs} [val]  ", leave=False)
    for noisy, clean, _ in pbar:
        noisy = noisy.to(DEVICE, non_blocking=True); clean = clean.to(DEVICE, non_blocking=True)
        denoised = model(noisy).clamp(0.0, 1.0)
        loss = criterion(denoised, clean)
        psnr_metric.update(denoised, clean); ssim_metric.update(denoised, clean)
        bs = noisy.size(0); running += loss.item() * bs; n_seen += bs
    return (running / max(1, n_seen),
            psnr_metric.compute().item(),
            ssim_metric.compute().item())

@torch.no_grad()
def save_qualitative(epoch, n_examples=3):
    model.eval(); examples = []
    for idx in range(min(n_examples, len(val_ds))):
        noisy, clean, level = val_ds[idx]
        denoised = model(noisy.unsqueeze(0).to(DEVICE)).clamp(0.0, 1.0).squeeze(0).cpu()
        examples.append((noisy, denoised, clean, level))
    fig, axes = plt.subplots(n_examples, 3, figsize=(12, 4 * n_examples))
    if n_examples == 1: axes = axes[None, :]
    for i, (noisy, denoised, clean, level) in enumerate(examples):
        for ax, img, title in zip(axes[i], [noisy, denoised, clean],
                                  [f"noisy ({level})", "denoised (hyp encoder)", "clean"]):
            ax.imshow(img.permute(1, 2, 0).numpy())
            ax.set_title(title); ax.axis("off")
    plt.tight_layout()
    out = Path(cfg.out_dir) / f"viz_epoch{epoch:02d}{cfg.suffix}.png"
    plt.savefig(out, bbox_inches="tight"); plt.close()
    return out

# %% [markdown]
# ## 11. Training driver
#
# Tracks the learned curvature each epoch alongside the loss / PSNR curves so
# we can see whether `c` drifts again the way it did in Phase 3.1.

# %% Train
history = {"epoch": [], "train_loss": [], "val_loss": [],
           "val_psnr": [], "val_ssim": [], "lr": [], "curvature": []}
best_psnr = -1.0
best_ckpt_path = Path(cfg.out_dir) / f"unet_denoiser_best{cfg.suffix}.pt"
t0 = time.time()

for epoch in range(1, cfg.epochs + 1):
    train_loss = train_one_epoch(epoch)
    if epoch % cfg.eval_every == 0:
        val_loss, val_psnr, val_ssim = validate(epoch)
    else:
        val_loss = val_psnr = val_ssim = float("nan")

    try:
        cur_c = float(model.manifold.c().detach().cpu().item())
    except Exception:
        cur_c = float(cfg.curvature)

    history["epoch"].append(epoch); history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss); history["val_psnr"].append(val_psnr)
    history["val_ssim"].append(val_ssim); history["lr"].append(scheduler.get_last_lr()[0])
    history["curvature"].append(cur_c)

    elapsed = time.time() - t0
    print(f"epoch {epoch:02d}/{cfg.epochs}  "
          f"train_L1={train_loss:.4f}  val_L1={val_loss:.4f}  "
          f"PSNR={val_psnr:.2f}  SSIM={val_ssim:.4f}  c={cur_c:.4f}  "
          f"lr={scheduler.get_last_lr()[0]:.2e}  ({elapsed/60:.1f} min total)")

    if val_psnr > best_psnr:
        best_psnr = val_psnr
        torch.save({"epoch": epoch, "model_state": model.state_dict(),
                    "optim_state": optimizer.state_dict(),
                    "config": asdict(cfg), "val_psnr": val_psnr, "val_ssim": val_ssim,
                    "curvature": cur_c},
                   best_ckpt_path)
        print(f"  ↳ new best PSNR={val_psnr:.2f}  saved to {best_ckpt_path.name}")

    if epoch % cfg.viz_every == 0 or epoch == cfg.epochs:
        viz = save_qualitative(epoch)
        print(f"  ↳ qualitative dump: {viz.name}")

print(f"\nTraining done in {(time.time()-t0)/60:.1f} min. Best val PSNR = {best_psnr:.2f} dB.")

# %% [markdown]
# ## 12. Training curves (loss / PSNR / learned curvature)

# %% Curves
fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(history["epoch"], history["train_loss"], label="train L1", color="C0")
ax1.plot(history["epoch"], history["val_loss"],   label="val L1",   color="C1")
ax1.set_xlabel("epoch"); ax1.set_ylabel("L1 loss")
ax1.legend(loc="upper left"); ax1.grid(alpha=0.3)
ax2 = ax1.twinx()
ax2.plot(history["epoch"], history["val_psnr"], label="val PSNR", color="C2",
         linestyle="--", marker="o", markersize=3)
ax2.set_ylabel("PSNR (dB)"); ax2.legend(loc="upper right")
ax1.set_title("Loss / PSNR")
ax3.plot(history["epoch"], history["curvature"], color="C3", marker="o", markersize=3)
ax3.axhline(0.6931, color="gray", linestyle=":", alpha=0.6,
            label="Phase 3.1 learned c ≈ 0.69")
ax3.set_xlabel("epoch"); ax3.set_ylabel("curvature c")
ax3.set_title("Learned curvature trajectory")
ax3.legend(); ax3.grid(alpha=0.3)
plt.tight_layout()
curves_path = Path(cfg.out_dir) / f"training_curves{cfg.suffix}.png"
plt.savefig(curves_path, bbox_inches="tight"); plt.show()
print(f"Saved {curves_path}")

# %% [markdown]
# ## 13. Per-level eval + head-to-head with Phase 2.1 and Phase 3.1

# %% Eval
ckpt = torch.load(best_ckpt_path, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt["model_state"]); model.eval()
print(f"Loaded Phase 4 checkpoint from epoch {ckpt['epoch']} "
      f"(val PSNR {ckpt['val_psnr']:.2f}, learned c={ckpt.get('curvature', cfg.curvature):.4f}).")

@torch.no_grad()
def evaluate_level(level_name):
    ds = RFMiDDataset(test_paths, cfg.image_size, cfg.patch_size,
                      augment=False, deterministic_seed=7, fixed_noise=level_name)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=cfg.num_workers, pin_memory=True)
    reset_metrics()
    psnr_in = PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)
    ssim_in = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)
    n_total = 0
    for noisy, clean, _ in tqdm(loader, desc=f"test [{level_name}]", leave=False):
        noisy = noisy.to(DEVICE, non_blocking=True); clean = clean.to(DEVICE, non_blocking=True)
        denoised = model(noisy).clamp(0.0, 1.0)
        psnr_metric.update(denoised, clean); ssim_metric.update(denoised, clean)
        psnr_in.update(noisy, clean); ssim_in.update(noisy, clean)
        n_total += noisy.size(0)
    return {"n": n_total,
            "psnr_in":  psnr_in.compute().item(),  "ssim_in":  ssim_in.compute().item(),
            "psnr_out": psnr_metric.compute().item(), "ssim_out": ssim_metric.compute().item()}

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

# Hardcoded reference numbers from Phase 2.1 and Phase 3.1 for inline comparison
P21_PER_LEVEL = {  # Euclidean baseline, official splits
    "light":  {"psnr_out": 40.98, "ssim_out": 0.9535},
    "medium": {"psnr_out": 39.61, "ssim_out": 0.9390},
    "heavy":  {"psnr_out": 38.39, "ssim_out": 0.9254},
}
P31_PER_LEVEL = {  # Optimized Poincaré (bottleneck-adjacent + residual + trainable c)
    "light":  {"psnr_out": 40.84, "ssim_out": 0.9536},
    "medium": {"psnr_out": 39.47, "ssim_out": 0.9398},
    "heavy":  {"psnr_out": 38.03, "ssim_out": 0.9218},
}

print()
print(f"{'level':>8}  "
      f"{'P2.1':>6}  {'P3.1':>6}  {'P4':>6}  "
      f"{'P4 vs P2.1':>11}  {'P4 vs P3.1':>11}")
for name in NOISE_LEVELS:
    a = P21_PER_LEVEL[name]; b = P31_PER_LEVEL[name]; c = per_level[name]
    print(f"{name:>8}  "
          f"{a['psnr_out']:>6.2f}  {b['psnr_out']:>6.2f}  {c['psnr_out']:>6.2f}  "
          f"{c['psnr_out']-a['psnr_out']:>+11.2f}  "
          f"{c['psnr_out']-b['psnr_out']:>+11.2f}")

# %% [markdown]
# ## 14. Final qualitative + persist

# %% Save
@torch.no_grad()
def final_qualitative():
    fig, axes = plt.subplots(len(NOISE_LEVELS), 3, figsize=(12, 4 * len(NOISE_LEVELS)))
    for i, level in enumerate(NOISE_LEVELS):
        ds = RFMiDDataset([test_paths[0]], cfg.image_size, cfg.patch_size,
                          augment=False, deterministic_seed=99, fixed_noise=level)
        noisy, clean, _ = ds[0]
        denoised = model(noisy.unsqueeze(0).to(DEVICE)).clamp(0.0, 1.0).squeeze(0).cpu()
        for ax, img, title in zip(axes[i], [noisy, denoised, clean],
                                  [f"noisy ({level})", "denoised (hyp enc)", "clean"]):
            ax.imshow(img.permute(1, 2, 0).numpy())
            ax.set_title(title); ax.axis("off")
    plt.tight_layout()
    out = Path(cfg.out_dir) / f"final_qualitative{cfg.suffix}.png"
    plt.savefig(out, bbox_inches="tight"); plt.show()
    return out

final_q = final_qualitative()
print(f"Saved {final_q}")

results = {
    "phase":          "phase4_hyperbolic_encoder_euclidean_decoder",
    "config":         asdict(cfg),
    "splits":         {"train": len(train_paths), "val": len(val_paths), "test": len(test_paths)},
    "hcv_has_padding": HCONV_HAS_PADDING,
    "best_epoch":     int(ckpt["epoch"]),
    "best_val_psnr":  float(ckpt["val_psnr"]),
    "best_val_ssim":  float(ckpt["val_ssim"]),
    "final_curvature": float(ckpt.get("curvature", cfg.curvature)),
    "test_per_level": per_level,
    "noise_levels":   NOISE_LEVELS,
    "n_params":       int(n_params),
    "n_encoder_hyperbolic_params": int(n_enc_params),
    "n_decoder_euclidean_params":  int(n_dec_params),
    "phase2_1_reference": P21_PER_LEVEL,
    "phase3_1_reference": P31_PER_LEVEL,
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
# ## 15. Interpretation guide
#
# - **If Phase 4 beats Phase 2.1**, the hypothesis "hyperbolic feature
#   extraction helps retinal denoising" is supported. The decoder being
#   Euclidean would not be the limiting factor, since the wins come from
#   the encoder.
# - **If Phase 4 ties Phase 2.1 (±0.1 dB)**, the geometry probably needs to
#   be present in the decoder too — Phase 5 would be the full hyperbolic
#   U-Net.
# - **If Phase 4 loses to Phase 2.1 by a margin similar to Phase 3.1's
#   (≈0.2 dB)**, partial hyperbolic interventions don't help even when
#   "partial" means the entire encoder. At that point either full hyperbolic
#   or a different angle (perceptual loss, Lorentz, downstream-task eval)
#   becomes the right next step.
# - **Watch the learned curvature in §12's right panel.** If it lands close
#   to Phase 3.1's c≈0.69 again, that's reproducibility evidence. If it
#   settles elsewhere given the changed architecture, that's also data.
