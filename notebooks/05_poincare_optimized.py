# %% [markdown]
# # RFMiD Denoising U-Net — Phase 3.1 (Optimized Poincaré + official splits)
#
# **Three Poincaré optimizations on top of Phase 3:**
#
# 1. **Move the hyperbolic block one level earlier** — from after `down[3]`
#    (16×16, 512 ch) to after `down[2]` (32×32, 256 ch). Conv FLOPs stay
#    roughly constant (H·W·C² is invariant when you halve C and double H/W),
#    but the features at 32×32 still carry some vessel-level structure that's
#    been pooled away by the bottleneck.
#
# 2. **Residual skip around the hyperbolic block** — `out = x + HypBlock(x)`.
#    The block can now degrade to identity if the manifold round-trip isn't
#    helping. Guaranteed floor at Phase 2.1 performance.
#
# 3. **Trainable curvature** — `c` becomes a learnable parameter. Lets the
#    network find its own optimal manifold curvature instead of committing to
#    the Mishra default `c=0.1`.
#
# **Plus:** the same official RFMiD splits used by Phase 2.1, so the
# comparison is fully apples-to-apples.
#
# **Outputs:** suffixed `_hyp_v2` so they don't collide with Phase 3.
#
# **Runtime:** ~3–4 h on T4 (fp32, AMP off for Poincaré stability).

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
import os, json, math, random, time
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

    # Model
    base_channels: int = 32
    depth: int = 4

    # Hyperbolic
    curvature: float = 0.1
    trainable_curvature: bool = True       # CHANGE 3: trainable
    hyperbolic_kernel: int = 3
    hyp_after_down: int = 2                # CHANGE 1: move from 3 (bottleneck) → 2 (one level up)
    residual_hyperbolic: bool = True       # CHANGE 2: residual skip

    # Optimization
    batch_size: int = 16
    epochs: int = 30
    lr: float = 2e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 1
    grad_clip: float = 1.0
    amp: bool = False                      # fp16 unsafe near Poincaré boundary

    # Eval / I/O
    eval_every: int = 1
    viz_every: int = 5
    suffix: str = "_hyp_v2"

cfg = Config()
Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
print("Config:")
for k, v in asdict(cfg).items():
    print(f"  {k:>22}: {v}")

# %% [markdown]
# ## 3. Locate the official RFMiD splits
# (Identical path-finding logic to Phase 2.1 so the splits match exactly.)

# %% Find split folders
EXTS = (".png", ".jpg", ".jpeg")

DATA_ROOT_CANDIDATES = [
    Path("/kaggle/input/datasets/andrewmvd/retinal-disease-classification"),
    Path("/kaggle/input/retinal-disease-classification"),
    Path("./retinal-disease-classification"),
]
DATA_ROOT = next((p for p in DATA_ROOT_CANDIDATES if p.exists()), None)
assert DATA_ROOT is not None, (
    "RFMiD dataset not found. Checked:\n  "
    + "\n  ".join(str(p) for p in DATA_ROOT_CANDIDATES)
)
print(f"Data root: {DATA_ROOT}")

def find_split_images(root, set_dir, inner_dir):
    candidates = [
        root / set_dir / set_dir / inner_dir,
        root / set_dir / inner_dir,
        root / inner_dir,
    ]
    for c in candidates:
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
# ## 4. Noise model + Dataset + loaders
# (Identical to Phase 2.1.)

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

class RFMiDDataset(Dataset):
    def __init__(self, paths, image_size=384, patch_size=256,
                 noise_levels=None, fixed_noise=None,
                 augment=True, deterministic_seed=None):
        self.paths = list(paths)
        self.image_size = image_size; self.patch_size = patch_size
        self.noise_levels = noise_levels or NOISE_LEVELS
        self.fixed_noise = fixed_noise
        self.augment = augment; self.deterministic_seed = deterministic_seed
        self._level_names = list(self.noise_levels.keys())

    def __len__(self): return len(self.paths)

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
# ## 5. Euclidean U-Net blocks (unchanged)

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
# ## 6. HyperbolicBlock — now with residual skip
#
# `out = x + HypBlock(x)`. If the manifold round-trip is unhelpful, the
# optimizer can drive the block contribution toward zero, recovering the
# Euclidean Phase 2 behaviour. This is a literal architectural "escape valve."

# %% HyperbolicBlock
class HyperbolicBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3,
                 curvature: float = 0.1, trainable_curvature: bool = False,
                 residual: bool = True):
        super().__init__()
        self.manifold = PoincareBall(
            c=Curvature(value=curvature, requires_grad=trainable_curvature)
        )
        self.hconv1 = hnn.HConvolution2d(
            in_channels=channels, out_channels=channels,
            kernel_size=kernel_size, manifold=self.manifold,
        )
        self.hrelu  = hnn.HReLU(manifold=self.manifold)
        self.hconv2 = hnn.HConvolution2d(
            in_channels=channels, out_channels=channels,
            kernel_size=kernel_size, manifold=self.manifold,
        )
        self.residual = residual
        # hypll's HConvolution2d does not pad; two consecutive k×k convs shrink
        # each spatial dim by 2*(k-1). Pre-pad by (k-1) per side so the block's
        # output matches the input shape — required for the residual skip and
        # cleaner for the rest of the U-Net regardless.
        self._pad = kernel_size - 1

    def forward(self, x_euc):
        if self._pad > 0:
            x_padded = F.pad(x_euc, [self._pad] * 4, mode='replicate')
        else:
            x_padded = x_euc
        tangent = TangentTensor(data=x_padded, man_dim=1, manifold=self.manifold)
        h = self.manifold.expmap(tangent)
        h = self.hconv1(h); h = self.hrelu(h); h = self.hconv2(h)
        h_tan = self.manifold.logmap(None, h)
        out = h_tan.tensor                       # shape now matches x_euc
        if self.residual:
            out = x_euc + out
        return out

# %% [markdown]
# ## 7. UNetDenoiser — hyperbolic block insertion is now configurable
#
# `hyp_after_down=2` puts the block after the third Down (32×32×256 features).
# Setting it to 3 would restore Phase 3's bottleneck behaviour.

# %% UNet
class UNetDenoiser(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, base=32, depth=4,
                 hyp_after_down=2,
                 hyp_kernel=3, curvature=0.1, trainable_curvature=False,
                 residual_hyp=True):
        super().__init__()
        widths = [base * (2 ** i) for i in range(depth + 1)]
        self.depth = depth
        self.hyp_after_down = hyp_after_down

        self.inc   = DoubleConv(in_ch, widths[0])
        self.downs = nn.ModuleList([Down(widths[i], widths[i + 1]) for i in range(depth)])

        # The hyperbolic block sits after down[hyp_after_down], so its channel
        # count is widths[hyp_after_down + 1].
        hyp_channels = widths[hyp_after_down + 1]
        self.hyperbolic = HyperbolicBlock(
            channels=hyp_channels, kernel_size=hyp_kernel,
            curvature=curvature, trainable_curvature=trainable_curvature,
            residual=residual_hyp,
        )

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
        for i, down in enumerate(self.downs):
            h = down(h)
            if i == self.hyp_after_down:
                # Apply hyperbolic transformation. Because we append AFTER, both
                # the next down AND the skip used by the matching up see the
                # post-hyperbolic features.
                h = self.hyperbolic(h)
            skips.append(h)
        for i, up in enumerate(self.ups):
            skip = skips[-i - 2]
            h = up(h, skip)
        return x - self.head(h)

# %% Build + sanity
model = UNetDenoiser(
    in_ch=3, out_ch=3, base=cfg.base_channels, depth=cfg.depth,
    hyp_after_down=cfg.hyp_after_down,
    hyp_kernel=cfg.hyperbolic_kernel,
    curvature=cfg.curvature,
    trainable_curvature=cfg.trainable_curvature,
    residual_hyp=cfg.residual_hyperbolic,
).to(DEVICE)

def _numel(p):
    return p.tensor.numel() if hasattr(p, 'tensor') else p.numel()

n_params       = sum(_numel(p) for p in model.parameters())
n_train_params = sum(_numel(p) for p in model.parameters() if p.requires_grad)
n_hyp_params   = sum(_numel(p) for p in model.hyperbolic.parameters())
print(f"UNetDenoiser  total: {n_params/1e6:.2f}M  "
      f"trainable: {n_train_params/1e6:.2f}M  "
      f"hyperbolic-block: {n_hyp_params/1e6:.2f}M")
print(f"Hyperbolic block: at 32×32 with 256 ch, residual={cfg.residual_hyperbolic}, "
      f"trainable_c={cfg.trainable_curvature}")

with torch.no_grad():
    _x = torch.randn(1, 3, cfg.patch_size, cfg.patch_size, device=DEVICE)
    _y = model(_x)
print(f"Forward sanity: in={tuple(_x.shape)}  out={tuple(_y.shape)}  "
      f"finite={torch.isfinite(_y).all().item()}")
del _x, _y

# %% [markdown]
# ## 8. Loss + RiemannianAdam + cosine schedule + metrics

# %% Optim
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
      f"warmup_steps={warmup_steps}  amp=False (Poincaré stability)")

# %% [markdown]
# ## 9. Train / val / viz

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
                                  [f"noisy ({level})", "denoised (hyp v2)", "clean"]):
            ax.imshow(img.permute(1, 2, 0).numpy())
            ax.set_title(title); ax.axis("off")
    plt.tight_layout()
    out = Path(cfg.out_dir) / f"viz_epoch{epoch:02d}{cfg.suffix}.png"
    plt.savefig(out, bbox_inches="tight"); plt.close()
    return out

# %% [markdown]
# ## 10. Training driver

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

    # Pull current curvature (informative when trainable_curvature=True)
    try:
        cur_c = float(model.hyperbolic.manifold.c().detach().cpu().item())
    except Exception:
        cur_c = float(cfg.curvature)

    history["epoch"].append(epoch); history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss); history["val_psnr"].append(val_psnr)
    history["val_ssim"].append(val_ssim); history["lr"].append(scheduler.get_last_lr()[0])
    history["curvature"].append(cur_c)

    elapsed = time.time() - t0
    print(f"epoch {epoch:02d}/{cfg.epochs}  "
          f"train_L1={train_loss:.4f}  val_L1={val_loss:.4f}  "
          f"PSNR={val_psnr:.2f}  SSIM={val_ssim:.4f}  "
          f"c={cur_c:.4f}  "
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
# ## 11. Training curves (incl. learned curvature)

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
ax3.set_xlabel("epoch"); ax3.set_ylabel("curvature c")
ax3.set_title("Learned curvature trajectory" if cfg.trainable_curvature else "Curvature (fixed)")
ax3.grid(alpha=0.3)

plt.tight_layout()
curves_path = Path(cfg.out_dir) / f"training_curves{cfg.suffix}.png"
plt.savefig(curves_path, bbox_inches="tight"); plt.show()
print(f"Saved {curves_path}")

# %% [markdown]
# ## 12. Per-noise-level eval on the official test split

# %% Per-level
ckpt = torch.load(best_ckpt_path, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt["model_state"]); model.eval()
print(f"Loaded best Phase 3.1 checkpoint from epoch {ckpt['epoch']}  "
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

# %% [markdown]
# ## 13. Head-to-head with Phase 2.1
#
# Load Phase 2.1's `results_v2.json` if it's attached to this notebook as an
# input (Add Data → upload the file). Otherwise prints a placeholder so you
# can fill the numbers in manually.

# %% Comparison
phase21_results_path = None
for candidate in [
    Path("/kaggle/input/phase2-1-results/results_v2.json"),
    Path("/kaggle/working/results_v2.json"),
]:
    if candidate.exists():
        phase21_results_path = candidate
        break

if phase21_results_path is not None:
    with open(phase21_results_path) as f:
        p2 = json.load(f)
    p2_per_level = p2["test_per_level"]
    p2_best_val = p2["best_val_psnr"]
    print(f"Loaded Phase 2.1 results from {phase21_results_path}")
    print(f"{'level':>8}  {'P2.1 PSNR':>10}  {'P3.1 PSNR':>10}  {'ΔPSNR':>7}  "
          f"{'P2.1 SSIM':>10}  {'P3.1 SSIM':>10}  {'ΔSSIM':>8}")
    for name in NOISE_LEVELS:
        a = p2_per_level[name]; b = per_level[name]
        print(f"{name:>8}  "
              f"{a['psnr_out']:>10.2f}  {b['psnr_out']:>10.2f}  "
              f"{b['psnr_out']-a['psnr_out']:>+7.2f}  "
              f"{a['ssim_out']:>10.4f}  {b['ssim_out']:>10.4f}  "
              f"{b['ssim_out']-a['ssim_out']:>+8.4f}")
    print(f"\nBest val PSNR: P2.1 = {p2_best_val:.2f} dB  |  "
          f"P3.1 = {ckpt['val_psnr']:.2f} dB  "
          f"({ckpt['val_psnr'] - p2_best_val:+.2f})")
else:
    print("Phase 2.1 results not found. Run Phase 2.1 first and either:")
    print("  (a) attach its results_v2.json as an input dataset to this notebook")
    print("  (b) or run both notebooks in the same Kaggle session so /kaggle/working/results_v2.json exists")

# %% [markdown]
# ## 14. Final qualitative + persist

# %% Final qual + save
@torch.no_grad()
def final_qualitative():
    fig, axes = plt.subplots(len(NOISE_LEVELS), 3, figsize=(12, 4 * len(NOISE_LEVELS)))
    for i, level in enumerate(NOISE_LEVELS):
        ds = RFMiDDataset([test_paths[0]], cfg.image_size, cfg.patch_size,
                          augment=False, deterministic_seed=99, fixed_noise=level)
        noisy, clean, _ = ds[0]
        denoised = model(noisy.unsqueeze(0).to(DEVICE)).clamp(0.0, 1.0).squeeze(0).cpu()
        for ax, img, title in zip(axes[i], [noisy, denoised, clean],
                                  [f"noisy ({level})", "denoised (hyp v2)", "clean"]):
            ax.imshow(img.permute(1, 2, 0).numpy())
            ax.set_title(title); ax.axis("off")
    plt.tight_layout()
    out = Path(cfg.out_dir) / f"final_qualitative{cfg.suffix}.png"
    plt.savefig(out, bbox_inches="tight"); plt.show()
    return out

final_q = final_qualitative()
print(f"Saved {final_q}")

results = {
    "phase":          "phase3_1_poincare_optimized_official_splits",
    "config":         asdict(cfg),
    "splits":         {"train": len(train_paths), "val": len(val_paths), "test": len(test_paths)},
    "best_epoch":     int(ckpt["epoch"]),
    "best_val_psnr":  float(ckpt["val_psnr"]),
    "best_val_ssim":  float(ckpt["val_ssim"]),
    "final_curvature": float(ckpt.get("curvature", cfg.curvature)),
    "test_per_level": per_level,
    "noise_levels":   NOISE_LEVELS,
    "n_params":       int(n_params),
    "n_hyperbolic_params": int(n_hyp_params),
    "changes_vs_phase3": [
        f"hyp_after_down: 3 (bottleneck) -> {cfg.hyp_after_down}",
        f"residual_hyperbolic: False -> {cfg.residual_hyperbolic}",
        f"trainable_curvature: False -> {cfg.trainable_curvature}",
        "split: random 80/10/10 -> official RFMiD Training/Validation/Test",
    ],
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
# After Phase 2.1 and Phase 3.1 both finish, the relevant numbers to compare
# are the per-level test table in §12 (Phase 3.1) vs the corresponding table
# from Phase 2.1's notebook. Look at:
#
# - **ΔPSNR at heavy noise.** This is the regime where hyperbolic should win
#   if it's going to. A +0.5 dB or larger here is a publishable finding.
# - **The learned curvature in §11's right panel.** If `c` settles near 0.1
#   (the initial value), the trainable-curvature option didn't matter — that's
#   information too. If it drifts substantially (say to 0.3 or 0.02), the
#   network found a better operating point we couldn't have hand-picked.
# - **Whether the residual skip went near-zero.** Compare `out` and `x` at
#   the hyperbolic block during inference; if `||out - x|| << ||x||`, the
#   network is using the skip to bypass the manifold. Diagnostic for Phase 4.
#
# If Phase 3.1 still loses to Phase 2.1, the natural next step is **full
# hyperbolic encoder + decoder** (matching Mishra et al. exactly) — the
# bottleneck-only and one-level-up interventions will have both proved
# insufficient, which is itself a strong signal that the geometry needs to
# be present throughout the network or not at all. Lorentz at that point
# becomes a worthwhile separate ablation for numerical-stability comparison.
