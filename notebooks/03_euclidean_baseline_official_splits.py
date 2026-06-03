# %% [markdown]
# # RFMiD Denoising U-Net — Phase 2.1 (Euclidean baseline on official splits)
#
# **What's new vs Phase 2:** the random 80/10/10 split is replaced with
# RFMiD's **official Training / Validation / Test folder splits** (1920 / 640 /
# 640 images per the dataset publication). Every other component — U-Net
# architecture, residual learning, L1 loss, AdamW + cosine, AMP, augmentations,
# noise model — is byte-identical to Phase 2.
#
# **Why this matters.** Using the author-provided splits makes our results
# directly comparable to every other paper on RFMiD, removes the methodological
# weakness of an unreproducible random partition, and gives us the proper
# apples-to-apples baseline for Phase 3.1 (optimized Poincaré).
#
# **Output naming.** All artifacts suffixed `_v2` so they coexist with the
# original Phase 2 outputs in `/kaggle/working`.
#
# **Runtime:** same as Phase 2, ~1.5–2 h on T4 with AMP.

# %% [markdown]
# ## 0. Install + 1. Imports + 2. Config
# (Setup is the same as Phase 2 — see `rfmid_unet_denoiser.ipynb` for the
# full design rationale on each choice.)

# %% Install
import subprocess, sys
def _pip(*pkgs):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])
try:
    import torchmetrics  # noqa: F401
except ImportError:
    _pip("torchmetrics==1.4.0")

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
from torch.cuda.amp import autocast, GradScaler

from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

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
    batch_size: int = 16
    epochs: int = 30
    lr: float = 2e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 1
    grad_clip: float = 1.0
    amp: bool = True
    eval_every: int = 1
    viz_every: int = 5
    suffix: str = "_v2"

cfg = Config()
Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
print("Config:")
for k, v in asdict(cfg).items():
    print(f"  {k:>16}: {v}")

# %% [markdown]
# ## 3. Locate the official RFMiD splits
#
# Kaggle attaches the dataset under one of two paths depending on how it was
# uploaded. Inside, each split lives in a nested folder structure like
# `Training_Set/Training_Set/Training/`. We auto-discover the right one.

# %% Find split folders
EXTS = (".png", ".jpg", ".jpeg")

DATA_ROOT_CANDIDATES = [
    Path("/kaggle/input/datasets/andrewmvd/retinal-disease-classification"),
    Path("/kaggle/input/retinal-disease-classification"),
    Path("./retinal-disease-classification"),
]
DATA_ROOT = next((p for p in DATA_ROOT_CANDIDATES if p.exists()), None)
assert DATA_ROOT is not None, (
    "RFMiD dataset not found. Checked:\n  " +
    "\n  ".join(str(p) for p in DATA_ROOT_CANDIDATES)
)
print(f"Data root: {DATA_ROOT}")

def find_split_images(root: Path, set_dir: str, inner_dir: str) -> list:
    """Robust split-folder discovery; tries the doubly-nested Kaggle layout
    plus the flat fallback layouts."""
    candidates = [
        root / set_dir / set_dir / inner_dir,   # most common Kaggle layout
        root / set_dir / inner_dir,             # single-nested
        root / inner_dir,                       # flat
    ]
    for c in candidates:
        if c.is_dir():
            imgs = sorted(p for p in c.iterdir()
                          if p.suffix.lower() in EXTS)
            if len(imgs) > 0:
                print(f"  {inner_dir:>12}: {len(imgs)} images @ {c.relative_to(root)}")
                return imgs
    raise FileNotFoundError(
        f"No images found for {set_dir}/{inner_dir}. Tried: "
        + ", ".join(str(c) for c in candidates)
    )

train_paths = find_split_images(DATA_ROOT, "Training_Set",   "Training")
val_paths   = find_split_images(DATA_ROOT, "Evaluation_Set", "Validation")
test_paths  = find_split_images(DATA_ROOT, "Test_Set",       "Test")

print(f"\nTotal: train={len(train_paths)}  val={len(val_paths)}  test={len(test_paths)}")
assert len(train_paths) > 100 and len(val_paths) > 50 and len(test_paths) > 50

# %% [markdown]
# ## 4. Noise model
# (Identical to Phase 2.)

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

# %% [markdown]
# ## 5. Dataset + loaders
# `RFMiDDataset` is identical to Phase 2. The only change is that we feed it
# three pre-computed path lists instead of doing a random shuffle.

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

# %% Build loaders
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
# ## 6. U-Net architecture
# (Identical to Phase 2 — 4-level, widths 32/64/128/256/512, GroupNorm + ReLU,
# bilinear-up + 1×1, zero-init residual head.)

# %% U-Net
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

class UNetDenoiser(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, base=32, depth=4):
        super().__init__()
        widths = [base * (2 ** i) for i in range(depth + 1)]
        self.inc   = DoubleConv(in_ch, widths[0])
        self.downs = nn.ModuleList([Down(widths[i], widths[i + 1]) for i in range(depth)])
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
        for i, up in enumerate(self.ups):
            skip = skips[-i - 2]
            h = up(h, skip)
        return x - self.head(h)

model = UNetDenoiser(3, 3, cfg.base_channels, cfg.depth).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
print(f"UNetDenoiser params: {n_params/1e6:.2f}M")
with torch.no_grad():
    _y = model(torch.randn(1, 3, cfg.patch_size, cfg.patch_size, device=DEVICE))
print(f"Forward OK. out shape = {tuple(_y.shape)}")
del _y

# %% [markdown]
# ## 7. Loss / optim / scheduler / metrics
# (Identical to Phase 2.)

# %% Optim
criterion = nn.L1Loss()
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

steps_per_epoch = max(1, len(train_loader))
total_steps  = cfg.epochs * steps_per_epoch
warmup_steps = cfg.warmup_epochs * steps_per_epoch

def lr_lambda(step):
    if step < warmup_steps:
        return float(step + 1) / float(max(1, warmup_steps))
    progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return 0.5 * (1.0 + math.cos(math.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
scaler = GradScaler(enabled=(cfg.amp and DEVICE.type == "cuda"))

psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)
ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)
def reset_metrics():
    psnr_metric.reset(); ssim_metric.reset()

# %% [markdown]
# ## 8. Train / val loops + viz
# (Identical to Phase 2.)

# %% Train / val / viz
def train_one_epoch(epoch):
    model.train()
    running = 0.0; n_seen = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{cfg.epochs} [train]", leave=False)
    for noisy, clean, _ in pbar:
        noisy = noisy.to(DEVICE, non_blocking=True)
        clean = clean.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=cfg.amp and DEVICE.type == "cuda"):
            denoised = model(noisy)
            loss = criterion(denoised, clean)
        scaler.scale(loss).backward()
        if cfg.grad_clip is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer); scaler.update(); scheduler.step()
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
        noisy = noisy.to(DEVICE, non_blocking=True)
        clean = clean.to(DEVICE, non_blocking=True)
        with autocast(enabled=cfg.amp and DEVICE.type == "cuda"):
            denoised = model(noisy).clamp(0.0, 1.0)
            loss = criterion(denoised, clean)
        psnr_metric.update(denoised, clean); ssim_metric.update(denoised, clean)
        bs = noisy.size(0); running += loss.item() * bs; n_seen += bs
    return (running / max(1, n_seen),
            psnr_metric.compute().item(),
            ssim_metric.compute().item())

@torch.no_grad()
def save_qualitative(epoch, n_examples=3):
    model.eval()
    examples = []
    for idx in range(min(n_examples, len(val_ds))):
        noisy, clean, level = val_ds[idx]
        denoised = model(noisy.unsqueeze(0).to(DEVICE)).clamp(0.0, 1.0).squeeze(0).cpu()
        examples.append((noisy, denoised, clean, level))
    fig, axes = plt.subplots(n_examples, 3, figsize=(12, 4 * n_examples))
    if n_examples == 1:
        axes = axes[None, :]
    for i, (noisy, denoised, clean, level) in enumerate(examples):
        for ax, img, title in zip(axes[i], [noisy, denoised, clean],
                                  [f"noisy ({level})", "denoised", "clean"]):
            ax.imshow(img.permute(1, 2, 0).numpy())
            ax.set_title(title); ax.axis("off")
    plt.tight_layout()
    out = Path(cfg.out_dir) / f"viz_epoch{epoch:02d}{cfg.suffix}.png"
    plt.savefig(out, bbox_inches="tight"); plt.close()
    return out

# %% [markdown]
# ## 9. Training driver

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

    history["epoch"].append(epoch); history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss); history["val_psnr"].append(val_psnr)
    history["val_ssim"].append(val_ssim); history["lr"].append(scheduler.get_last_lr()[0])

    elapsed = time.time() - t0
    print(f"epoch {epoch:02d}/{cfg.epochs}  "
          f"train_L1={train_loss:.4f}  val_L1={val_loss:.4f}  "
          f"PSNR={val_psnr:.2f}  SSIM={val_ssim:.4f}  "
          f"lr={scheduler.get_last_lr()[0]:.2e}  ({elapsed/60:.1f} min total)")

    if val_psnr > best_psnr:
        best_psnr = val_psnr
        torch.save({"epoch": epoch, "model_state": model.state_dict(),
                    "optim_state": optimizer.state_dict(),
                    "config": asdict(cfg), "val_psnr": val_psnr, "val_ssim": val_ssim},
                   best_ckpt_path)
        print(f"  ↳ new best PSNR={val_psnr:.2f}  saved to {best_ckpt_path.name}")

    if epoch % cfg.viz_every == 0 or epoch == cfg.epochs:
        viz = save_qualitative(epoch)
        print(f"  ↳ qualitative dump: {viz.name}")

print(f"\nTraining done in {(time.time()-t0)/60:.1f} min. Best val PSNR = {best_psnr:.2f} dB.")

# %% [markdown]
# ## 10. Training curves

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
plt.title("Training curves — Phase 2.1 (official splits)")
plt.tight_layout()
curves_path = Path(cfg.out_dir) / f"training_curves{cfg.suffix}.png"
plt.savefig(curves_path, bbox_inches="tight"); plt.show()
print(f"Saved {curves_path}")

# %% [markdown]
# ## 11. Per-noise-level eval on the official test split

# %% Per-level
ckpt = torch.load(best_ckpt_path, map_location=DEVICE)
model.load_state_dict(ckpt["model_state"]); model.eval()
print(f"Loaded best Phase 2.1 checkpoint from epoch {ckpt['epoch']}  "
      f"(val PSNR {ckpt['val_psnr']:.2f}).")

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
        with autocast(enabled=cfg.amp and DEVICE.type == "cuda"):
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
# ## 12. Final qualitative + persist

# %% Final qual + save
@torch.no_grad()
def final_qualitative():
    fig, axes = plt.subplots(len(NOISE_LEVELS), 3, figsize=(12, 4 * len(NOISE_LEVELS)))
    for i, level in enumerate(NOISE_LEVELS):
        ds = RFMiDDataset([test_paths[0]], cfg.image_size, cfg.patch_size,
                          augment=False, deterministic_seed=99, fixed_noise=level)
        noisy, clean, _ = ds[0]
        with autocast(enabled=cfg.amp and DEVICE.type == "cuda"):
            denoised = model(noisy.unsqueeze(0).to(DEVICE)).clamp(0.0, 1.0).squeeze(0).cpu()
        for ax, img, title in zip(axes[i], [noisy, denoised, clean],
                                  [f"noisy ({level})", "denoised", "clean"]):
            ax.imshow(img.permute(1, 2, 0).numpy()); ax.set_title(title); ax.axis("off")
    plt.tight_layout()
    out = Path(cfg.out_dir) / f"final_qualitative{cfg.suffix}.png"
    plt.savefig(out, bbox_inches="tight"); plt.show()
    return out

final_q = final_qualitative()
print(f"Saved {final_q}")

results = {
    "phase":          "phase2_1_euclidean_official_splits",
    "config":         asdict(cfg),
    "splits":         {"train": len(train_paths), "val": len(val_paths), "test": len(test_paths)},
    "best_epoch":     int(ckpt["epoch"]),
    "best_val_psnr":  float(ckpt["val_psnr"]),
    "best_val_ssim":  float(ckpt["val_ssim"]),
    "test_per_level": per_level,
    "noise_levels":   NOISE_LEVELS,
    "n_params":       int(n_params),
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
# ## Next step
#
# Once this finishes, the per-level table above becomes the new baseline.
# Then run `rfmid_unet_denoiser_hyperbolic_v2.ipynb` (Phase 3.1) which uses
# the **same splits** and adds the three Poincaré optimizations:
# block-moved-earlier, residual-skip, trainable-curvature.
