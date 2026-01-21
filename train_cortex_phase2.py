# ============================================================
# train_cortex_phase2.py  (PHASE 2 TRAINER)
# ------------------------------------------------------------
# What this does:
#   Phase 1 (synthetic): your existing synthetic curriculum
#     - supervised heads (color/shape/size + sides)
#     - contrastive InfoNCE with explicit negatives
#
#   Phase 2 (real images): adds *unlabeled* real-image self-supervision
#     - SimCLR-style NT-Xent (in-batch negatives)
#     - trains the SAME adapter space, so your cortex becomes robust to
#       textures, lighting, backgrounds, and real-world structure.
#
# Backbone:
#   Frozen DINOv2 (torch.hub)
#
# Output:
#   brain_vector_v12.pth (adapter + heads only)
#
# Expected real data layout (unlabeled):
#   REAL_DIR/
#      img1.jpg
#      img2.png
#      ...
#   or nested subfolders (any depth):
#      REAL_DIR/classA/*.jpg
#      REAL_DIR/classB/*.png
#
# Run on AMD CPU (recommended):
#   export OMP_NUM_THREADS=24 MKL_NUM_THREADS=24 OMP_PROC_BIND=true OMP_PLACES=cores
#   python3 train_cortex_phase2.py --real_dir /path/to/REAL_DIR --steps 8000 --batch_real 64
#
# If you have no real images:
#   it will fall back to synthetic-only training.
# ============================================================

import os
import ssl
import math
import glob
import random
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Optional insecure SSL for torch.hub (OFF by default)
if os.getenv("ALLOW_INSECURE_SSL", "0") == "1":
    ssl._create_default_https_context = ssl._create_unverified_context

# Import your core model + renderer + vocab
from vl_jepa_llm_v12 import (
    OmniJEPA,
    draw_tensor,
    C_LIST, S_LIST, Z_LIST,
    SHAPE_PROPERTIES,
)

# ----------------------------
# Device / CPU performance
# ----------------------------
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

def set_threads(threads: int, interop: int = 2):
    try:
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(interop)
    except Exception:
        pass

# ----------------------------
# Real image dataset (unlabeled)
# ----------------------------
IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

class RealImageDataset(Dataset):
    def __init__(self, root: str, max_files: Optional[int] = None):
        self.root = root
        files = []
        for ext in IMG_EXTS:
            files.extend(glob.glob(os.path.join(root, "**", f"*{ext}"), recursive=True))
        files = [f for f in files if os.path.isfile(f)]
        random.shuffle(files)
        if max_files is not None:
            files = files[:max_files]
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        img = Image.open(path).convert("RGB")
        return img

# ----------------------------
# Augmentations for real images (PIL -> torch)
# ----------------------------
def pil_to_tensor01(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    return t

def random_resized_crop(img: Image.Image, out_size: int = 128) -> Image.Image:
    # Simple manual crop (no torchvision dependency)
    w, h = img.size
    if w < 16 or h < 16:
        return img.resize((out_size, out_size), Image.BILINEAR)

    # scale range like SimCLR-ish
    scale = random.uniform(0.5, 1.0)
    new_w = max(8, int(w * scale))
    new_h = max(8, int(h * scale))

    x0 = random.randint(0, max(0, w - new_w))
    y0 = random.randint(0, max(0, h - new_h))
    crop = img.crop((x0, y0, x0 + new_w, y0 + new_h))
    return crop.resize((out_size, out_size), Image.BILINEAR)

def color_jitter_tensor(x: torch.Tensor, strength: float = 0.25) -> torch.Tensor:
    # x in [0,1], [3,H,W]
    # apply simple brightness/contrast/saturation jitter
    if strength <= 0:
        return x
    b = random.uniform(1 - strength, 1 + strength)
    c = random.uniform(1 - strength, 1 + strength)
    s = random.uniform(1 - strength, 1 + strength)

    # brightness
    x = (x * b).clamp(0, 1)

    # contrast: center around mean
    mean = x.mean(dim=(1,2), keepdim=True)
    x = ((x - mean) * c + mean).clamp(0, 1)

    # saturation: blend with grayscale
    gray = (0.299 * x[0] + 0.587 * x[1] + 0.114 * x[2]).unsqueeze(0)
    x = (x * s + gray * (1 - s)).clamp(0, 1)
    return x

def random_gray(x: torch.Tensor, p: float = 0.2) -> torch.Tensor:
    if random.random() > p:
        return x
    gray = (0.299 * x[0] + 0.587 * x[1] + 0.114 * x[2]).unsqueeze(0)
    return gray.repeat(3, 1, 1)

def gaussian_noise(x: torch.Tensor, sigma: float = 0.02) -> torch.Tensor:
    if sigma <= 0:
        return x
    return (x + torch.randn_like(x) * sigma).clamp(0, 1)

def real_augment(img: Image.Image, out_size: int = 128) -> torch.Tensor:
    img = random_resized_crop(img, out_size=out_size)
    t = pil_to_tensor01(img)
    t = color_jitter_tensor(t, strength=0.25)
    t = random_gray(t, p=0.15)
    t = gaussian_noise(t, sigma=0.01)
    return t

# ----------------------------
# Synthetic augment
# ----------------------------
def synth_augment(shape: str, color: str, size: str, jitter: int = 8, noise: float = 0.02) -> torch.Tensor:
    loc = (64 + random.randint(-jitter, jitter), 64 + random.randint(-jitter, jitter))
    x = draw_tensor(shape, color, size, loc=loc)
    if noise > 0:
        x = (x + torch.randn_like(x) * noise).clamp(0, 1)
    return x

# ----------------------------
# Contrastive losses
# ----------------------------
def info_nce_pairwise(z_a: torch.Tensor, z_p: torch.Tensor, z_negs: torch.Tensor, t: float = 0.07) -> torch.Tensor:
    """
    z_a: [B,D]
    z_p: [B,D]
    z_negs: [B,K,D]
    """
    z_a = F.normalize(z_a, dim=-1)
    z_p = F.normalize(z_p, dim=-1)
    z_negs = F.normalize(z_negs, dim=-1)

    pos = (z_a * z_p).sum(-1, keepdim=True) / t
    neg = (z_a.unsqueeze(1) * z_negs).sum(-1) / t
    logits = torch.cat([pos, neg], dim=1)
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, labels)

def nt_xent_inbatch(z1: torch.Tensor, z2: torch.Tensor, t: float = 0.2) -> torch.Tensor:
    """
    SimCLR NT-Xent with in-batch negatives.
    z1,z2: [B,D]
    """
    B = z1.size(0)
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)

    reps = torch.cat([z1, z2], dim=0)  # [2B,D]
    sim = reps @ reps.t()              # [2B,2B]

    # remove self-similarity
    mask = torch.eye(2 * B, device=sim.device, dtype=torch.bool)
    sim = sim.masked_fill(mask, -1e9)

    # positives: (i, i+B) and (i+B, i)
    pos = torch.cat([torch.arange(B, 2*B, device=sim.device),
                     torch.arange(0, B, device=sim.device)], dim=0)
    logits = sim / t
    labels = pos
    return F.cross_entropy(logits, labels)

# ----------------------------
# Training
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_dir", type=str, default="", help="Folder with real images (optional)")
    ap.add_argument("--max_real", type=int, default=50000, help="Max real images to index")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--lr", type=float, default=5e-4)

    ap.add_argument("--batch_real", type=int, default=64)
    ap.add_argument("--batch_synth", type=int, default=32)
    ap.add_argument("--neg_k", type=int, default=7)

    ap.add_argument("--temp_synth", type=float, default=0.07)
    ap.add_argument("--temp_real", type=float, default=0.2)

    ap.add_argument("--lambda_real", type=float, default=1.0, help="weight for real NT-Xent loss")
    ap.add_argument("--lambda_synth_contrast", type=float, default=1.0)
    ap.add_argument("--lambda_sides", type=float, default=0.25)

    ap.add_argument("--save", type=str, default="brain_vector_v12.pth")
    ap.add_argument("--threads", type=int, default=int(os.getenv("OMP_NUM_THREADS", "24")))
    ap.add_argument("--interop", type=int, default=2)
    args = ap.parse_args()

    set_threads(args.threads, args.interop)
    try:
        torch.backends.mkldnn.enabled = True
    except Exception:
        pass

    print(f"✅ Device: {DEVICE}")
    if DEVICE.type == "cpu":
        print(f"✅ Threads: {torch.get_num_threads()}  Interop: {torch.get_num_interop_threads()}")

    # model
    model = OmniJEPA().to(DEVICE)
    model.train()

    # optimizer (train adapter + heads; backbone already frozen in OmniJEPA)
    optimizer = optim.Adam(
        list(model.adapter.parameters())
        + list(model.head_c.parameters())
        + list(model.head_s.parameters())
        + list(model.head_z.parameters())
        + list(model.logic_sides.parameters()),
        lr=args.lr
    )

    # supervised losses for synthetic
    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()

    # real dataset (optional)
    real_loader = None
    real_iter = None
    if args.real_dir and os.path.isdir(args.real_dir):
        ds = RealImageDataset(args.real_dir, max_files=args.max_real)
        if len(ds) > 0:
            # DataLoader returns PIL images; we augment in training loop
            real_loader = DataLoader(ds, batch_size=args.batch_real, shuffle=True, num_workers=0, drop_last=True)
            real_iter = iter(real_loader)
            print(f"📸 Real images indexed: {len(ds)} from {args.real_dir}")
        else:
            print(f"⚠️ No real images found under: {args.real_dir}")
    else:
        if args.real_dir:
            print(f"⚠️ real_dir not found: {args.real_dir}")
        print("📌 Running synthetic-only (Phase 1) because no real data supplied.")

    def forward_feats(x: torch.Tensor):
        feat, pc, ps, pz, sides = model(x)
        return feat, pc, ps, pz, sides

    # training loop
    for step in range(args.steps):
        optimizer.zero_grad()

        # ----------------------------
        # A) SYNTHETIC BATCH (Phase 1)
        # ----------------------------
        # Build a small batch of synthetic anchors/positives
        synth_imgs_a = []
        synth_imgs_p = []
        t_color = []
        t_shape = []
        t_size = []
        t_sides = []

        # For pairwise InfoNCE, we also build per-sample negative sets (different shapes)
        neg_imgs = []

        for _ in range(args.batch_synth):
            c = random.choice(C_LIST)
            s = random.choice(S_LIST)
            z = random.choice(Z_LIST)

            synth_imgs_a.append(synth_augment(s, c, z))
            synth_imgs_p.append(synth_augment(s, c, z))

            t_color.append(C_LIST.index(c))
            t_shape.append(S_LIST.index(s))
            t_size.append(Z_LIST.index(z))
            t_sides.append(SHAPE_PROPERTIES.get(s, {"sides": 0.0})["sides"])

            # negatives: same color/size, different shapes (forces shape discrimination)
            neg_shapes = [ss for ss in S_LIST if ss != s]
            random.shuffle(neg_shapes)
            neg_shapes = neg_shapes[:args.neg_k]
            neg_imgs.append(torch.stack([synth_augment(ns, c, z) for ns in neg_shapes], dim=0))  # [K,3,128,128]

        x_a = torch.stack(synth_imgs_a, dim=0).to(DEVICE)  # [B,3,128,128]
        x_p = torch.stack(synth_imgs_p, dim=0).to(DEVICE)
        x_neg = torch.stack(neg_imgs, dim=0).to(DEVICE)    # [B,K,3,128,128]

        feat_a, pc, ps, pz, sides = forward_feats(x_a)
        feat_p, _, _, _, _ = forward_feats(x_p)

        # embed negatives efficiently: flatten [B*K,3,H,W]
        B, K = x_neg.size(0), x_neg.size(1)
        feat_neg, _, _, _, _ = forward_feats(x_neg.view(B * K, 3, IMG_RES, IMG_RES))
        feat_neg = feat_neg.view(B, K, -1)

        y_c = torch.tensor(t_color, device=DEVICE)
        y_s = torch.tensor(t_shape, device=DEVICE)
        y_z = torch.tensor(t_size, device=DEVICE)
        y_sides = torch.tensor(t_sides, device=DEVICE, dtype=torch.float32).view(-1, 1)

        loss_sup = ce(pc, y_c) + ce(ps, y_s) + ce(pz, y_z)
        loss_sides = mse(sides, y_sides)
        loss_synth_contrast = info_nce_pairwise(feat_a, feat_p, feat_neg, t=args.temp_synth)

        loss = loss_sup + args.lambda_sides * loss_sides + args.lambda_synth_contrast * loss_synth_contrast

        # ----------------------------
        # B) REAL IMAGE BATCH (Phase 2)
        # ----------------------------
        loss_real = torch.tensor(0.0, device=DEVICE)
        if real_loader is not None:
            try:
                imgs_pil = next(real_iter)
            except StopIteration:
                real_iter = iter(real_loader)
                imgs_pil = next(real_iter)

            # Create two augmented views per image
            v1 = torch.stack([real_augment(img, out_size=IMG_RES) for img in imgs_pil], dim=0).to(DEVICE)
            v2 = torch.stack([real_augment(img, out_size=IMG_RES) for img in imgs_pil], dim=0).to(DEVICE)

            z1, _, _, _, _ = forward_feats(v1)
            z2, _, _, _, _ = forward_feats(v2)

            loss_real = nt_xent_inbatch(z1, z2, t=args.temp_real)
            loss = loss + args.lambda_real * loss_real

        loss.backward()
        optimizer.step()

        # ----------------------------
        # Logging
        # ----------------------------
        if step % 200 == 0:
            with torch.no_grad():
                ap_sim = F.cosine_similarity(
                    F.normalize(feat_a, dim=-1),
                    F.normalize(feat_p, dim=-1),
                    dim=-1
                ).mean().item()

            print(
                f"[{step:05d}/{args.steps}] "
                f"loss={loss.item():.4f} "
                f"sup={loss_sup.item():.4f} "
                f"synthNCE={loss_synth_contrast.item():.4f} "
                f"realNTX={loss_real.item():.4f} "
                f"A-Psim={ap_sim:.3f}"
            )

    # Save only trainable parts
    print("💾 Saving cortex (adapter+heads only)...")
    torch.save(
        {
            "adapter": model.adapter.state_dict(),
            "head_c": model.head_c.state_dict(),
            "head_s": model.head_s.state_dict(),
            "head_z": model.head_z.state_dict(),
            "logic_sides": model.logic_sides.state_dict(),
        },
        args.save
    )
    print(f"✅ Saved → {args.save}")

if __name__ == "__main__":
    main()