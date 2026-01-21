# ============================================================
# train_cortex_phase2_tinyimagenet.py
# ------------------------------------------------------------
# PHASE 2 TRAINER (Tiny-ImageNet-200) — UPDATED (Windows-safe)
#
# Trains your VL-JEPA "cortex adapter space" using TWO streams:
#
# (A) Synthetic stream (your renderer):
#     - supervised heads: color / shape / size (+ sides regression)
#     - explicit-negative InfoNCE: (anchor, positive, K negatives)
#
# (B) Tiny-ImageNet stream (real images, unlabeled self-supervision):
#     - SimCLR NT-Xent with in-batch negatives
#     - uses TWO augmented views per image
#
# Backbone:
#   - Frozen DINOv2 (inside OmniJEPA in vl_jepa_llm_v12.py)
#
# Saves:
#   - periodic checkpoints: <save_dir>/cortex_stepXXXXX.pt
#   - final checkpoint:     <save_dir>/cortex_final.pt
#
# Usage (PowerShell example):
#   python train_cortex_phase2_tinyimagenet.py `
#     --tiny_root .\datasets\tiny-imagenet-200 `
#     --steps 12000 `
#     --batch_real 64 `
#     --batch_synth 32 `
#     --threads 24 `
#     --interop 2 `
#     --save_dir runs\phase2_tiny `
#     --save_every 200
#
# Notes:
#   - Only uses tiny-imagenet-200/train/*/images/*.JPEG
#   - No torchvision dependency required.
#   - num_workers defaults to 0 for maximum Windows stability.
# ============================================================

import os
import ssl
import math
import glob
import random
import argparse
from typing import Optional, List

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ----------------------------
# Optional insecure SSL (OFF by default)
# ----------------------------
if os.getenv("ALLOW_INSECURE_SSL", "0") == "1":
    ssl._create_default_https_context = ssl._create_unverified_context

# ----------------------------
# Import your project core
# ----------------------------
from vl_jepa_llm_v12 import (
    OmniJEPA,
    draw_tensor,
    C_LIST, S_LIST, Z_LIST,
    SHAPE_PROPERTIES,
    IMG_RES,  # should be 128 in your setup; trainer will match it
)

# ----------------------------
# Device
# ----------------------------
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# ----------------------------
# Tiny-ImageNet Dataset (train split)
# ----------------------------
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPEG", ".JPG", ".PNG")


class TinyImageNetTrain(Dataset):
    """
    Loads tiny-imagenet-200/train/*/images/*.JPEG
    Returns PIL images only (unlabeled for SimCLR).
    """
    def __init__(self, root: str, max_files: Optional[int] = None):
        self.root = root
        train_dir = os.path.join(root, "train")
        if not os.path.isdir(train_dir):
            raise FileNotFoundError(f"Missing train/ folder at: {train_dir}")

        # Use wnids.txt if present for deterministic ordering
        wnids_path = os.path.join(root, "wnids.txt")
        if os.path.isfile(wnids_path):
            with open(wnids_path, "r", encoding="utf-8") as f:
                wnids = [line.strip() for line in f if line.strip()]
        else:
            wnids = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])

        items: List[str] = []
        for wnid in wnids:
            img_dir = os.path.join(train_dir, wnid, "images")
            if not os.path.isdir(img_dir):
                continue
            for ext in IMG_EXTS:
                items.extend(glob.glob(os.path.join(img_dir, f"*{ext}")))

        items = [p for p in items if os.path.isfile(p)]
        random.shuffle(items)

        if max_files is not None:
            items = items[:max_files]

        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        path = self.items[idx]
        img = Image.open(path).convert("RGB")
        return img


# ----------------------------
# Augmentations (no torchvision)
# ----------------------------
def pil_to_tensor01(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # [3,H,W]


def random_resized_crop(img: Image.Image, out_size: int) -> Image.Image:
    w, h = img.size
    if w < 16 or h < 16:
        return img.resize((out_size, out_size), Image.BILINEAR)

    scale = random.uniform(0.5, 1.0)
    new_w = max(8, int(w * scale))
    new_h = max(8, int(h * scale))
    x0 = random.randint(0, max(0, w - new_w))
    y0 = random.randint(0, max(0, h - new_h))
    crop = img.crop((x0, y0, x0 + new_w, y0 + new_h))
    return crop.resize((out_size, out_size), Image.BILINEAR)


def color_jitter_tensor(x: torch.Tensor, strength: float = 0.25) -> torch.Tensor:
    if strength <= 0:
        return x
    b = random.uniform(1 - strength, 1 + strength)
    c = random.uniform(1 - strength, 1 + strength)
    s = random.uniform(1 - strength, 1 + strength)

    x = (x * b).clamp(0, 1)  # brightness

    mean = x.mean(dim=(1, 2), keepdim=True)
    x = ((x - mean) * c + mean).clamp(0, 1)  # contrast

    gray = (0.299 * x[0] + 0.587 * x[1] + 0.114 * x[2]).unsqueeze(0)
    x = (x * s + gray * (1 - s)).clamp(0, 1)  # saturation
    return x


def random_gray(x: torch.Tensor, p: float = 0.15) -> torch.Tensor:
    if random.random() > p:
        return x
    gray = (0.299 * x[0] + 0.587 * x[1] + 0.114 * x[2]).unsqueeze(0)
    return gray.repeat(3, 1, 1)


def gaussian_noise(x: torch.Tensor, sigma: float = 0.01) -> torch.Tensor:
    if sigma <= 0:
        return x
    return (x + torch.randn_like(x) * sigma).clamp(0, 1)


def real_augment(img: Image.Image, out_size: int) -> torch.Tensor:
    img = random_resized_crop(img, out_size=out_size)
    t = pil_to_tensor01(img)
    t = color_jitter_tensor(t, strength=0.25)
    t = random_gray(t, p=0.15)
    t = gaussian_noise(t, sigma=0.01)
    return t


def synth_augment(shape: str, color: str, size: str, jitter: int = 8, noise: float = 0.02) -> torch.Tensor:
    loc = (64 + random.randint(-jitter, jitter), 64 + random.randint(-jitter, jitter))
    x = draw_tensor(shape, color, size, loc=loc)
    if noise > 0:
        x = (x + torch.randn_like(x) * noise).clamp(0, 1)
    return x


# ----------------------------
# Contrastive losses
# ----------------------------
def info_nce_pairwise(z_a: torch.Tensor, z_p: torch.Tensor, z_negs: torch.Tensor, t: float) -> torch.Tensor:
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


def nt_xent_inbatch(z1: torch.Tensor, z2: torch.Tensor, t: float) -> torch.Tensor:
    """
    SimCLR NT-Xent with in-batch negatives.
    z1,z2: [B,D]
    """
    B = z1.size(0)
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)

    reps = torch.cat([z1, z2], dim=0)      # [2B,D]
    sim = reps @ reps.t()                  # [2B,2B]

    mask = torch.eye(2 * B, device=sim.device, dtype=torch.bool)
    sim = sim.masked_fill(mask, -1e9)

    # positives: i->i+B, i+B->i
    labels = torch.cat([
        torch.arange(B, 2 * B, device=sim.device),
        torch.arange(0, B, device=sim.device)
    ], dim=0)

    logits = sim / t
    return F.cross_entropy(logits, labels)


# ----------------------------
# Collate (keeps PIL images)
# ----------------------------
def pil_collate(batch):
    return batch  # list[Image.Image]


# ----------------------------
# Checkpoint helpers
# ----------------------------
def pack_cortex_state(model: OmniJEPA) -> dict:
    return {
        "adapter": model.adapter.state_dict(),
        "head_c": model.head_c.state_dict(),
        "head_s": model.head_s.state_dict(),
        "head_z": model.head_z.state_dict(),
        "logic_sides": model.logic_sides.state_dict(),
    }


def save_checkpoint(model: OmniJEPA, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(pack_cortex_state(model), path)


# ----------------------------
# Main train
# ----------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--tiny_root", type=str, required=True, help="Path to tiny-imagenet-200")
    ap.add_argument("--max_real", type=int, default=200000, help="Max images to index from tiny train/")
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--lr", type=float, default=5e-4)

    ap.add_argument("--batch_real", type=int, default=64)
    ap.add_argument("--batch_synth", type=int, default=32)
    ap.add_argument("--neg_k", type=int, default=7)

    ap.add_argument("--temp_synth", type=float, default=0.07)
    ap.add_argument("--temp_real", type=float, default=0.2)

    ap.add_argument("--lambda_real", type=float, default=1.0)
    ap.add_argument("--lambda_synth_contrast", type=float, default=1.0)
    ap.add_argument("--lambda_sides", type=float, default=0.25)

    # NEW: save_dir + cadence (instead of "save is a file")
    ap.add_argument("--save_dir", type=str, default="runs/phase2_tiny", help="Directory for checkpoints")
    ap.add_argument("--save_every", type=int, default=200, help="Checkpoint every N steps (0 disables)")
    ap.add_argument("--log_every", type=int, default=200, help="Log every N steps")

    # Windows stability
    ap.add_argument("--num_workers", type=int, default=0, help="DataLoader workers (Windows: 0-4)")

    ap.add_argument("--threads", type=int, default=int(os.getenv("OMP_NUM_THREADS", "24")))
    ap.add_argument("--interop", type=int, default=2)

    args = ap.parse_args()

    # CPU tuning
    try:
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(args.interop)
        torch.backends.mkldnn.enabled = True
    except Exception:
        pass

    print(f"✅ Device: {DEVICE}")
    if DEVICE.type == "cpu":
        print(f"✅ Threads: {torch.get_num_threads()}  Interop: {torch.get_num_interop_threads()}")

    # Prepare save dir immediately (so it exists even before first save)
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"💾 Save dir: {args.save_dir}")

    # Load Tiny-ImageNet train images
    ds = TinyImageNetTrain(args.tiny_root, max_files=args.max_real)
    if len(ds) == 0:
        raise RuntimeError("Indexed 0 Tiny-ImageNet images. Check path / structure.")

    real_loader = DataLoader(
        ds,
        batch_size=args.batch_real,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        collate_fn=pil_collate,
        pin_memory=False,
        persistent_workers=(args.num_workers > 0),
    )
    real_iter = iter(real_loader)
    print(f"📸 Tiny-ImageNet train images indexed: {len(ds)}")

    # Model
    model = OmniJEPA().to(DEVICE)
    model.train()

    # Trainable params only (adapter + heads + sides head)
    optimizer = optim.Adam(
        list(model.adapter.parameters())
        + list(model.head_c.parameters())
        + list(model.head_s.parameters())
        + list(model.head_z.parameters())
        + list(model.logic_sides.parameters()),
        lr=args.lr
    )

    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()

    def forward_feats(x: torch.Tensor):
        feat, pc, ps, pz, sides = model(x)
        return feat, pc, ps, pz, sides

    def safe_next_real_batch():
        nonlocal real_iter
        try:
            return next(real_iter)
        except StopIteration:
            real_iter = iter(real_loader)
            return next(real_iter)

    # ----------------------------
    # Training loop
    # ----------------------------
    try:
        for step in range(args.steps):
            optimizer.zero_grad()

            # ===== A) SYNTHETIC BATCH =====
            synth_imgs_a = []
            synth_imgs_p = []
            t_color = []
            t_shape = []
            t_size = []
            t_sides = []
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
                t_sides.append(float(SHAPE_PROPERTIES.get(s, {"sides": 0.0})["sides"]))

                # Negatives: same color/size, different shape
                neg_shapes = [ss for ss in S_LIST if ss != s]
                random.shuffle(neg_shapes)
                neg_shapes = neg_shapes[:args.neg_k]
                neg_imgs.append(torch.stack([synth_augment(ns, c, z) for ns in neg_shapes], dim=0))  # [K,3,H,W]

            x_a = torch.stack(synth_imgs_a, dim=0).to(DEVICE)                 # [B,3,H,W]
            x_p = torch.stack(synth_imgs_p, dim=0).to(DEVICE)
            x_neg = torch.stack(neg_imgs, dim=0).to(DEVICE)                   # [B,K,3,H,W]

            feat_a, pc, ps, pz, sides = forward_feats(x_a)
            feat_p, _, _, _, _ = forward_feats(x_p)

            # Embed negatives efficiently by flattening
            B, K = x_neg.size(0), x_neg.size(1)
            feat_neg, _, _, _, _ = forward_feats(x_neg.view(B * K, 3, IMG_RES, IMG_RES))
            feat_neg = feat_neg.view(B, K, -1)

            y_c = torch.tensor(t_color, device=DEVICE)
            y_s = torch.tensor(t_shape, device=DEVICE)
            y_z = torch.tensor(t_size, device=DEVICE)
            y_sides = torch.tensor(t_sides, device=DEVICE, dtype=torch.float32).view(-1, 1)

            loss_sup = ce(pc, y_c) + ce(ps, y_s) + ce(pz, y_z)
            loss_sides = mse(sides, y_sides)
            loss_synth = info_nce_pairwise(feat_a, feat_p, feat_neg, t=args.temp_synth)

            # ===== B) REAL (Tiny-ImageNet) BATCH =====
            imgs_pil = safe_next_real_batch()

            v1 = torch.stack([real_augment(img, out_size=IMG_RES) for img in imgs_pil], dim=0).to(DEVICE)
            v2 = torch.stack([real_augment(img, out_size=IMG_RES) for img in imgs_pil], dim=0).to(DEVICE)

            z1, _, _, _, _ = forward_feats(v1)
            z2, _, _, _, _ = forward_feats(v2)
            loss_real = nt_xent_inbatch(z1, z2, t=args.temp_real)

            # ===== TOTAL =====
            loss = (
                loss_sup
                + args.lambda_sides * loss_sides
                + args.lambda_synth_contrast * loss_synth
                + args.lambda_real * loss_real
            )

            loss.backward()
            optimizer.step()

            # ===== Logging =====
            if step % args.log_every == 0:
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
                    f"synthNCE={loss_synth.item():.4f} "
                    f"realNTX={loss_real.item():.4f} "
                    f"A-Psim={ap_sim:.3f}"
                )

            # ===== Periodic checkpoint =====
            if args.save_every > 0 and step > 0 and (step % args.save_every == 0):
                ckpt_path = os.path.join(args.save_dir, f"cortex_step{step:05d}.pt")
                save_checkpoint(model, ckpt_path)
                print(f"💾 Checkpoint saved → {ckpt_path}")

    except KeyboardInterrupt:
        # Graceful save on Ctrl+C
        print("\n🛑 KeyboardInterrupt detected — saving interrupt checkpoint...")
        ckpt_path = os.path.join(args.save_dir, "cortex_interrupt.pt")
        save_checkpoint(model, ckpt_path)
        print(f"✅ Saved → {ckpt_path}")
        return

    # Final save
    print("💾 Saving final cortex (adapter+heads only)...")
    final_path = os.path.join(args.save_dir, "cortex_final.pt")
    save_checkpoint(model, final_path)
    print(f"✅ Saved → {final_path}")


if __name__ == "__main__":
    main()
