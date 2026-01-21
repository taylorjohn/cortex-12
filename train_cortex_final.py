# ============================================================
# train_cortex_final.py (PATCHED)
# Key fixes:
# - Correct shape supervision (no mislabeled composites)
# - Real contrastive learning: InfoNCE with positives + negatives
# - Augmentations via jitter + noise (cheap but effective)
# - Saves only trainable parts (adapter + heads)
# ============================================================
import os
import random
import ssl

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Optional insecure SSL (opt-in)
if os.getenv("ALLOW_INSECURE_SSL", "0") == "1":
    ssl._create_default_https_context = ssl._create_unverified_context

from vl_jepa_llm_v12 import OmniJEPA, draw_tensor, C_LIST, S_LIST, Z_LIST, SHAPE_PROPERTIES

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

CYCLES = 3000
LR = 5e-4

# InfoNCE params
NEG_K = 7
TEMP = 0.07

BRAIN_OUT = "brain_vector_v12.pth"

print(f"✅ Hardware: {DEVICE}")

def add_noise(x: torch.Tensor, sigma: float = 0.02) -> torch.Tensor:
    if sigma <= 0:
        return x
    n = torch.randn_like(x) * sigma
    return (x + n).clamp(0.0, 1.0)

def draw_aug(shape_name: str, color_name: str, size_name: str) -> torch.Tensor:
    # cheap augmentation: small spatial jitter + mild noise
    jitter = (64 + random.randint(-8, 8), 64 + random.randint(-8, 8))
    img = draw_tensor(shape_name, color_name, size_name, loc=jitter)
    img = add_noise(img, sigma=0.02)
    return img

def info_nce(z_anchor: torch.Tensor, z_pos: torch.Tensor, z_negs: torch.Tensor, t: float = 0.07) -> torch.Tensor:
    """
    z_anchor: [B,D]
    z_pos:    [B,D]
    z_negs:   [B,K,D]
    """
    z_anchor = F.normalize(z_anchor, dim=-1)
    z_pos = F.normalize(z_pos, dim=-1)
    z_negs = F.normalize(z_negs, dim=-1)

    pos = (z_anchor * z_pos).sum(-1, keepdim=True) / t          # [B,1]
    neg = (z_anchor.unsqueeze(1) * z_negs).sum(-1) / t          # [B,K]
    logits = torch.cat([pos, neg], dim=1)                       # [B,1+K]
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, labels)

def train():
    print("🧠 INITIALIZING CONTRASTIVE CORTEX TRAINING (InfoNCE)...")
    print("   (Device: " + str(DEVICE) + ")")

    model = OmniJEPA().to(DEVICE)
    model.train()

    # Only train trainable parts (backbone frozen already)
    optimizer = optim.Adam(
        list(model.adapter.parameters())
        + list(model.head_c.parameters())
        + list(model.head_s.parameters())
        + list(model.head_z.parameters())
        + list(model.logic_sides.parameters()),
        lr=LR
    )

    criterion_c = nn.CrossEntropyLoss()
    criterion_s = nn.CrossEntropyLoss()
    criterion_z = nn.CrossEntropyLoss()
    criterion_sides = nn.MSELoss()

    print(f"   ▶ Starting {CYCLES} cycles | NEG_K={NEG_K} | TEMP={TEMP}")

    for i in range(CYCLES):
        optimizer.zero_grad()

        # ----- Sample anchor concept -----
        c_name = random.choice(C_LIST)
        s_name = random.choice(S_LIST)
        z_name = random.choice(Z_LIST)

        # Anchor and positive (same concept under augmentation)
        img_anchor = draw_aug(s_name, c_name, z_name).unsqueeze(0).to(DEVICE)
        img_pos = draw_aug(s_name, c_name, z_name).unsqueeze(0).to(DEVICE)

        # Negatives: different shapes (keep color/size same to force shape discrimination)
        neg_shapes = [s for s in S_LIST if s != s_name]
        random.shuffle(neg_shapes)
        neg_shapes = neg_shapes[:NEG_K]
        neg_imgs = torch.stack([draw_aug(ns, c_name, z_name) for ns in neg_shapes], dim=0).to(DEVICE)  # [K,3,128,128]

        # ----- Forward -----
        feat_a, p_c, p_s, p_z, p_sides = model(img_anchor)  # feat_a: [B,128]
        feat_p, _, _, _, _ = model(img_pos)

        # negatives -> [K,128] then reshape [B,K,128]
        feat_negs = []
        for k in range(NEG_K):
            fk, _, _, _, _ = model(neg_imgs[k].unsqueeze(0))
            feat_negs.append(fk.squeeze(0))
        feat_negs = torch.stack(feat_negs, dim=0).unsqueeze(0)  # [1,K,128]

        # ----- Supervised targets (correct labels) -----
        target_c = torch.tensor([C_LIST.index(c_name)], device=DEVICE)
        target_s = torch.tensor([S_LIST.index(s_name)], device=DEVICE)
        target_z = torch.tensor([Z_LIST.index(z_name)], device=DEVICE)

        # sides regression target (weak supervision, but helps “complexity”)
        sides_target = torch.tensor([[SHAPE_PROPERTIES.get(s_name, {"sides": 0.0})["sides"]]], device=DEVICE, dtype=torch.float32)

        # ----- Losses -----
        loss_class = (
            criterion_c(p_c, target_c)
            + criterion_s(p_s, target_s)
            + criterion_z(p_z, target_z)
        )

        loss_sides = criterion_sides(p_sides, sides_target)

        loss_contrast = info_nce(feat_a, feat_p, feat_negs, t=TEMP)

        loss = loss_class + 0.25 * loss_sides + 1.0 * loss_contrast

        loss.backward()
        optimizer.step()

        if i % 250 == 0:
            with torch.no_grad():
                sim_ap = F.cosine_similarity(F.normalize(feat_a, dim=-1), F.normalize(feat_p, dim=-1)).item()
            print(f"   Step {i}/{CYCLES} | Loss: {loss.item():.4f} | InfoNCE: {loss_contrast.item():.4f} | A-P sim: {sim_ap:.3f}")

    print("✨ Training complete. Saving brain (trainable parts only)...")

    torch.save(
        {
            "adapter": model.adapter.state_dict(),
            "head_c": model.head_c.state_dict(),
            "head_s": model.head_s.state_dict(),
            "head_z": model.head_z.state_dict(),
            "logic_sides": model.logic_sides.state_dict(),
        },
        BRAIN_OUT
    )
    print(f"✅ BRAIN SAVED -> {BRAIN_OUT}")

if __name__ == "__main__":
    try:
        train()
    except KeyboardInterrupt:
        print("\n🛑 Training Interrupted.")