"""
Phase 3 Curriculum Trainer for CORTEX-12 — LARGE-SCALE TRAINING OPTIMIZED
- Enhanced loss weights for color/shape/orientation
- Cosine annealing with lower minimum LR
- Resumes from best checkpoint
- CPU-only, deterministic
"""

import os
import json
import random
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.hub
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from cortex_adapter_v12 import CortexAdapter


def collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    labels = [item[1] for item in batch]
    return images, labels


class CurriculumDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        # Use the merged labels file
        labels_path = os.path.join(data_dir, "labels_merged.json")
        if not os.path.exists(labels_path):
            labels_path = os.path.join(data_dir, "labels.json")  # Fallback
        with open(labels_path, 'r') as f:
            self.labels = json.load(f)
        self.filenames = list(self.labels.keys())

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        img = Image.open(os.path.join(self.data_dir, "images", fname)).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[fname]


def contrastive_axis_loss(embeddings, labels_list, axis_key, axis_dims, label_map):
    start, end = axis_dims
    sub_emb = embeddings[:, start:end+1]
    sub_emb = F.normalize(sub_emb, dim=1)
    
    label_ids = []
    for lbl in labels_list:
        val = lbl[axis_key]
        if val not in label_map:
            raise KeyError(f"Value '{val}' not in label_map for axis '{axis_key}'. Available: {list(label_map.keys())}")
        label_ids.append(label_map[val])
    label_ids = torch.tensor(label_ids, dtype=torch.long)
    
    sim = torch.mm(sub_emb, sub_emb.t())
    same = (label_ids.unsqueeze(0) == label_ids.unsqueeze(1)).float()
    eps = 1e-8
    pos_sim = torch.exp(sim * same)
    neg_sim = torch.exp(sim * (1 - same))
    denominator = pos_sim + neg_sim.sum(dim=1, keepdim=True) + eps
    loss = -torch.log((pos_sim / denominator).sum(dim=1) + eps).mean()
    return loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/curriculum")
    parser.add_argument("--output_dir", type=str, default="runs/phase3")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--resume_from", type=str, default="runs/phase3/cortex_step_phase3_0050.pt")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    AXIS_LAYOUT = {
        "shape": (0, 31),
        "size": (32, 47),
        "material": (48, 63),
        "color": (64, 79),
        "location": (80, 87),
        "orientation": (88, 103)
    }

    # Load labels and build SAFE label maps
    with open(os.path.join(args.data_dir, "labels.json"), 'r') as f:
        all_labels = json.load(f)
    
    label_maps = {}
    for axis in AXIS_LAYOUT:
        if axis in next(iter(all_labels.values())):
            sample_val = next(iter(all_labels.values()))[axis]
            if isinstance(sample_val, (list, tuple)):
                label_maps[axis] = None  # continuous
            else:
                values = sorted({lbl[axis] for lbl in all_labels.values()})
                label_maps[axis] = {v: i for i, v in enumerate(values)}
        else:
            label_maps[axis] = None

    print("✅ Label maps built:")
    for axis, lm in label_maps.items():
        if lm is not None:
            print(f"  {axis}: {len(lm)} classes")
        else:
            print(f"  {axis}: (continuous)")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = CurriculumDataset("data/balanced_images", transform=transform)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=0,
        collate_fn=collate_fn
    )

    device = torch.device("cpu")

    print("Loading DINOv2 ViT-S/14 backbone...")
    dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    dinov2.eval()
    dinov2.to(device)
    for p in dinov2.parameters():
        p.requires_grad = False

    model = CortexAdapter()
    
    # Resume from best checkpoint
    if os.path.exists(args.resume_from):
        print(f"Resuming from checkpoint: {args.resume_from}")
        ckpt = torch.load(args.resume_from, map_location=device, weights_only=True)
        model.load_state_dict(ckpt['cortex_state_dict'], strict=False)
    
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=args.epochs,
        eta_min=1e-6  # Lower minimum learning rate
    )

    LOSS_WEIGHTS = {
        "shape": 1.0,
        "size": 0.8,
        "material": 1.0,
        "color": 1.0,
        "orientation": 0.8,  # Increased from 0.6
        "location": 0.0
    }

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        for batch_idx, (images, labels_list) in enumerate(dataloader):
            images = images.to(device)
            
            with torch.no_grad():
                features = dinov2(images)
            embeddings = model(features)

            loss = 0.0
            for axis, dims in AXIS_LAYOUT.items():
                if label_maps[axis] is not None:
                    try:
                        axis_loss = contrastive_axis_loss(
                            embeddings, labels_list, axis, dims, label_maps[axis]
                        )
                        loss += LOSS_WEIGHTS[axis] * axis_loss
                    except KeyError as e:
                        print(f"Skipping axis '{axis}' due to error: {e}")
                        continue

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{args.epochs} | Avg Loss: {avg_loss:.4f} | LR: {current_lr:.2e}")
        scheduler.step()

        if (epoch + 1) % 25 == 0 or epoch == args.epochs - 1:
            save_path = os.path.join(args.output_dir, f"cortex_step_phase3_{epoch+1:04d}.pt")
            torch.save({'cortex_state_dict': model.state_dict()}, save_path)
            print(f"Saved checkpoint: {save_path}")

    print("✅ Large-scale training complete!")


if __name__ == "__main__":
    main()