# Training — CORTEX-12 Phase-2

This document describes Phase-2 training, which aligns compact visual
representations with real images using Tiny-ImageNet.

---

## Objectives

Phase-2 training is designed to:

- Preserve semantic axes learned from synthetic data
- Improve robustness to real-world images
- Avoid representation collapse
- Maintain stability over long CPU-only runs

---

## Training Setup

- Backbone: Frozen DINOv2 ViT
- Trainable components: Adapter + heads only
- Dataset: Tiny-ImageNet (train split)
- Compute: CPU-only

---

## Loss Components

| Term | Description |
|-----|-------------|
| sup | Supervised synthetic alignment loss |
| synthNCE | Contrastive loss on rendered objects |
| realNTX | NT-Xent loss on real images |
| A-Psim | Diagnostic perceptual similarity (monitor only) |

---

## Expected Behavior

### Early Training
- Rapid decrease in supervised loss
- High similarity clustering

### Mid Training
- Oscillating total loss
- Stable SAME > DIFF separation

### Late Training
- Loss plateaus
- No collapse
- Stable embeddings across checkpoints

---

## Runtime Characteristics

- ~300–330 steps/hour on AMD CPU
- ~34–40 hours for 12k steps
- Safe to run unattended
- Low disk and memory pressure

---

## Checkpoints

Saved periodically as:
- cortex_stepXXXXX.pt
- cortex_final.pt

All checkpoints are compatible with the runtime.
