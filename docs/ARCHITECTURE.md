# Architecture — CORTEX-12

CORTEX-12 is structured as a compact visual cortex composed of a frozen
perceptual backbone and a small, interpretable latent representation layer.

The architecture explicitly separates **perception**, **representation**,
and **memory**.

---

## High-Level Architecture

RGB Image (128×128)
│
▼
Frozen DINOv2 Vision Transformer
│
▼
Adapter MLP
│
▼
128-D Latent Vector
│
├─ Color Head
├─ Shape Head
├─ Size Head
└─ Logic Head

Only the **adapter and heads** are trainable.
The backbone remains frozen at all times.

---

## Frozen Backbone

CORTEX-12 uses a pretrained DINOv2 Vision Transformer as a fixed feature
extractor.

### Rationale

- Preserves general visual knowledge
- Prevents catastrophic forgetting
- Stabilizes long CPU-only training runs
- Ensures representations remain comparable across checkpoints

The backbone is never fine-tuned.

---

## Adapter Layer

The adapter is a lightweight MLP that projects high-dimensional backbone
features into a compact latent space.

- Input: frozen backbone features
- Output: 128-dimensional latent vector
- Trainable: yes

This is the **only location where representation learning occurs**.

---

## Latent Space

The 128-dimensional latent vector is the core representational object in
CORTEX-12.

All reasoning and comparison reduces to geometric operations such as:

cosine_similarity(v1, v2)

The latent space is designed to be:
- Stable
- Low-dimensional
- Interpretable
- Comparable across time

---

## Semantic Heads

Each head provides a structured projection of the latent vector.

| Head  | Purpose |
|------|---------|
| Color | Encodes semantic color information |
| Shape | Encodes geometric identity |
| Size  | Encodes relative scale |
| Logic | Encodes continuous complexity signal |

Heads are small, linear or shallow modules attached to the shared latent space.

---

## Memory System

CORTEX-12 uses **explicit external memory** rather than implicit storage in
model weights.

Example memory entry:

```json
{
  "ruby": {
    "color": "red",
    "shape": "diamond",
    "size": "small"
  }
}
```
## Concepts are:
	1.	Rendered deterministically
	2.	Embedded through the cortex
	3.	Cached as vectors
	4.	Reused for comparison and reasoning

Memory updates do not require retraining.

⸻

## Design Principles

CORTEX-12 architecture prioritizes:
	•	Explicit separation of concerns
	•	Representation stability over accuracy
	•	Interpretability over scale
	•	Long unattended CPU execution

This makes the system suitable as a visual cortex, not a task-optimized
model.

