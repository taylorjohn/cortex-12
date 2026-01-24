# Model Card: CORTEX-12

## Key Claim (TL;DR)
**CORTEX-12 learns a stable symbolic representation space whose invariances are significantly improved by exposure to real images, without sacrificing symbolic discrimination or collapsing geometry.**

---

## Model Overview

**Model name:** CORTEX-12  
**Former name:** VL-JEPA v12  
**Version:** Phase-2 Final  
**Checkpoint:** `cortex_final.pt`  

CORTEX-12 is a neuro-symbolic visual representation model trained to align symbolic concepts with visual embeddings while preserving invariance, compositionality, and geometric stability. The model combines a frozen self-supervised vision backbone with a lightweight symbolic cortex trained via contrastive and supervised objectives.

The design goal is not end-to-end perception, but **stable symbolic grounding**: concepts should remain distinguishable, comparable, and invariant under nuisance transformations.

---

## Architecture

- **Vision backbone:** DINOv2 ViT-S/14 (frozen)
- **Trainable components:** Symbolic cortex adapters + concept heads
- **Output spaces:**
  - Visual feature embedding (128-D)
  - Concept logits
  - Size, shape, and relational heads

Only the cortex and symbolic heads are trained; the vision backbone remains fixed.

---

## Training Data

### Synthetic Data
Procedurally generated symbolic scenes used to define explicit concepts (e.g., color, shape, size).

### Real Images
**Tiny-ImageNet-200** is used *only* to encourage invariance and robustness in the visual embedding space. No class labels are used.

### Ablation
A matched **no-real-images** training run (synthetic-only) is used to isolate the effect of real visual data on invariance.

---

## Training Procedure

- **Total Phase-2 steps:** 12,000
- **Optimizer:** AdamW
- **Backbone:** Frozen throughout training
- **Losses:**
  - Supervised symbolic loss
  - Synthetic contrastive loss
  - Real-image contrastive (NT-Xent) loss
- **Checkpoints:** Saved every 200 steps
- **Final artifact:** `cortex_final.pt` (adapter + heads only)

---

## Evaluation

### 1. Symbolic Stability (Compare-Stability Test)

Measures whether identical concepts remain close and distinct concepts remain separated under random jitter.

**Final checkpoint (`cortex_final.pt`):**

- **SAME:** 0.9880 ± 0.0027  
- **DIFF:** 0.5706 ± 0.0082  

This confirms a well-structured symbolic embedding space with low variance and no collapse.

---

### 2. Real-Image Invariance (Tiny-ImageNet)

Measures cosine similarity between embeddings of two independently augmented views of the same real image.

- **Metric:** Mean cosine similarity
- **Samples:** n = 256
- **Seed:** 0

| Model Variant | REAL-INVAR (mean ± std) |
|--------------|-------------------------|
| No-real (1k) | 0.6855 ± 0.2152 |
| With-real (1k) | 0.8353 ± 0.1266 |
| **With-real (final)** | **0.8608 ± 0.1158** |

**Observation:**  
Exposure to real images produces a **large and persistent improvement** in invariance (+0.175 over no-real), while maintaining symbolic discrimination.

---

### 3. Semantic Probe Consistency

Manual symbolic probes demonstrate correct relational ordering:

| Comparison | Similarity |
|-----------|------------|
| ruby ↔ ruby | 0.9765 |
| ruby ↔ ruby_big | 0.8659 |
| ruby ↔ green_diamond | 0.7864 |
| ruby ↔ red_square | 0.7527 |

The ordering matches expected semantic proximity: identity > size > color > shape.

---

## Ablation Summary

The **synthetic-only** model achieves strong symbolic stability but **fails to generalize invariance to real images**. Adding unlabeled real images improves invariance substantially without degrading symbolic structure.

This supports the claim that **real data is not required for symbol learning, but is critical for invariance**.

---

## Failure Modes & Limitations

- **No end-to-end perception:** The model relies on a frozen backbone and does not learn low-level visual features.
- **Limited semantic richness:** Concepts are simple and compositional; abstract or high-level semantics are out of scope.
- **Invariance ceiling:** Invariance improves with real data but saturates; further gains may require backbone adaptation.
- **Not a classifier:** CORTEX-12 is not designed for ImageNet-style classification tasks.
- **No temporal reasoning:** The model does not process video or sequential data.

---

## Intended Use

- Neuro-symbolic research
- Representation learning analysis
- Grounded concept learning
- Invariance and compositionality studies

**Not intended for:**  
Safety-critical perception, real-world deployment, or production inference.

---

## Reproducibility

All reported metrics are produced by scripts included in the repository:

- `test_v12_smoke.py`
- `test_v12_compare_stability.py`
- `eval_real_invariance.py`

The final checkpoint is evaluated with no fine-tuning or test-time adaptation.

---

## Citation

If you reference this model, please cite as:

> *CORTEX-12: Stable Neuro-Symbolic Representations with Real-Image Invariance* (Phase-2)

---
---

# 2️⃣ MODEL_CARD.md — Evaluation Section (Conference Style)

```md
## Evaluation

### Key Claim (One-Sentence Summary)

> **CORTEX-12 preserves symbolic concept geometry while acquiring strong
> real-image invariance; this invariance arises from real-image exposure rather
> than increased synthetic-only training.**

---

### Experimental Setup

We evaluate CORTEX-12 using three complementary probes:

1. **Symbolic Stability Test**  
   Measures cosine similarity for SAME vs DIFF symbolic concepts to ensure
   concept geometry remains stable.

2. **Real-Image Invariance Test**  
   Measures cosine similarity between embeddings of two augmented views of the
   same Tiny-ImageNet image.

3. **Ablation Study**  
   Compares:
   - synthetic-only training (1k and 4k steps)
   - training with real images (1k steps)
   - the final mixed-training model

All results are averaged over *n = 256* samples with a fixed seed.

---

### Quantitative Results

| Model Variant      | SAME (↑)        | DIFF (↓)        | Real Invariance (↑) |
|--------------------|-----------------|-----------------|---------------------|
| Final (with-real)  | 0.9880 ± 0.0026 | 0.5734 ± 0.0117 | **0.8549 ± 0.1246** |
| With-real @1k      | 0.9880 ± 0.0027 | 0.5766 ± 0.0117 | 0.8201 ± 0.1388     |
| No-real @1k        | 0.9882 ± 0.0017 | 0.5729 ± 0.0140 | 0.6899 ± 0.1979     |
| No-real @4k        | 0.9879 ± 0.0028 | 0.5769 ± 0.0107 | 0.7177 ± 0.2062     |

---

### Interpretation

Symbolic SAME/DIFF scores remain statistically unchanged across all training
conditions, indicating that real-image exposure does not distort the learned
concept geometry.

In contrast, real-image invariance improves substantially with real-image
training (+0.16 absolute gain from No-real@1k to Final). Increasing synthetic-only
training from 1k to 4k steps yields only a marginal improvement and fails to
close the gap.

These results indicate that **real-image exposure is the primary driver of
invariance**, rather than training duration alone.

---

### Failure Modes and Limitations

- Synthetic-only training exhibits high variance in real-image invariance,
  indicating unstable generalization.
- Longer synthetic-only training partially improves invariance but plateaus
  well below real-image–trained models.
- Current evaluation focuses on low-resolution natural images (Tiny-ImageNet);
  higher-resolution or domain-shifted datasets remain future work.

---

### Reproducibility

All results can be reproduced using:

```bash
python demo_cortex12_showcase.py --seed 0 --n 256


---

## Contact

Project maintained by the original author.  
Contributions and discussions are welcome via pull requests or issues.

