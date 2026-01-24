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

## Contact

Project maintained by the original author.  
Contributions and discussions are welcome via pull requests or issues.

