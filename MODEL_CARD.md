# Model Card: CORTEX-12 (VL-JEPA v12) (Conference Version)

## Key Claim

**Incorporating real-image exposure during Phase-2 training significantly improves perceptual invariance to real-world variation without degrading or destabilizing the learned symbolic concept space.**

---

## Abstract

CORTEX-12 is a neuro-symbolic vision–language agent designed to disentangle *perceptual grounding* from *symbolic structure*. The system combines a frozen self-supervised vision backbone (DINOv2) with a lightweight, trainable symbolic cortex optimized for stability, compositionality, and invariance. We evaluate the contribution of real-image exposure via a controlled ablation on Tiny-ImageNet, demonstrating that real images improve perceptual invariance while leaving symbolic geometry unchanged. These results support a staged training paradigm in which perceptual robustness and symbolic organization are learned under distinct objectives.

---

## Model Overview

**Architecture.** CORTEX-12 consists of:

* A frozen DINOv2 ViT-S/14 backbone for perceptual feature extraction
* A trainable "cortex" module mapping perceptual embeddings into a symbolic concept space
* A lightweight neuro-symbolic interface supporting concept learning, comparison, and composition

**Design Principle.** Symbolic stability is treated as a *first-class constraint*. The model is explicitly designed so that perceptual updates do not distort the relative geometry of learned concepts.

---

## Training Procedure

### Phase 1 (Synthetic Pretraining)

* Procedurally generated shapes, colors, sizes, and positions
* Objectives: concept separation, SAME/DIFF stability, compositional consistency

### Phase 2 (Grounding / Invariance)

Two variants were trained:

1. **With Real Images:** Phase-1 objectives + Tiny-ImageNet samples
2. **No-Real Ablation:** Phase-1 objectives only (synthetic data)

Both variants were trained with identical hyperparameters and evaluation schedules.

---

## Evaluation

### 1. Symbolic Stability (Compare-Stability Test)

Measures whether symbolic SAME/DIFF relationships remain consistent under sampling and jitter.

**Results (Step 1k):**

| Training Variant | SAME (↑)        | DIFF (↓)        |
| ---------------- | --------------- | --------------- |
| No Real Images   | 0.9884 ± 0.0021 | 0.5712 ± 0.0131 |
| With Real Images | 0.9886 ± 0.0022 | 0.5727 ± 0.0101 |

**Observation.** Symbolic geometry is statistically indistinguishable across variants, indicating that real-image exposure does not destabilize learned concepts.

---

### 2. Real-Image Invariance (Tiny-ImageNet)

Measures cosine similarity between embeddings of perturbed views of the same real image.

**Results (n = 256):**

| Training Variant | Mean Invariance (↑) | Std (↓) |
| ---------------- | ------------------- | ------- |
| No Real Images   | 0.6855              | 0.2152  |
| With Real Images | 0.8353              | 0.1266  |

**Observation.** Real-image exposure yields a large absolute gain in invariance (+0.15) and substantially reduces variance, indicating more consistent perceptual grounding.

---

## Ablation Claim

The ablation demonstrates a clean separation of effects:

* **What improves:** perceptual invariance to real-world variation
* **What remains unchanged:** symbolic structure, SAME/DIFF geometry, and stability

This supports the hypothesis that perceptual grounding can be improved without entangling or corrupting symbolic representations.

---

## Failure Modes & Limitations

* **No symbolic margin gains:** Real images do not widen symbolic separation or improve concept discrimination.
* **Invariance ≠ semantics:** Improvements reflect robustness to visual perturbations, not semantic reasoning.
* **Single-domain evaluation:** Real-image tests are limited to Tiny-ImageNet; broader domain transfer is untested.
* **Ceiling effects in stability metrics:** SAME/DIFF metrics saturate early and may obscure later regressions.

These limitations are intentional consequences of the phased training design.

---

## Intended Use

CORTEX-12 is intended as a research system for:

* Studying perceptual–symbolic decoupling
* Evaluating invariance without symbolic drift
* Prototyping staged neuro-symbolic learning pipelines

It is **not** intended as a general-purpose vision–language model or end-to-end LLM.

---

## Broader Impact

This work suggests that symbolic reasoning systems can be grounded in real perception without sacrificing stability, supporting modular and interpretable alternatives to monolithic end-to-end models.

---

## Reproducibility

All evaluations are script-driven and checkpointed. Ablation checkpoints, evaluation scripts, and fixed seeds are provided to enable exact reproduction of reported metrics.

---

## Summary

CORTEX-12 provides empirical evidence that perceptual grounding and symbolic organization can be trained as separable objectives. Real images improve invariance; symbols remain stable. This separation enables controlled scaling toward more robust neuro-symbolic agents.
