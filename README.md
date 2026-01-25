# CORTEX-12
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
<a href="#prerequisites"><img src="https://img.shields.io/badge/Compute-CPU--Only-blue.svg" alt="CPU Only"></a>
<a href="#prerequisites"><img src="https://img.shields.io/badge/Python-3.10%2B-blue.svg" alt="Python 3.10+"></a>

<p align="center">
  <img src="Cortex-12_logo.png" alt="CORTEX-12 Logo - A compact visual cortex for grounded, neuro-symbolic reasoning" width="800"/>
</p>

## A Compact Visual Cortex for Grounded, Neuro-Symbolic Reasoning

**CORTEX-12** is a CPU-only visual cortex that learns stable, interpretable
vector representations for grounded perception using JEPA principles,
contrastive alignment, and explicit memory. It prioritizes **clarity,
stability, and reproducibility** over scale or benchmark performance.

---

## Table of Contents

- [Overview](#overview)
- [Core Capabilities](#core-capabilities)
- [Why CORTEX-12](#why-cortex-12)
- [Quick Start](#quick-start)
- [Phase-2 Training](#phase-2-training)
- [Use Cases](#use-cases)
- [Evaluation Philosophy](#evaluation-philosophy)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

---

## Overview

CORTEX-12 is designed as a **representation substrate** rather than an
end-to-end agent:

- Learns **128-dim visual embeddings** from pixels  
- Supports **interpretable semantic axes** (color, shape, size)  
- Uses **explicit external memory** rather than implicit weights  
- Safe for long unattended CPU training

Unlike large models, CORTEX-12 is **simple, inspectable, and deterministic**.

---

## Core Capabilities

- **Visual Embeddings**: RGB images → compact 128-dimensional latent vectors
- **Semantic Attributes**: Explicit semantic axes (color, shape, size)
- **Stable Similarity**: Stable similarity-based reasoning across checkpoints
- **External Memory**: Inspectable concept memory (not implicit weights)
- **Compositional Imagination**: Structured rendering for compositional reasoning
- **CPU-Only Execution**: AMD-friendly, CPU-only operation

---

## Why CORTEX-12

**Not a large language model, generative model, or foundation model.**  
Rather, CORTEX-12 focuses on:
- **Grounded perception** with explicit memory
- **Interpretable geometry** instead of opaque weight embeddings
- **Representation stability** over competitive accuracy

This makes it suitable as **a visual cortex module** rather than a
standalone task solver.

---

## Quick Start

### Prerequisites
- **Operating System**: Windows 11 (Linux/macOS may work but are untested)
- **Python**: 3.10 or higher (3.11 supported)
- **Hardware**: AMD Ryzen-class CPU recommended (CPU-only, no GPU required)

### Setup (Windows PowerShell)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

## What CORTEX-12 Is (and Is Not)

**CORTEX-12 is:**
- A visual representation system
- A compact “cortex” rather than an end-to-end agent
- Explicitly grounded in perception
- Designed for long, unattended CPU training
- Suitable for neuro-symbolic research

**CORTEX-12 is not:**
- A large language model (LLM)
- A foundation model
- A generative image model
- An end-to-end task optimizer

---

## Repository Structure

```
📁 cortex-12/
├── 📁 docs/              # Architecture, roadmap, and technical documentation
├── 📁 examples/          # Example scripts and demonstrations
├── 📁 tools/             # Utilities for certification and validation
├── 📁 figs/              # Figures and visualizations
├── 📄 vl_jepa_llm_v12.py # CORTEX-12 main runtime
├── 📄 train_cortex_*.py  # Training scripts for Phase 1 & 2
├── 📄 test_v12_*.py      # Test suite
├── 📄 README.md          # Project overview and setup
├── 📄 LICENSE            # MIT License
└── 📄 requirements.txt   # Python dependencies
```

### Core Runtime
- `vl_jepa_llm_v12.py` — CORTEX-12 runtime (visual cortex + memory)
- `brain_vector_v12.pth` — active cortex weights (adapter + heads)
- `memory_vector_v12.json` — explicit concept memory

### Training
- `train_cortex_phase2_tinyimagenet.py` — Phase-2 trainer (Tiny-ImageNet)
- `runs/` — training checkpoints

### Tests & Utilities
- `run_all_v12_tests.py`
- `test_v12_smoke.py`
- `test_v12_parse.py`
- `test_v12_size_compare.py`
- `test_v12_compare_stability.py`
- `bench_v12_forward.py`
- `amd_batch_stress_test.py`

---

## Setup (Windows PowerShell)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```
## CPU-only PyTorch (if needed)
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```
## Verifiable Perception via Semantic Axis Certification

CORTEX-12 supports **semantic axis certification**—a method to make its 128-d embeddings interpretable and auditable.

### Quick Start
```bash
# Generate synthetic validation data
python tools/generate_certification_data.py

# Certify color, shape, and size axes
python tools/certify_cortex12.py

# Run a verifiable perception demo
python examples/verify_perception.py
Start
```

# 🧠 CORTEX-12 Semantic Axis Certification: A Novel Approach to Verifiable Perception

This document explains why **semantic axis certification** in CORTEX-12 is **unique and novel** compared to existing AI systems—including JEPAs, LLMs, and mainstream machine learning models.

---

## 🔍 Core Idea

CORTEX-12 enables **verifiable perception** by:
- Assigning fixed subspaces of its 128-d embedding to human-interpretable attributes (e.g., color, shape, size)
- **Certifying** these mappings using synthetic validation data
- Saving lightweight, human-readable **JSON certificates**
- Allowing runtime **probing** and **validation** without retraining

This turns perception into a **calibrated, auditable instrument**—not a black box.

---

## ❌ What Other Systems *Don’t* Do

### 1. **JEPAs (Joint Embedding Predictive Architectures)**
- **Examples**: I-JEPA, video JEPAs (Meta AI)
- **Limitations**:
  - Learn **implicit, high-dimensional embeddings** with no semantic guarantees
  - No mechanism to say “dimension 5 = redness”
  - Require GPUs and large-scale training
  - **Not designed for inspection or verification**
- ✅ **CORTEX-12**: Uses JEPA *principles* but enforces **explicit, certified semantic axes**

> 🚫 **No JEPA offers verifiable, interpretable axes out of the box.**

---

### 2. **Large Language Models (LLMs) & Vision-Language Models (VLMs)**
- **Examples**: CLIP, GPT-4V, LLaVA, Flamingo
- **Limitations**:
  - Representations are **emergent and distributed**
  - Interpretability relies on **post-hoc probing** (e.g., linear classifiers)—not guaranteed
  - Cannot produce **formal statements** like “this is red because subspace X matches centroid Y”
  - Not deterministic or CPU-friendly for long-term use
- ✅ **CORTEX-12**: Embedding space is **designed for human inspection**

> 🚫 **LLMs/VLMs are inherently opaque at the representation level.**

---

### 3. **Interpretable / Disentangled ML Models**
- **Examples**: β-VAE, FactorVAE, Concept Bottleneck Models
- **Limitations**:
  - **β-VAE**: Statistically disentangled—but latent meanings are **unknown without probing**
  - **Concept Bottleneck Models**: Require **human labels during training**—not post-hoc certifiable
  - None produce **human-readable certificates** (e.g., JSON files)
  - Most assume GPU training and lack **explicit memory**
- ✅ **CORTEX-12**: Certification is **post-hoc, reproducible, and decoupled from training**

> ⚠️ **Closest relatives—but still lack verifiability and CPU focus.**

---

### 4. **Neuro-Symbolic & Cognitive Architectures**
- **Examples**: Neural Turing Machines, ACT-R hybrids
- **Limitations**:
  - Often use **symbols as input**, not grounded visual perception
  - Rarely map **pixel inputs → certified symbolic predicates**
  - Tend to be complex research prototypes
- ✅ **CORTEX-12**: Perception **directly outputs logic-ready facts** via certified axes

> 🚫 **No system bridges pixels to symbols with geometric semantics.**

---

### 5. **Edge AI / TinyML Models**
- **Examples**: TensorFlow Lite, MobileNet
- **Limitations**:
  - Optimized for **speed/size**, not **trust or auditability**
  - Outputs are **class logits or opaque features**
- ✅ **CORTEX-12**: Designed for **trust on the edge**, not just efficiency

> 🚫 **Edge AI prioritizes inference speed, not explainability.**

---

## ✅ What Makes CORTEX-12 Unique?

| Feature | CORTEX-12 | Others |
|--------|----------|--------|
| **Semantic axes certified via validation** | ✅ | ❌ |
| **Human-readable JSON certificates** | ✅ | ❌ |
| **Works without retraining** | ✅ | ❌ |
| **CPU-only, deterministic, safe for unattended use** | ✅ | ❌ |
| **Embedding subspaces = symbolic predicates** | ✅ | ❌ |
| **Explicit memory + JEPA principles** | ✅ | ❌ |

---

## 💡 Why This Matters

Modern AI has prioritized **scale and performance** over **trust and transparency**. CORTEX-12 offers a counter-paradigm:

> **“What if we built AI that is small enough to understand, structured enough to verify, and honest enough to explain?”**

This is critical for:
- **Safety-critical robotics**
- **Assistive technology**
- **Scientific instrumentation**
- **Education and AI literacy**

---

## 📌 Conclusion

> **No existing AI system—whether JEPA, LLM, or ML model—combines semantic axis certification, CPU-only operation, explicit memory, and post-hoc verifiability like CORTEX-12.**

It fills a vital gap: **verifiable perception for trustworthy, grounded AI**.

CORTEX-12 proves that **you don’t need scale to build systems that are simple, inspectable, and accountable**.

--- 

*For implementation details, see [`cortex12/semantic_axes.py`](../cortex12/semantic_axes.py) and the [certification tools](../tools/).*





## 🚧 Training Progress (Phase 2)

**Status:** In progress  
**Dataset:** Tiny-ImageNet-200  
**Backbone:** DINOv2 ViT-S/14 (loaded via `torch.hub`)  
**Device:** CPU  
**Checkpoint Type:** Cortex head + concept memory (partial; backbone external)

---

### 📍 Mid-Phase-2 Milestone (~Step 5,600 / 12,000)

A mid-run checkpoint (`cortex_step05600.pt`) was evaluated to validate representation stability, concept separation, and overall system health.

---

### ✅ Load & Structural Integrity

- Checkpoint loads successfully in eval mode
- Cortex weights load as a **partial state_dict** (expected by design)
- Concept memory loads correctly
- Forward pass produces valid tensor shapes
- No NaNs or shape mismatches observed

**Smoke Test Output:**

feat: [1, 128]
pc:   [1, 19]
ps:   [1, 25]
pz:   [1, 4]
sides:[1, 1]
OK

---

### ✅ Representation Stability (Automated)

`test_v12_compare_stability.py` was run against the checkpoint.

**Results:**

SAME  mean = 0.9887   std = 0.0027
DIFF  mean = 0.5720   std = 0.0100
OK Compare stability verified

## Phase-2 Real-Image Ablation (TinyImageNet)

We evaluate the effect of incorporating real images during Phase-2 training using a controlled ablation against a no-real-image baseline. At equivalent early checkpoints (1k steps), both models exhibit nearly identical symbolic stability (**SAME ≈ 0.988**, **DIFF ≈ 0.572**), demonstrating that the introduction of real images does not distort or collapse the learned concept manifold. Crucially, extending real-image training to later checkpoints (10k steps) preserves this stability (**SAME = 0.9878 ± 0.0024**, **DIFF = 0.5736 ± 0.0109**), indicating no late-stage degradation.

Qualitative semantic probes further show consistent ordering across identity, color, size, and shape variations (e.g., *ruby–ruby* ≈ 0.978, *ruby–size* ≈ 0.865, *ruby–color* ≈ 0.787, *ruby–shape* ≈ 0.754), closely matching earlier checkpoints and the no-real-image baseline. Together, these results demonstrate that Phase-2 real-image exposure improves perceptual grounding and invariance **without overwriting symbolic structure**, validating the JEPA-style separation between representation stabilization and semantic geometry.


**Interpretation:**
- Extremely high self-similarity with very low variance → **stable embeddings**
- Clear separation between different concepts
- No evidence of representation collapse or drift

---

### 🧠 Manual Concept Geometry Probes

Manual evaluations were performed using the interactive eval interface.

| Comparison | Similarity (↑ = closer) | Interpretation |
|-----------|--------------------------|---------------|
| ruby ↔ ruby | ~0.974 | High self-consistency |
| ruby ↔ ruby_big | ~0.865 | Size encoded as a separable attribute |
| ruby ↔ emerald | ~0.78 | Color separation present but weaker |
| ruby ↔ green_diamond | ~0.78 | Confirms color is a lower-weight axis |
| ruby ↔ red_square | ~0.75 | Shape difference dominates separation |

**Key Insight:**  
At this stage, the model prioritizes **object structure / shape**, followed by **size**, with **color encoded but less dominant**. This behavior is consistent with expected JEPA-style training dynamics, where structural identity stabilizes before fine-grained attribute disentanglement.

---

### 🎨 Scene Composition Check

The imagination / composition pathway was verified:

[SCENE] GENERATED: objects=2 complexity=1.17

Repeated runs produced consistent outputs, confirming deterministic behavior and a functioning composition pipeline.

---

### 📦 Checkpoint Characteristics

- File size: ~680 KB
- Contains:
  - Cortex projection / adapter weights
  - Concept memory state
- Does **not** include backbone weights (loaded from `torch.hub`)

This design allows fast copying, versioning, and evaluation without duplicating large backbone files.

---

### 📈 Summary (Step ~5600)

At this mid-Phase-2 checkpoint, the model demonstrates:

- ✅ Stable representations
- ✅ Clear concept separation
- ✅ Sensible attribute geometry (shape > size > color)
- ✅ Fully operational eval and tooling stack

This checkpoint serves as a **baseline reference** for later Phase-2 checkpoints (e.g. ~8k, ~10k, ~12k), where further attribute disentanglement—particularly color—is expected.

---

### Tiny-ImageNet doesn’t rewrite the concept manifold — it makes real-image embeddings invariant.

“On a real-image invariance benchmark (two augmented views of the same Tiny-ImageNet image), the with-real model achieves 0.835 ± 0.127 cosine similarity vs 0.686 ± 0.215 for the no-real ablation at 1k steps, demonstrating substantially improved real-world invariance without degrading symbolic stability.”

### Real-image invariance (same step count: 1,000)

### No-real (λ_real=0)

- mean = 0.6855
- std = 0.2152

### With-real (Tiny-ImageNet NT-Xent)

- mean = 0.8353
- std = 0.1266
  
---
### Results Summary (Phase-2)

| Metric | No-real @1k | With-real @1k | With-real Final |
|---|---:|---:|---:|
| Compare-Stability SAME (mean ± std) | 0.9884 ± 0.0021 | 0.9886 ± 0.0022 | 0.9880 ± 0.0027 |
| Compare-Stability DIFF (mean ± std) | 0.5712 ± 0.0131 | 0.5727 ± 0.0101 | 0.5706 ± 0.0082 |
| Real-Image Invariance REAL-INVAR (mean ± std) | 0.6855 ± 0.2152 | 0.8353 ± 0.1266 | 0.8608 ± 0.1158 |

### Demo Results (n=256, seed=0)

| Variant | SAME (mean±std) | DIFF (mean±std) | REAL-INVAR (mean±std) |
|---|---:|---:|---:|
| Final (with-real @12k) | 0.9875 ± 0.0025 | 0.5739 ± 0.0109 | 0.8549 ± 0.1246 |
| With-real @1k | 0.9881 ± 0.0019 | 0.5739 ± 0.0111 | 0.8201 ± 0.1388 |
| No-real @1k | 0.9882 ± 0.0027 | 0.5708 ± 0.0098 | 0.6899 ± 0.1979 |
| No-real @4k (final) | 0.9883 ± 0.0029 | 0.5759 ± 0.0124 | 0.7177 ± 0.2062 |

---

## Quick Demo: Stability, Invariance, and Ablation (CORTEX-12)

This demo reproduces the core claims of **CORTEX-12** in a single command:
symbolic stability, real-image invariance, and a controlled ablation showing
that real images — not just longer training — drive invariance.

### Run the Demo

```powershell
python demo_cortex12_showcase.py `
  --tiny_root .\datasets\tiny-imagenet-200 `
  --final_ckpt runs\eval_snapshots\cortex_final.pt `
  --withreal_1k runs\eval_snapshots\cortex_withreal_step01000.pt `
  --noreal_1k runs\eval_snapshots\cortex_noreal_step01000.pt `
  --noreal_4k runs\eval_snapshots\cortex_noreal_4k_final.pt `
  --n 256 `
  --seed 0
```

---
## Figures

### Real-Image Invariance (Tiny-ImageNet)
![CORTEX-12 Real-Image Invariance](figs/real_invariance_plot.png)

### Symbolic Stability (Compare-Stability)
![CORTEX-12 Symbolic Stability](figs/symbolic_stability_plot.png)



---

## Contact & Support

For questions, issues, or contributions:

- **Issues**: Report bugs or request features via [GitHub Issues](../../issues)
- **Contributing**: See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines
- **Vision & Roadmap**: Review [VISION.md](VISION.md) and [docs/ROADMAP.md](docs/ROADMAP.md)
- **Model Details**: Refer to [MODEL_CARD.md](MODEL_CARD.md) for technical specifications

For research collaborations or academic inquiries, see [AUTHORS.md](AUTHORS.md).

---
## Diagrams testing

![Two-world training diagram](docs/diagrams/training_two_worlds.svg)

![Geometry stability](docs/diagrams/geometry_contract.svg)

![External memory loop](docs/diagrams/external_memory_loop.svg)

![NT-Xent intuition](docs/diagrams/nt_xent.svg)

The contrastive objective minimizes:

$$
\mathcal{L}_{i} = -\log
\frac{\exp(\text{sim}(z_i,z_j)/\tau)}
{\sum_{k\neq i}\exp(\text{sim}(z_i,z_k)/\tau)}
$$
