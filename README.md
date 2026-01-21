# cortex-12

### A Compact Visual Cortex for Grounded, Neuro-Symbolic Reasoning (CPU-Only)

CORTEX-12 is a compact, interpretable **visual cortex** built on JEPA principles.
It learns stable, low-dimensional vector representations for objects and scenes,
supports explicit memory and comparison, and adapts to real images using
contrastive self-supervision — all on **CPU-only hardware** rather than end-to-end prediction.

This project prioritizes **representation stability, interpretability, and
reproducibility** over scale.

---

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

## Core Capabilities

- RGB → compact 128-D latent vectors
- Explicit semantic axes (color, shape, size)
- Stable similarity-based reasoning
- External, inspectable concept memory
- Compositional imagination via rendering
- CPU-only operation (AMD-friendly)

---

## Repository Structure

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

## Requirements

- Windows 11
- Python 3.10+ (3.11 supported)
- CPU-only PyTorch
- AMD Ryzen-class CPU recommended

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

---


