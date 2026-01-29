
# CORTEX-12: A Verifiable Visual Cortex for Grounded Perception

**CPU-only · Interpretable · Certifiable · Deterministic**

CORTEX-12 is a **visual representation substrate** that learns stable, interpretable 128-D embeddings from pixels using JEPA principles, contrastive alignment, and explicit memory. It prioritizes **clarity, stability, and reproducibility** over scale or benchmark performance.

> “What if we built AI that is small enough to understand, structured enough to verify, and honest enough to explain?”

---

## 🧠 Core Capabilities

CORTEX-12 transforms raw pixels into **logic-ready perceptual facts**:

- ✅ **RGB → 128-D latent vectors** (DINOv2 ViT-S/14 backbone + lightweight adapter)
- ✅ **Explicit semantic axes**: color, shape, size, material, orientation, location
- ✅ **Post-hoc verifiable perception** via human-readable JSON certificates
- ✅ **Fixed embedding subspaces**: e.g., “dimensions 64–79 = color”
- ✅ **External, inspectable concept memory** (`memory_vector_v12.json`)
- ✅ **Compositional imagination** via structured rendering
- ✅ **CPU-only execution** — safe for long unattended runs (AMD Ryzen tested)

---

## 🎯 Why CORTEX-12?

Modern AI prioritizes **scale and performance** over **trust and transparency**. CORTEX-12 offers a counter-paradigm:

| Feature | CORTEX-12 | JEPAs / LLMs / VLMs |
|--------|-----------|---------------------|
| Semantic axes certified via validation | ✅ | ❌ |
| Human-readable JSON certificates | ✅ | ❌ |
| Works without retraining | ✅ | ❌ |
| CPU-only, deterministic, safe for unattended use | ✅ | ❌ |
| Embedding subspaces = symbolic predicates | ✅ | ❌ |
| Explicit memory + JEPA principles | ✅ | ❌ |

CORTEX-12 is **not**:
- ❌ A large language model (LLM)
- ❌ A foundation model
- ❌ A generative image model
- ❌ An end-to-end task optimizer

It **is**:
- ✅ A **visual cortex module** for neuro-symbolic systems
- ✅ A **calibrated perceptual instrument**
- ✅ A research platform for **verifiable grounded perception**

---

## 🚀 Quick Start

```powershell
# Setup
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Run smoke test
python test_v12_smoke.py
```

> 💡 **Requirements**: Windows 11, Python 3.10+, CPU-only PyTorch, AMD Ryzen-class CPU recommended

---

## 🧪 Phase-3: Curriculum-Based Semantic Grounding (Production-Ready)

CORTEX-12 now supports **verifiable multi-attribute perception** over synthetically generated scenes with explicit control over **six grounded attributes**:

- **Color** (12 classes: red, blue, amber, chartreuse, etc.)  
- **Shape** (6 classes: square, circle, hexagon, triangle, rectangle, star)  
- **Size** (3 classes: small, medium, large)  
- **Material** (5 classes: matte, glossy, metallic, glass, fabric)  
- **Orientation** (4 views → 3 certified classes due to 2D symmetry)  
- **Location** (continuous x,y coordinates)

### 🔑 Key Innovations

#### ✅ Verifiable Perception via Semantic Axis Certification
- Each attribute mapped to a **fixed subspace** of the 128-D embedding
- Runtime verification validates: *“dimension 64–79 = color”*
- Human-readable **JSON certificates** replace black-box probing

#### ✅ Physically Grounded Orientation Handling
- Recognizes that **0° and 180° are visually identical** for front-facing cubes in 2D
- Merges them into a single orientation class — **not a bug, but a feature**
- Achieves **76.5% orientation accuracy** with **0.61 confidence**

#### ✅ Transparent Failure Modes
- Low circle confidence? → **Add more circle examples**
- Amber/yellow confusion? → **Refine color boundaries**
- All issues are **diagnosable and fixable** without retraining from scratch

### 📊 Performance (Final Model: `cortex_step_phase3_0200.pt`)

| Attribute | Accuracy | Avg Confidence | Status |
|----------|----------|----------------|--------|
| **Material** | 99.4% | 0.618 | ✅ Outstanding |
| **Size** | 95.6% | 0.728 | ✅ Excellent |
| **Shape** | 90.9% | 0.346 | ⚠️ Good (circle weakness) |
| **Color** | 90.2% | 0.531 | ⚠️ Good (amber/yellow boundary) |
| **Orientation** | 76.5% | 0.610 | ✅ Correctly handles 2D symmetry |

> 💡 Confidence is calibrated via exponential distance-to-centroid for honest uncertainty.

### 🛠️ Usage

```powershell
# Train (CPU-only, ~24 hours)
python train_cortex_phase3_curriculum.py --epochs 200 --batch_size 4

# Certify axes
python tools/certify_cortex12_phase3.py --checkpoint runs/phase3/cortex_step_phase3_0200.pt --output_dir certs/phase3

# Verify perception
python examples/verify_perception_phase3.py --image data/curriculum/images/red_square_medium_0deg_matte_0_25_0_25.png --checkpoint runs/phase3/cortex_step_phase3_0200.pt --cert_dir certs/phase3
```

---

## 🧪 Phase-2: Tiny-ImageNet Foundation

Early training used **Tiny-ImageNet-200** to establish stable base representations:

- **Backbone**: DINOv2 ViT-S/14 (loaded via `torch.hub`)
- **Checkpoint**: `cortex_step05600.pt` (~680 KB)
- **Results**: Stable embeddings, clear concept separation, shape > size > color hierarchy

This phase validated the **JEPA-inspired architecture** before moving to controlled curriculum learning.

---

## 🧩 Use Cases

CORTEX-12 is ideal for applications requiring **trustworthy perception**:

- **Safety-critical robotics** (verifiable object understanding)
- **Assistive technology** (explainable visual reasoning)
- **Scientific instrumentation** (calibrated perceptual measurements)
- **Education and AI literacy** (transparent representation learning)
- **Neuro-symbolic AI** (pixels → logic-ready facts)

---

## 📏 Evaluation Philosophy

CORTEX-12 rejects standard accuracy benchmarks in favor of:

- **Verifiability**: Can you prove what the model knows?
- **Stability**: Do embeddings remain consistent across runs?
- **Interpretability**: Are semantic axes human-understandable?
- **Reproducibility**: Can others audit and reproduce your results?

> “We measure success not by leaderboard rank, but by how much we can understand.”

---

## 📁 Key Files

### Core System
- `vl_jepa_llm_v12.py` — CORTEX-12 runtime (visual cortex + memory)
- `cortex_adapter_v12.py` — Lightweight adapter with 6 projection heads
- `brain_vector_v12.pth` — Active cortex weights (adapter + heads)
- `memory_vector_v12.json` — Explicit concept memory

### Training
- `train_cortex_phase2_tinyimagenet.py` — Phase-2 trainer (Tiny-ImageNet)
- `train_cortex_phase3_curriculum.py` — Phase-3 trainer (synthetic curriculum)

### Verification & Tools
- `tools/certify_cortex12_phase3.py` — Axis certification with merged labels
- `examples/verify_perception_phase3.py` — Runtime perception verification
- `tools/validate_labels.ps1` — PowerShell label validation

### Testing
- `run_all_v12_tests.py`
- `test_v12_smoke.py`
- `test_v12_compare_stability.py`
- `bench_v12_forward.py`

---

## 🤝 Contributing

Contributions are welcome! Focus areas:
- Improved synthetic data generation
- Enhanced certification tooling
- New verification examples
- Documentation improvements

Please preserve the core principles: **CPU-first, verifiable, deterministic**.

---

## 📜 License

MIT License

---

## 📚 Citation

If you use CORTEX-12 in research, please cite:

```bibtex
@software{cortex12,
  author = {Taylor, John},
  title = {CORTEX-12: A Verifiable Visual Cortex for Grounded Perception},
  year = {2026},
  url = {https://github.com/taylorjohn/cortex-12}
}
```

---

> **CORTEX-12 proves that you don’t need scale to build systems that are simple, inspectable, and accountable.**  
> This is **perception as a calibrated scientific instrument** — not a black box.
```
