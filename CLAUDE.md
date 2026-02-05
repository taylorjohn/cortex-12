# CLAUDE.md - AI Assistant Guide for CORTEX-12

This document provides comprehensive guidance for AI assistants working with the CORTEX-12 codebase.

## Project Overview

**CORTEX-12** is a compact, CPU-trainable visual perception system that learns verifiable semantic representations through explicit axis structuring and compositional understanding.

**Key differentiator**: Unlike foundation models or classifiers, CORTEX-12 focuses on **concept-stabilized perception** - forming, preserving, and refining semantic concepts as stable, inspectable entities suitable for reasoning, memory, and decision-making.

### Current Status (February 2026)

- **Phase 3 Complete** with breakthrough results
- **100% shape certification** (perfect geometric shape discrimination)
- **Grade A compositional generalization** (3/4 tests passed)
- **CPU-only training** (3.5 hours on AMD Ryzen)
- **680KB trainable parameters** (vs 428MB for CLIP)

**Note**: Model weights (`*.pth`, `*.pt`) are gitignored and not included in the repository. Training produces checkpoints in `runs/` directory.

## Quick Reference

### Build & Run Commands

```bash
# Environment setup
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\Activate.ps1  # Windows PowerShell
pip install -r requirements.txt

# Run all tests
python run_all_v12_tests.py

# Individual tests
python test_v12_smoke.py
python test_v12_parse.py
python test_v12_size_compare.py
python test_v12_compare_stability.py

# Phase 3 training (default: 150 epochs)
python train_cortex_phase3_curriculum.py \
  --data_dir data/curriculum \
  --epochs 150 \
  --batch_size 4 \
  --output_dir runs/phase3

# Certification (after training produces a checkpoint)
python tools/certify_cortex12_phase3.py \
  --checkpoint runs/phase3/cortex_step_phase3_0050.pt \
  --data_dir data/balanced_images

# Inference example
python examples/verify_perception_phase3.py
```

### Key Files Quick Reference

| File | Purpose |
|------|---------|
| `cortex_adapter_v12.py` | Core architecture (680KB adapter) |
| `vl_jepa_llm_v12.py` | Runtime inference class |
| `train_cortex_phase3_curriculum.py` | Main training script |
| `tools/axis_schema_v3.json` | Semantic axis configuration |
| `run_all_v12_tests.py` | Master test runner |

## Architecture

### System Stack

```
RGB Image (224×224)
    ↓
DINOv2 ViT-S/14 (frozen, 21M parameters, loaded via torch.hub)
    ↓
CortexAdapter (trainable, 680KB parameters)
    ↓
128-D Structured Semantic Embedding
├─ Shape:       dims 0-31   (32-D) [100% certified]
├─ Size:        dims 32-47  (16-D) [54% certified]
├─ Material:    dims 48-63  (16-D)
├─ Color:       dims 64-79  (16-D) [93% certified]
├─ Location:    dims 80-87  (8-D)
├─ Orientation: dims 88-103 (16-D)
└─ Reserved:    dims 104-127 (24-D)
```

### Core Components

**CortexAdapter** (`cortex_adapter_v12.py`):
- Projects 384-D DINOv2 features to 128-D semantic space
- Uses separate linear heads for each semantic axis
- Enforces fixed dimensional allocation per attribute

**Cortex12Runtime** (`vl_jepa_llm_v12.py`):
- Loads DINOv2 backbone via `torch.hub` (not stored in checkpoint)
- Loads adapter weights from checkpoint
- Provides `perceive()` method for image embedding
- Supports explicit memory via JSON catalog

## Directory Structure

```
cortex-12/
├── cortex_adapter_v12.py       # Core adapter architecture
├── vl_jepa_llm_v12.py          # Runtime inference
├── train_cortex_phase3_curriculum.py  # Phase 3 training
├── train_cortex_phase2.py      # Phase 2 training
├── requirements.txt            # Dependencies (19 packages)
│
├── test_v12_*.py               # Standalone test scripts
├── run_all_v12_tests.py        # Master test suite
│
├── tools/
│   ├── certify_cortex12_phase3.py  # Certification tool
│   ├── axis_schema_v3.json         # Semantic axis schema
│   └── simple_comprehensive_test.py
│
├── examples/
│   └── verify_perception_phase3.py  # Inference demo
│
├── data/curriculum/
│   └── labels.json             # Training labels (427KB)
│   # Note: images/ directory must be populated for training
│
├── docs/
│   ├── ARCHITECTURE.md         # System architecture
│   ├── TRAINING.md             # Training details
│   ├── USE_CASES.md            # Application domains
│   ├── POSITIONING.md          # Market positioning
│   ├── ROADMAP.md              # Future directions
│   ├── KNOWN_LIMITATIONS.md    # System constraints
│   ├── NEUROSYMBOLIC_NOTES.md  # Neuro-symbolic integration
│   ├── architecture.svg        # Architecture diagram
│   └── diagrams/               # Additional diagrams
│
├── README.md                   # Project overview
├── CONTRIBUTING.md             # Contributor guide
├── MODEL_CARD.md               # Model specifications
├── VISION.md                   # Project philosophy
├── POSTTRAIN.md                # Post-training runbook
├── PHASE3_RESULTS.md           # Phase 3 detailed results
├── AUTHORS.md                  # Project authors
├── README_v13.md               # Version 13 notes
├── READMEv2.md                 # Version 2 notes
└── memory_vector_v12.json      # Semantic memory catalog (see Known Issues)
```

## Code Patterns & Conventions

### Deterministic Training

All training scripts enforce determinism:

```python
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

### Device Handling

CORTEX-12 is CPU-only by design:

```python
self.device = torch.device("cpu")
model.to(self.device)
```

### Model Checkpoint Format

Checkpoints contain only the adapter weights (not the frozen backbone):

```python
# Saving
torch.save({'cortex_state_dict': adapter.state_dict()}, path)

# Loading
ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
adapter.load_state_dict(ckpt['cortex_state_dict'], strict=False)
```

### Semantic Axis Access

Access specific semantic subspaces by dimension ranges:

```python
embedding = runtime.perceive('image.png')  # [128]
shape_features = embedding[0:32]      # 32-D shape
size_features = embedding[32:48]      # 16-D size
material_features = embedding[48:64]  # 16-D material
color_features = embedding[64:80]     # 16-D color
location_features = embedding[80:88]  # 8-D location
orientation_features = embedding[88:104]  # 16-D orientation
```

### Per-Axis Contrastive Loss

The key training innovation - prevents semantic axis collapse:

```python
def contrastive_axis_loss(embeddings, labels_list, axis_key, axis_dims, label_map):
    start, end = axis_dims
    sub_emb = embeddings[:, start:end+1]
    sub_emb = F.normalize(sub_emb, dim=1)
    # Apply contrastive loss only on this subspace
```

## Testing

### Test Framework

CORTEX-12 uses standalone Python scripts (not pytest):

- Each test is a self-contained executable
- Tests output to stdout with pass/fail indicators
- Run individually or via `run_all_v12_tests.py`

### Test Files

| Test | Purpose |
|------|---------|
| `test_v12_smoke.py` | Basic functionality check |
| `test_v12_parse.py` | Label parsing validation |
| `test_v12_size_compare.py` | Size invariance testing |
| `test_v12_compare_stability.py` | SAME vs DIFF stability |
| `bench_v12_forward.py` | Forward pass performance benchmark |
| `run_all_v12_tests.py` | Master runner (runs all above tests) |
| `tools/simple_comprehensive_test.py` | Phase 3 comprehensive evaluation |

### Expected Test Output

```
Test result: PASS
SAME similarity: 0.988 ± 0.002
DIFF similarity: 0.574 ± 0.010
Clear separation confirmed
```

## Dependencies

Core dependencies from `requirements.txt`:

```
# ML Stack
torch>=2.1
torchvision
torchaudio

# Core
numpy>=1.23
scipy
einops

# Image Processing
pillow
opencv-python
scikit-image

# Utilities
pyyaml
tqdm
rich
tensorboard
matplotlib
psutil
```

**External dependency**: DINOv2 is loaded via `torch.hub` from Facebook Research (not bundled).

## Important Considerations

### When Modifying Code

1. **Preserve axis layout**: The 128-D embedding structure is fixed - don't change dimension allocations
2. **Maintain determinism**: Keep random seeds and deterministic flags
3. **CPU-only**: Don't add GPU dependencies
4. **Checkpoint format**: Keep `cortex_state_dict` key for adapter weights

### When Adding Tests

1. Use standalone scripts, not pytest
2. Focus on stability metrics (SAME vs DIFF similarity)
3. Output clear pass/fail indicators
4. Can be run individually or via master runner

### When Training

1. Use per-axis contrastive loss to prevent collapse
2. Training data must be real geometric shapes (not solid colors)
3. Dynamic curriculum weights help balance axes
4. Expected training time: ~3.5 hours on CPU

### File Exclusions (.gitignore)

The following are NOT tracked in git:
- `venv/` - Virtual environment
- `runs/` - Training outputs and checkpoints
- `data/balanced_images/` - Generated images
- `*.pth`, `*.pt` - Model weights

## Common Tasks

### Evaluate a Model

```bash
# Run certification
python tools/certify_cortex12_phase3.py --checkpoint path/to/model.pt

# Run comprehensive test
python tools/simple_comprehensive_test.py
```

### Add a New Semantic Axis

1. Update `CortexAdapter` in `cortex_adapter_v12.py`
2. Update axis schema in `tools/axis_schema_v3.json`
3. Update training scripts with new axis loss
4. Update certification tools

### Debug Embeddings

```bash
python debug_embeddings.py    # General embedding analysis
python debug_dims.py          # Per-dimension analysis
python debug_intra_class.py   # Within-class variance
```

## Coding Standards

From `CONTRIBUTING.md`:

- **Style**: PEP 8 with Black formatting (88 char lines)
- **Type hints**: Encouraged for function signatures
- **Docstrings**: Google-style with usage examples
- **Imports**: stdlib → third-party → local

### Example

```python
from typing import Dict, List, Tuple

import torch
import numpy as np
from PIL import Image

from cortex_adapter_v12 import CortexAdapter

def compute_similarity(
    embedding1: torch.Tensor,
    embedding2: torch.Tensor,
    normalize: bool = True
) -> float:
    """
    Compute cosine similarity between two embeddings.

    Args:
        embedding1: First embedding [D]
        embedding2: Second embedding [D]
        normalize: Whether to L2-normalize before computing

    Returns:
        Similarity score in [-1, 1] range
    """
    ...
```

## Semantic Axis Schema

The axis schema (`tools/axis_schema_v3.json`) defines dimension ranges and tolerances:

```json
{
  "axes": {
    "shape": { "dims": [0, 31], "tolerance": 3.0 },
    "size": { "dims": [32, 47], "tolerance": 3.0 },
    "material": { "dims": [48, 63], "tolerance": 3.0 },
    "color": { "dims": [64, 79], "tolerance": 3.0 },
    "location": { "dims": [80, 87], "tolerance": 3.0 },
    "orientation": { "dims": [88, 103], "tolerance": 3.0 }
  }
}
```

## Performance Metrics

### Current Certification Results

| Axis | Certification | Status |
|------|---------------|--------|
| Shape | 100.00% | PERFECT |
| Color | 93.08% | CERTIFIED |
| Size | 54.33% | PROVISIONAL |
| **Average** | **82.47%** | +48% from baseline |

### Compositional Generalization

| Test | Score | Status |
|------|-------|--------|
| Zero-Shot Composition | 0.914 | PASS |
| Systematic Generalization | 0.763 | CLOSE |
| Additive Composition | 0.862 | PASS |
| Cross-Attribute Transfer | 0.998 | PERFECT |
| **Grade** | **A** | 3/4 passed |

## Known Issues

### memory_vector_v12.json

The `memory_vector_v12.json` file currently contains Python cleanup code instead of valid JSON. The `Cortex12Runtime` class will fail if initialized with the default memory path. This file needs to be regenerated or replaced with a proper JSON semantic memory catalog.

### Training Data

The `data/curriculum/images/` directory is not included in the repository. Training scripts expect this directory to be populated with generated training images. You may need to run image generation scripts or obtain the training dataset separately.

### Model Weights

All model checkpoint files (`*.pth`, `*.pt`) are excluded via `.gitignore`. After training, checkpoints are saved to `runs/` directory. For evaluation or inference, you need to either:
1. Train a model first using `train_cortex_phase3_curriculum.py`
2. Obtain pre-trained weights separately

## References

- **README.md**: Project overview and quick start
- **docs/ARCHITECTURE.md**: Detailed architecture explanation
- **docs/TRAINING.md**: Training methodology
- **POSTTRAIN.md**: Post-training evaluation runbook
- **CONTRIBUTING.md**: Full contributor guidelines
- **MODEL_CARD.md**: Model specifications and limitations
