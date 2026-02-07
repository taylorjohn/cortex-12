```
[INFO] Loading CORTEX-12 adapter from runs/cortex_v13_supervised/cortex_v13_supervised_best.pt...
[OK] Loaded from cortex_state_dict (epoch 95)

================================================================================
CORTEX-12: COMPOSITIONAL GENERALIZATION TEST SUITE
Inspired by VL-JEPA Principles
================================================================================

Model: runs/cortex_v13_supervised/cortex_v13_supervised_best.pt
Device: cpu

Running 4 compositional tests...

======================================================================
TEST 1: ZERO-SHOT COMPOSITIONAL GENERALIZATION
======================================================================

Hypothesis: Model can compose color + shape for unseen combinations
Training: red+circle, blue+square
Testing: red+square, blue+circle (never seen!)

[1/4] Encoding known combinations...
[2/4] Encoding novel combinations...
[3/4] Extracting semantic features...
[4/4] Testing compositional understanding...

Results:

  Red Square (unseen):
    Color matches 'red':     0.9032 ✓
    Shape matches 'square':  0.9907 ✓

  Blue Circle (unseen):
    Color matches 'blue':    1.0000 ✓
    Shape matches 'circle':  0.9987 ✓

  Average Similarity: 0.9731
  Status: ✓ COMPOSITIONAL

======================================================================
TEST 2: SYSTEMATIC GENERALIZATION
======================================================================

Hypothesis: Color is consistent across all shapes
Rule: red+circle, red+square, red+triangle → all have same 'red'

[1/2] Encoding red shapes...
[2/2] Computing color consistency...

Pairwise Color Similarities:
  red+circle   ↔ red+square  : 0.9032 ✓
  red+circle   ↔ red+triangle: 0.9190 ✓
  red+square   ↔ red+triangle: 0.8798 ○

  Average Consistency: 0.9006
  Status: ✓ SYSTEMATIC

======================================================================
TEST 3: ADDITIVE COMPOSITION (VL-JEPA PRINCIPLE)
======================================================================

Hypothesis: Embeddings form algebraic vector space
Test: red+square = red+circle - blue+circle + blue+square

[1/3] Encoding base combinations...
[2/3] Encoding target (red square)...
[3/3] Computing algebraic composition...

Vector Algebra Results:
  red+square = red+circle - blue+circle + blue+square

  Full embedding:  0.9457 ✓
  Color subspace:  0.9459 ✓
  Shape subspace:  0.9865 ✓

  Status: ✓ ADDITIVE

======================================================================
TEST 4: CROSS-ATTRIBUTE TRANSFER (BONUS)
======================================================================

Hypothesis: Color patterns transfer to unseen colors
Known: red+circle, blue+circle
Test: yellow+circle (new color, known shape)

[1/2] Encoding known and novel...
[2/2] Testing shape consistency across colors...

Shape Consistency (all circles):
  red+circle   ↔ yellow+circle: 0.9985
  blue+circle  ↔ yellow+circle: 0.9963
  red+circle   ↔ blue+circle:   0.9987

  Average: 0.9978
  Status: ✓ TRANSFERS

================================================================================
OVERALL COMPOSITIONAL ASSESSMENT
================================================================================

  Tests Passed: 4/4

    1. Zero-Shot:      ✓ (avg: 0.973)
    2. Systematic:     ✓ (avg: 0.901)
    3. Additive:       ✓ (sim: 0.946)
    4. Transfer:       ✓ (avg: 0.998)

  Final Grade: A+
  🎉 FULL COMPOSITIONAL GENERALIZATION

[OK] Results saved: results\compositional_tests.json
================================================================================

PS C:\Users\MeteorAI\desktop\cortex-12>

```
```
================================================================================
CORTEX-12 PHASE 3 SEMANTIC CERTIFICATION (WITH REAL SHAPES)
================================================================================
Model: runs/cortex_v13_supervised/cortex_v13_supervised_best.pt
Output: results/v13_certification
Samples per class: 1000
Axis layout source: constants.py (single source of truth)

Loading DINOv2 ViT-S/14 backbone...
Using cache found in C:\Users\MeteorAI/.cache\torch\hub\facebookresearch_dinov2_main
C:\Users\MeteorAI/.cache\torch\hub\facebookresearch_dinov2_main\dinov2\layers\swiglu_ffn.py:51: UserWarning: xFormers is not available (SwiGLU)
  warnings.warn("xFormers is not available (SwiGLU)")
C:\Users\MeteorAI/.cache\torch\hub\facebookresearch_dinov2_main\dinov2\layers\attention.py:33: UserWarning: xFormers is not available (Attention)
  warnings.warn("xFormers is not available (Attention)")
C:\Users\MeteorAI/.cache\torch\hub\facebookresearch_dinov2_main\dinov2\layers\block.py:40: UserWarning: xFormers is not available (Block)
  warnings.warn("xFormers is not available (Block)")
[OK] DINOv2 loaded
Loading model: runs/cortex_v13_supervised/cortex_v13_supervised_best.pt
[OK] Model loaded (epoch 95, loss 10.170817315857247)

```

```
PS C:\Users\MeteorAI\Desktop\cortex-12> python demo_vl_jepa_robust.py
======================================================================
 CORTEX-12 v13 DEMO: VL-JEPA Vector Algebra in Action
======================================================================

✅ Loading model (680 KB CPU-only)...
✅ Loading DINOv2 backbone...
Using cache found in C:\Users\MeteorAI/.cache\torch\hub\facebookresearch_dinov2_main
C:\Users\MeteorAI/.cache\torch\hub\facebookresearch_dinov2_main\dinov2\layers\swiglu_ffn.py:51: UserWarning: xFormers is not available (SwiGLU)
  warnings.warn("xFormers is not available (SwiGLU)")
C:\Users\MeteorAI/.cache\torch\hub\facebookresearch_dinov2_main\dinov2\layers\attention.py:33: UserWarning: xFormers is not available (Attention)
  warnings.warn("xFormers is not available (Attention)")
C:\Users\MeteorAI/.cache\torch\hub\facebookresearch_dinov2_main\dinov2\layers\block.py:40: UserWarning: xFormers is not available (Block)
  warnings.warn("xFormers is not available (Block)")

📸 Finding images in: data/enhanced_5sizes/images
   Found 54000 images

🧮 Computing: red+square = red+circle - blue+circle + blue+square

======================================================================
 RESULTS: Compositional Generalization
======================================================================
  Full embedding similarity:  1.000 ✓
  Color subspace (64-79):     1.000 ✓
  Shape subspace (0-31):      1.000 ✓

  VL-JEPA Threshold: 0.85
  Status: ✅ PASS

======================================================================
 🏆 CORTEX-12 v13 CERTIFICATION
======================================================================
  Shape:    100.0%  (6 geometric classes)
  Color:    100.0%  (12 colors)
  Size:      98.8%  (5-size task: tiny→huge)
  Average:   99.6%  🥇 PRODUCTION READY

  Compositional Grade: A+ (4/4 VL-JEPA tests passed)
  Training Cost: <$0.25 (CPU-only, 100 epochs)
  Model Size: 680 KB (vs 428 MB for CLIP)
======================================================================

💡 This proves: Structured representations enable reasoning.
   Not memorization — true compositional understanding.
======================================================================
PS C:\Users\MeteorAI\Desktop\cortex-12>

```
