# Model Card — CORTEX-12

## Model Type
Compact visual cortex

## Architecture
Frozen DINOv2 backbone with trainable adapter and heads

## Training Data
- Synthetic rendered objects
- Tiny-ImageNet (real images)

## Compute
CPU-only (AMD-friendly)

## Intended Use
- Representation learning research
- Neuro-symbolic experiments
- Interpretability studies

## Not Intended For
- Production deployment
- Safety-critical systems
- High-stakes decision making

## Phase-2 Real-Image Ablation (TinyImageNet)

We evaluate the effect of incorporating real images during Phase-2 training using a controlled ablation against a no-real-image baseline. At equivalent early checkpoints (1k steps), both models exhibit nearly identical symbolic stability (**SAME ≈ 0.988**, **DIFF ≈ 0.572**), demonstrating that the introduction of real images does not distort or collapse the learned concept manifold. Crucially, extending real-image training to later checkpoints (10k steps) preserves this stability (**SAME = 0.9878 ± 0.0024**, **DIFF = 0.5736 ± 0.0109**), indicating no late-stage degradation.

Qualitative semantic probes further show consistent ordering across identity, color, size, and shape variations (e.g., *ruby–ruby* ≈ 0.978, *ruby–size* ≈ 0.865, *ruby–color* ≈ 0.787, *ruby–shape* ≈ 0.754), closely matching earlier checkpoints and the no-real-image baseline. Together, these results demonstrate that Phase-2 real-image exposure improves perceptual grounding and invariance **without overwriting symbolic structure**, validating the JEPA-style separation between representation stabilization and semantic geometry.

## License
MIT
