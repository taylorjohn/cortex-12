# Contributing to Cortex-12

Thank you for your interest in contributing to **Cortex-12**, a neuro-symbolic vision–language research project.  
We welcome thoughtful contributions — especially experiments, evaluations, documentation, and tooling — while maintaining strict safeguards around reproducibility and core results.

This document explains **how to contribute**, **what kinds of contributions are welcome**, and **what is intentionally protected**.

---

## Guiding Principles

Cortex-12 prioritizes:

- **Scientific rigor and reproducibility**
- **Clear ablation boundaries**
- **Separation of core results from exploratory work**
- **Incremental, reviewable changes**

Contributions should **add clarity, evidence, or capability** — not overwrite or obscure existing results.

---

## What You Can Contribute

We welcome contributions in the following areas:

### 1. Evaluation & Analysis
- New **evaluation scripts**
- Additional **metrics or probes**
- Robustness, invariance, or stability tests
- Visualization or reporting improvements

👉 Preferred location:
```
/eval/
/tests/
/analysis/
```
---

### 2. Ablations & Experiments
- New ablation experiments
- Controlled comparisons (with clear flags/configs)
- Alternative training schedules or loss weightings

👉 Requirements:
- Must be **opt-in**, not default
- Must preserve existing baselines
- Results must be logged separately

👉 Preferred location:
```
/experiments/
/ablations/
```
---

### 3. Documentation
- README improvements
- Model Card clarifications
- Diagrams, explanations, or tutorials
- Reproducibility notes

👉 Preferred location:
```
README.md
MODEL_CARD.md
docs/
```

---

### 5. Bug Fixes (Non-Behavioral)
- Typo fixes
- Logging bugs
- Crashes or edge-case failures

⚠️ Bug fixes **must not change model behavior** unless explicitly discussed.

---

## What Is Protected (Do Not Modify Directly)

The following are **protected** and should **not be modified directly in a pull request** without prior discussion:

### 🚫 Core Training Code
- `train_cortex_phase2*.py`
- Loss definitions tied to published results
- Default hyperparameters used in reported experiments

### 🚫 Reference Checkpoints
- Any `.pt` files used in evaluation claims
- Files under:
```
/runs/
/eval_snapshots/
```
### 🚫 Reported Results
- Numbers cited in the README or Model Card
- Figures or tables tied to claims

If you want to improve or challenge a result:
➡️ **Add a new experiment or evaluation**, don’t overwrite the original.

---

## How to Propose Changes to Core Behavior

If you believe a core change is warranted:

1. **Open an issue first**
2. Clearly state:
   - What changes
   - Why it improves correctness or clarity
   - How it affects existing claims
3. Provide:
   - Side-by-side comparisons
   - Reproducible commands
   - Quantitative evidence

Core changes without discussion will not be merged.

---

## Pull Request Guidelines

### ✅ Good Pull Requests
- Small, focused, and well-documented
- Include:
  - Purpose
  - What changed
  - How to reproduce
- Do **not** delete or rewrite existing experiments

### ❌ Avoid
- Large refactors without justification
- Mixing unrelated changes
- Silent changes to training behavior
- Overwriting baseline results

---

## Style & Quality Expectations

- Python 3.10+
- Deterministic seeds where applicable
- Clear logging
- Descriptive variable names
- Comments for non-obvious logic

If it’s hard to understand, it’s hard to review.

---

## Licensing & Attribution

By contributing, you agree that your work may be included under the project’s license.

If your contribution is inspired by or derived from other work:
- Cite it clearly
- Add references in comments or docs

---

## Communication

- Use **issues** for discussion
- Be precise and technical
- Assume good faith
- Keep critiques evidence-based

This is a research project — disagreements are expected, but rigor is required.

---

## Final Note

Cortex-12 is designed to be:
- **Inspectable**
- **Auditable**
- **Extendable without erosion**

If your contribution makes the system clearer, more testable, or more robust — it’s welcome.

Thanks for helping push this work forward.



