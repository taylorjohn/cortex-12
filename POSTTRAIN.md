# CORTEX-12 — Post-Training Evaluation Commands

This file is a runbook for validating a completed Phase-2 training run once
cortex_final.pt has been generated.

It answers two questions:
	•	Does the model load and behave correctly?
	•	What can it actually do, quantitatively and qualitatively?

⸻

Assumptions
	•	You are in the repository root
	•	Your Python virtual environment is activated
	•	Training has completed and produced cortex_final.pt
	•	You are using the patched runtime (vl_jepa_llm_v12_patched2.py)

⸻

## 0) Install / activate environment (if not already active)
```
python -m venv venv
source venv/bin/activate        # Linux / macOS
# OR
venv\Scripts\Activate.ps1       # Windows PowerShell
```
```
pip install -r requirements.txt
```

⸻

## 1) Point runtime at the final checkpoint

Most tooling expects the active weights file to be named:

brain_vector_v12.pth

Windows PowerShell
```
copy /Y runs\phase2_tiny\cortex_final.pt brain_vector_v12.pth
```
Linux / macOS
```
cp -f runs/phase2_tiny/cortex_final.pt brain_vector_v12.pth
```
(Adjust the path if your run directory is named differently.)

⸻

## 2) Run the full automated test suite (fast, high-signal)
```
python run_all_v12_tests.py
```
This includes:
	•	smoke tests
	•	parse tests
	•	size invariance
	•	SAME vs DIFF stability checks

You can also run them individually:
```
python test_v12_smoke.py
python test_v12_parse.py
python test_v12_size_compare.py
python test_v12_compare_stability.py
```

⸻

## 3) SAME vs DIFF similarity histogram (most important diagnostic)

This visualizes whether the learned embedding space is healthy or collapsed.
```
python inspect_latent.py \
  --samples 300 \
  --pairs 800 \
  --jitter 6 \
  --bins 40 \
  --out latent_hist.png \
  --no-llm
```
Expected outcome:
	•	SAME histogram is shifted right of DIFF
	•	DIFF does not overlap heavily with SAME
	•	No collapse (DIFF creeping up to SAME)

⸻

## 4) Interactive capability check (manual, no LLM)

Launch the runtime:
```
python vl_jepa_llm_v12_patched2.py --no-llm
```
Paste commands manually:
```
{"action":"learn","concept":"ruby","definition":"small red diamond"}
```
```
{"action":"learn","concept":"emerald","definition":"small green diamond"}
```
```
{"action":"compare","a":"ruby","b":"emerald","samples":50,"jitter":6}
```
Expected:
	•	Similarity is moderate (same shape/size, different color)
	•	Re-running yields stable values

⸻

## 5) Forward-pass performance benchmark
```
python bench_v12_forward.py
```
Confirms:
	•	CPU throughput
	•	No memory leaks
	•	Stable inference speed

⸻

## 6) Optional: kNN “accuracy” on Tiny-ImageNet (quantitative)
CORTEX-12 is not a classifier, but kNN retrieval gives a useful proxy metric.

Run kNN evaluation
```
python eval_knn_tinyimagenet.py \
  --tiny_root ./datasets/tiny-imagenet-200 \
  --k 10 \
  --train_per_class 20 \
  --val_max 2000 \
  --no-llm
```

## Interpretation:
	•	Random baseline ≈ 0.5% (1 / 200 classes)
	•	Any meaningful improvement indicates useful geometry
	•	This score should improve vs synthetic-only checkpoints

⸻

## 7) What “good” looks like (summary)
A successful cortex_final.pt should show:
	•	All tests pass (or only trivial warnings)
	•	SAME > DIFF separation is clear and stable
	•	No embedding collapse
	•	kNN accuracy well above random
	•	Stable behavior across repeated runs

⸻

## 8) Recommended next steps
	•	Archive the run directory + logs
	•	Keep cortex_final.pt immutable
	•	Use the patched runtime for all future evaluation
	•	Compare against earlier checkpoints using inspect_latent.py

⸻

This file is intentionally executable documentation — you can follow it line by line after every training run.
