import re
import sys
import numpy as np
from vl_jepa_llm_v12 import AgentBrain

def sim(s: str) -> float:
    m = re.search(r"([-+]?\d*\.\d+|\d+)", s)
    if not m:
        raise ValueError(f"Could not parse sim from: {s}")
    return float(m.group(1))

def main():
    agent = AgentBrain()

    agent.visual_query("learn", {"action":"learn","concept":"a","definition":"red triangle","size":"medium"})
    agent.visual_query("learn", {"action":"learn","concept":"b","definition":"red triangle","size":"medium"})
    agent.visual_query("learn", {"action":"learn","concept":"c","definition":"blue square","size":"medium"})

    same = []
    diff = []
    for _ in range(40):
        same.append(sim(agent.visual_query("compare", {"action":"compare","concept_a":"a","concept_b":"b"})))
        diff.append(sim(agent.visual_query("compare", {"action":"compare","concept_a":"a","concept_b":"c"})))

    same = np.array(same, dtype=float)
    diff = np.array(diff, dtype=float)

    same_mean = float(same.mean())
    diff_mean = float(diff.mean())
    margin = same_mean - diff_mean
    same_std = float(same.std())

    print(f"same_mean={same_mean:.4f} same_std={same_std:.4f} diff_mean={diff_mean:.4f} margin={margin:.4f}")

    # --- TUNE THESE ONCE, THEN KEEP THEM ---
    MIN_MARGIN = 0.10
    MAX_SAME_STD = 0.10

    if margin < MIN_MARGIN:
        print(f"❌ REGRESSION: margin {margin:.4f} < {MIN_MARGIN}")
        sys.exit(2)

    if same_std > MAX_SAME_STD:
        print(f"❌ REGRESSION: same_std {same_std:.4f} > {MAX_SAME_STD}")
        sys.exit(3)

    print("✅ OK (no regression)")

if __name__ == "__main__":
    main()