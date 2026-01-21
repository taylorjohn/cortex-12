import os
import re
import shutil
import numpy as np

from vl_jepa_llm_v12 import AgentBrain, BRAIN_FILE

def extract_sim(s: str) -> float:
    m = re.search(r"([-+]?\d*\.\d+|\d+)", s)
    if not m:
        raise ValueError(f"Could not parse sim from: {s}")
    return float(m.group(1))

def run_suite(agent: AgentBrain, trials=30):
    # define a small eval set
    pairs_same = [
        ("red_tri_a", "red_tri_b", "red triangle", "red triangle"),
        ("blue_sq_a", "blue_sq_b", "blue square", "blue square"),
    ]
    pairs_diff = [
        ("red_tri_a", "blue_sq_a"),
        ("blue_sq_a", "green_circle"),
    ]

    # learn canonical concepts (fresh each run)
    agent.visual_query("learn", {"action":"learn","concept":"red_tri_a","definition":"red triangle","size":"medium"})
    agent.visual_query("learn", {"action":"learn","concept":"red_tri_b","definition":"red triangle","size":"medium"})
    agent.visual_query("learn", {"action":"learn","concept":"blue_sq_a","definition":"blue square","size":"medium"})
    agent.visual_query("learn", {"action":"learn","concept":"blue_sq_b","definition":"blue square","size":"medium"})
    agent.visual_query("learn", {"action":"learn","concept":"green_circle","definition":"green circle","size":"medium"})

    same = []
    diff = []

    for _ in range(trials):
        same.append(extract_sim(agent.visual_query("compare", {"action":"compare","concept_a":"red_tri_a","concept_b":"red_tri_b"})))
        same.append(extract_sim(agent.visual_query("compare", {"action":"compare","concept_a":"blue_sq_a","concept_b":"blue_sq_b"})))

        diff.append(extract_sim(agent.visual_query("compare", {"action":"compare","concept_a":"red_tri_a","concept_b":"blue_sq_a"})))
        diff.append(extract_sim(agent.visual_query("compare", {"action":"compare","concept_a":"blue_sq_a","concept_b":"green_circle"})))

    same = np.array(same, dtype=float)
    diff = np.array(diff, dtype=float)

    metrics = {
        "same_mean": float(same.mean()),
        "same_std": float(same.std()),
        "diff_mean": float(diff.mean()),
        "diff_std": float(diff.std()),
        "margin": float(same.mean() - diff.mean()),
    }
    return metrics

def main():
    # Expect user workflow:
    #  - Have some "pre" brain file saved elsewhere
    #  - Put "post" at brain_vector_v12.pth, run this script twice OR use swapping below.

    print("BRAIN_FILE:", BRAIN_FILE)
    if not os.path.exists(BRAIN_FILE):
        print("No brain file found. This still works, but you’re measuring baseline weights.")
    print("\nRunning evaluation with CURRENT brain...")

    agent = AgentBrain()
    m = run_suite(agent, trials=30)
    print("\nRESULTS (current):")
    for k,v in m.items():
        print(f"  {k}: {v:.4f}")

    # if you want a strict success criterion:
    # margin should be comfortably > 0
    if m["margin"] <= 0.05:
        print("\n⚠️ Margin is small. Consider more training / better contrastive negatives.")
    else:
        print("\n✅ Margin looks healthy.")

if __name__ == "__main__":
    main()