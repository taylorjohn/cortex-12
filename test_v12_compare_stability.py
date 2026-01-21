# ============================================================
# test_v12_compare_stability.py
# ------------------------------------------------------------
# PURPOSE:
# Verify that multisample similarity:
#   - Is stable (low variance)
#   - Separates same-concept vs different-concept
#
# PASS CRITERIA:
#   mean(sim_same) > mean(sim_diff)
#   std(sim_same) reasonably small
# ============================================================

import re
import numpy as np
from vl_jepa_llm_v12 import AgentBrain


def extract_similarity(text: str) -> float:
    """
    Extract the similarity float from strings like:
      "⚖️ Similarity(8): 0.8123"
    We want the value AFTER the colon, not the "(8)".
    """
    m = re.search(r":\s*([-+]?\d*\.\d+|\d+)\s*$", text)
    if m:
        return float(m.group(1))

    # Fallback: take the LAST number in the string
    nums = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", text)
    if not nums:
        raise ValueError(f"No numeric similarity found in: {text}")
    return float(nums[-1])


def main():
    print("[TEST] Compare Stability Test")

    agent = AgentBrain()

    # --- Learn concepts ---
    agent.visual_query("learn", {
        "action": "learn",
        "concept": "a",
        "definition": "red triangle",
        "size": "medium"
    })

    agent.visual_query("learn", {
        "action": "learn",
        "concept": "b",
        "definition": "red triangle",
        "size": "medium"
    })

    agent.visual_query("learn", {
        "action": "learn",
        "concept": "c",
        "definition": "blue square",
        "size": "medium"
    })

    # --- Run repeated comparisons ---
    same = []
    diff = []

    for _ in range(30):
        same.append(
            extract_similarity(
                agent.visual_query(
                    "compare",
                    {"action": "compare", "concept_a": "a", "concept_b": "b"}
                )
            )
        )
        diff.append(
            extract_similarity(
                agent.visual_query(
                    "compare",
                    {"action": "compare", "concept_a": "a", "concept_b": "c"}
                )
            )
        )

    same = np.array(same, dtype=float)
    diff = np.array(diff, dtype=float)

    print(f"SAME  mean={same.mean():.4f}  std={same.std():.4f}")
    print(f"DIFF  mean={diff.mean():.4f}  std={diff.std():.4f}")

    # --- Assertions ---
    assert same.mean() > diff.mean(), "Same-concept similarity should be higher than different-concept"
    assert same.std() < 0.10, "Same-concept similarity variance too high"

    print("OK Compare stability verified")


if __name__ == "__main__":
    main()