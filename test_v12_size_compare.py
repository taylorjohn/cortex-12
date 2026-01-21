from vl_jepa_llm_v12 import AgentBrain

def main():
    b = AgentBrain()

    b.visual_query("learn", {"action":"learn","concept":"tiny_red_ball", "definition":"red sphere", "size":"small"})
    b.visual_query("learn", {"action":"learn","concept":"huge_red_ball", "definition":"red sphere", "size":"extra_large"})

    # should be high-ish similarity but not identical if size influences features
    out = b.visual_query("compare", {"action":"compare","concept_a":"tiny_red_ball","concept_b":"huge_red_ball"})
    print(out)

if __name__ == "__main__":
    main()
