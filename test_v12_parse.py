from vl_jepa_llm_v12 import AgentBrain

def main():
    b = AgentBrain()

    # punctuation + casing + extra words
    b.visual_query("learn", {"action":"learn","concept":"ruby", "definition":"A small, RED, diamond.", "size":"small"})
    print("ruby ->", b.memory["ruby"])

    b.visual_query("learn", {"action":"learn","concept":"ice", "definition":"Cyan triangle!!!", "size":"medium"})
    print("ice ->", b.memory["ice"])

    assert b.memory["ruby"]["color"] == "red"
    assert b.memory["ruby"]["shape"] in ("diamond", "sphere")  # depending on your shape list
    assert b.memory["ice"]["color"] == "cyan"
    assert b.memory["ice"]["shape"] == "triangle"
    print("OK")

if __name__ == "__main__":
    main()
