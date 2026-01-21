import torch
from vl_jepa_llm_v12 import OmniJEPA, draw_tensor, DEVICE

def main():
    print("DEVICE:", DEVICE)

    model = OmniJEPA().to(DEVICE).eval()

    img = draw_tensor("triangle", "red", "medium")  # [3,128,128]
    x = img.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        feat, pc, ps, pz, sides = model(x)

    print("feat:", feat.shape)       # [1,128]
    print("pc:", pc.shape)           # [1,len(C_LIST)]
    print("ps:", ps.shape)           # [1,len(S_LIST)]
    print("pz:", pz.shape)           # [1,len(Z_LIST)]
    print("sides:", sides.shape)     # [1,1]
    print("OK")

if __name__ == "__main__":
    main()
