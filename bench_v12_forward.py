import time
import torch
from vl_jepa_llm_v12 import OmniJEPA, draw_tensor, DEVICE

def main():
    model = OmniJEPA().to(DEVICE).eval()
    x = draw_tensor("triangle", "red", "medium").unsqueeze(0).to(DEVICE)

    # warmup
    with torch.no_grad():
        for _ in range(10):
            model(x)

    t0 = time.time()
    iters = 100
    with torch.no_grad():
        for _ in range(iters):
            model(x)
    dt = time.time() - t0

    print(f"{iters} iters: {dt:.3f}s  => {iters/dt:.1f} it/s on {DEVICE}")

if __name__ == "__main__":
    main()
