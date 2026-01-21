import time
import torch
import numpy as np
from vl_jepa_llm_v12 import OmniJEPA, draw_tensor, DEVICE

def main():
    print("DEVICE:", DEVICE)
    model = OmniJEPA().to(DEVICE).eval()

    # pre-generate one sample tensor
    base = draw_tensor("triangle", "red", "medium").unsqueeze(0)  # [1,3,128,128]

    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 96, 128]
    iters = 50

    results = []

    for bs in batch_sizes:
        try:
            x = base.repeat(bs, 1, 1, 1).to(DEVICE)

            # warmup
            with torch.inference_mode():
                for _ in range(10):
                    model(x)

            t0 = time.time()
            with torch.inference_mode():
                for _ in range(iters):
                    model(x)
            dt = time.time() - t0

            imgs_per_sec = (bs * iters) / dt
            results.append((bs, imgs_per_sec))
            print(f"bs={bs:3d}  imgs/s={imgs_per_sec:10.2f}")

        except Exception as e:
            print(f"bs={bs:3d}  FAILED: {e}")
            break

    if results:
        best = max(results, key=lambda t: t[1])
        print("\nBEST:", best)

if __name__ == "__main__":
    main()