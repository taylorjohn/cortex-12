# batch_test.py
import os
images = [f for f in os.listdir("data/curriculum/images") if f.endswith(".png")][:10]
for img in images:
    os.system(f"python examples/verify_perception_phase3.py --image data/curriculum/images/{img} --checkpoint runs/phase3/cortex_step_phase3_0050.pt --cert_dir certs/phase3")