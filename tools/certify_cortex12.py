# tools/certify_cortex12.py
"""
End-to-end certification of CORTEX-12 semantic axes.
Assumes cortex12.CORTEX12 has an encode() method.
"""

import os
import json
import numpy as np
from PIL import Image
from cortex12 import CORTEX12
from cortex12.semantic_axes import certify_axis

def main():
    with open("cert_data/metadata.json") as f:
        metadata = json.load(f)
    
    model = CORTEX12()
    embeddings, hues, circularities, sizes = [], [], [], []
    
    print("[INFO] Encoding images with CORTEX-12...")
    for item in metadata:
        img = np.array(Image.open(os.path.join("cert_data", item["image"])))
        emb = model.encode(img)
        embeddings.append(emb)
        hues.append(item["labels"]["hue"])
        circularities.append(item["labels"]["shape_circularity"])
        sizes.append(item["labels"]["size_scalar"])
    
    embeddings = np.stack(embeddings)
    print(f"[INFO] Encoded {len(embeddings)} samples.")
    
    # Certify key axes
    acc_hue = certify_axis(embeddings, np.array(hues), "hue", method="cluster")
    r2_circ = certify_axis(embeddings, np.array(circularities), "shape_circularity", method="linear")
    r2_size = certify_axis(embeddings, np.array(sizes), "size", method="linear")
    
    print("\n✅ Certification complete!")
    print(f"  hue accuracy: {acc_hue:.3f}")
    print(f"  circularity R²: {r2_circ:.3f}")
    print(f"  size R²: {r2_size:.3f}")
    print("\nCertificates saved to 'certificates/'")

if __name__ == "__main__":
    main()