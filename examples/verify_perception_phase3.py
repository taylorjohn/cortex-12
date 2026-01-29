"""
Visual Perception Verifier for CORTEX-12 Phase 3
- Uses exponential confidence for large embedding distances
- CPU-only, deterministic
"""

import os
import json
import sys
import argparse
import torch
import torch.hub
from PIL import Image
from torchvision import transforms
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cortex_adapter_v12 import CortexAdapter


def load_image(img_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(img_path).convert("RGB")
    return transform(img).unsqueeze(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--cert_dir", default="certs/phase3", help="Directory with JSON certificates")
    args = parser.parse_args()

    device = torch.device("cpu")

    # Load DINOv2 backbone
    dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    dinov2.eval()
    dinov2.to(device)

    # Load adapter
    model = CortexAdapter()
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt['cortex_state_dict'], strict=False)
    model.eval()
    model.to(device)

    # Load certificates
    certs = {}
    for fname in os.listdir(args.cert_dir):
        if fname.endswith("_cert.json"):
            axis = fname.replace("_cert.json", "")
            with open(os.path.join(args.cert_dir, fname)) as f:
                certs[axis] = json.load(f)

    # Process image
    img_tensor = load_image(args.image)
    with torch.no_grad():
        features = dinov2(img_tensor.to(device))
        full_emb = model(features).squeeze().cpu().numpy()

    # Interpret each axis
    result = {"attributes": {}, "valid": True}
    for axis, cert in certs.items():
        start, end = cert["embedding_dims"]
        sub_emb = full_emb[start:end+1]
        best_label, best_dist = None, float('inf')
        
        # Find nearest centroid
        for label, centroid in cert["centroids"].items():
            dist = np.linalg.norm(sub_emb - np.array(centroid))
            if dist < best_dist:
                best_dist = dist
                best_label = label
        
        # Use exponential confidence (robust for large distances)
        tol = cert["tolerance"]
        conf = np.exp(-best_dist / tol)  # ✅ FIXED: Exponential decay
        
        if best_label is not None:
            result["attributes"][axis] = {"value": best_label, "confidence": round(conf, 3), "status": "CERTIFIED"}
        else:
            result["valid"] = False
            result["attributes"][axis] = {"value": None, "confidence": 0.0, "status": "UNKNOWN"}

    # Output
    print("\n🔍 CORTEX-12 PERCEPTION VERIFICATION")
    print("=" * 40)
    print(f"Image: {os.path.basename(args.image)}")
    print(f"Status: {'✅ VALID' if result['valid'] else '⚠️ UNKNOWN'}\n")
    for axis, attr in result["attributes"].items():
        if attr["status"] == "CERTIFIED":
            print(f"{axis.capitalize():>12}: {attr['value']} (conf: {attr['confidence']:.3f})")
        else:
            print(f"{axis.capitalize():>12}: ???")

if __name__ == "__main__":
    main()