"""
Phase 3 Certification Tool for CORTEX-12
- Uses merged orientation labels (0°/180° combined)
- Generates axis-specific certificates for verification
- CPU-only, deterministic
"""

import os
import json
import argparse
import torch
import torch.hub
from PIL import Image
from torchvision import transforms
import numpy as np
import sys

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
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--output_dir", default="certs/phase3", help="Output directory for certificates")
    parser.add_argument("--data_dir", default="data/balanced_images", help="Dataset directory with merged labels")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cpu")

    # Load DINOv2 backbone
    print("Loading DINOv2 ViT-S/14 backbone...")
    dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    dinov2.eval()
    dinov2.to(device)
    for p in dinov2.parameters():
        p.requires_grad = False

    # Load adapter
    model = CortexAdapter()
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt['cortex_state_dict'], strict=False)
    model.eval()
    model.to(device)

    # Load labels (PRIORITY: merged labels, then balanced, then original)
    labels_path = None
    possible_paths = [
        os.path.join(args.data_dir, "labels_merged.json"),
        os.path.join(args.data_dir, "labels.json"),
        os.path.join("data/curriculum", "labels.json")
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            labels_path = path
            break
    
    if labels_path is None:
        raise FileNotFoundError("No labels.json or labels_merged.json found in data directories")
    
    print(f"Using labels from: {labels_path}")
    with open(labels_path, 'r') as f:
        labels = json.load(f)

    # Define axis layout
    AXIS_LAYOUT = {
        "shape": (0, 31),
        "size": (32, 47),
        "material": (48, 63),
        "color": (64, 79),
        "location": (80, 87),
        "orientation": (88, 103)
    }

    # Build label maps (skip location - continuous)
    label_maps = {}
    for axis in ["shape", "size", "material", "color", "orientation"]:
        values = set()
        for attrs in labels.values():
            if axis in attrs:
                val = attrs[axis]
                # Handle orientation merging during certification
                if axis == "orientation" and val == "180":
                    val = "0"  # Merge 180° → 0° for certification
                values.add(val)
        label_maps[axis] = sorted(list(values))

    print("Label maps built:")
    for axis, values in label_maps.items():
        print(f"  {axis}: {len(values)} concepts")

    # Extract centroids for each concept
    centroids = {axis: {} for axis in AXIS_LAYOUT.keys()}
    counts = {axis: {} for axis in AXIS_LAYOUT.keys()}

    img_dir = os.path.join(os.path.dirname(labels_path), "images")
    total_imgs = len(labels)
    processed = 0

    for img_name, attrs in labels.items():
        img_path = os.path.join(img_dir, img_name)
        if not os.path.exists(img_path):
            continue

        try:
            # Process image
            img_tensor = load_image(img_path)
            with torch.no_grad():
                features = dinov2(img_tensor.to(device))
                full_emb = model(features).squeeze().cpu().numpy()

            # Update centroids for each axis
            for axis, (start, end) in AXIS_LAYOUT.items():
                if axis == "location":
                    continue  # Skip continuous axis
                
                if axis in attrs:
                    val = attrs[axis]
                    # Apply orientation merging during certification
                    if axis == "orientation" and val == "180":
                        val = "0"
                    
                    sub_emb = full_emb[start:end+1]
                    
                    if val not in centroids[axis]:
                        centroids[axis][val] = sub_emb.copy()
                        counts[axis][val] = 1
                    else:
                        centroids[axis][val] += sub_emb
                        counts[axis][val] += 1

            processed += 1
            if processed % 500 == 0:
                print(f"Processed {processed}/{total_imgs} images...")

        except Exception as e:
            print(f"Skipping {img_name}: {e}")
            continue

    # Average centroids
    for axis in centroids:
        if axis == "location":
            continue
        for val in centroids[axis]:
            centroids[axis][val] /= counts[axis][val]

    # Save certificates
    tolerance = 3.0  # Match your verification script
    for axis, centroid_dict in centroids.items():
        if axis == "location":
            continue
        
        cert = {
            "axis": axis,
            "embedding_dims": list(AXIS_LAYOUT[axis]),
            "tolerance": tolerance,
            "centroids": {k: v.tolist() for k, v in centroid_dict.items()},
            "concepts": len(centroid_dict)
        }
        
        cert_path = os.path.join(args.output_dir, f"{axis}_cert.json")
        with open(cert_path, 'w') as f:
            json.dump(cert, f, indent=2)
        
        status = "✅ VALID" if len(centroid_dict) >= 2 else "❌ INVALID"
        print(f"{axis.upper()}: {status} | {len(centroid_dict)} concepts")

    print(f"\n✅ Certification complete. Certificates saved to: {args.output_dir}")


if __name__ == "__main__":
    main()