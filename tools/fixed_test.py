"""
FIXED COMPREHENSIVE TEST - Uses correct merged labels
"""

import os
import json
import torch
import torch.hub
from PIL import Image
from torchvision import transforms
import numpy as np
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cortex_adapter_v12 import CortexAdapter

def load_model_and_certs(checkpoint_path, cert_dir):
    device = torch.device("cpu")
    dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    dinov2.eval()
    dinov2.to(device)
    
    model = CortexAdapter()
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt['cortex_state_dict'], strict=False)
    model.eval()
    model.to(device)
    
    certs = {}
    for fname in os.listdir(cert_dir):
        if fname.endswith("_cert.json"):
            axis = fname.replace("_cert.json", "")
            with open(os.path.join(cert_dir, fname)) as f:
                certs[axis] = json.load(f)
    
    return dinov2, model, certs, device

def load_image(img_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(img_path).convert("RGB")
    return transform(img).unsqueeze(0)

def predict_image(dinov2, model, certs, img_path, device):
    img_tensor = load_image(img_path)
    with torch.no_grad():
        features = dinov2(img_tensor.to(device))
        full_emb = model(features).squeeze().cpu().numpy()
    
    result = {}
    for axis, cert in certs.items():
        start, end = cert["embedding_dims"]
        sub_emb = full_emb[start:end+1]
        best_label, best_dist = None, float('inf')
        
        for label, centroid in cert["centroids"].items():
            dist = np.linalg.norm(sub_emb - np.array(centroid))
            if dist < best_dist:
                best_dist = dist
                best_label = label
        
        tol = cert["tolerance"]
        conf = np.exp(-best_dist / tol)
        result[axis] = {"prediction": best_label, "confidence": conf}
    
    return result

def main():
    checkpoint = "runs/phase3/cortex_step_phase3_0200.pt"
    cert_dir = "certs/phase3"
    data_dir = "data/balanced_images"  # Use balanced images with merged labels
    
    # Load everything
    dinov2, model, certs, device = load_model_and_certs(checkpoint, cert_dir)
    
    # Load MERGED labels
    labels_path = os.path.join(data_dir, "labels_merged.json")
    if not os.path.exists(labels_path):
        labels_path = os.path.join(data_dir, "labels.json")
    with open(labels_path, 'r') as f:
        labels = json.load(f)
    
    # Initialize metrics
    metrics = {
        'shape': {'correct': 0, 'total': 0, 'conf_sum': 0},
        'color': {'correct': 0, 'total': 0, 'conf_sum': 0},
        'size': {'correct': 0, 'total': 0, 'conf_sum': 0},
        'material': {'correct': 0, 'total': 0, 'conf_sum': 0},
        'orientation': {'correct': 0, 'total': 0, 'conf_sum': 0}
    }
    
    total_processed = 0
    img_dir = os.path.join(data_dir, "images")
    
    for img_name, gt_attrs in labels.items():
        img_path = os.path.join(img_dir, img_name)
        if not os.path.exists(img_path):
            continue
            
        try:
            predictions = predict_image(dinov2, model, certs, img_path, device)
            
            for axis in ['shape', 'color', 'size', 'material', 'orientation']:
                pred = predictions[axis]['prediction']
                conf = predictions[axis]['confidence']
                gt = gt_attrs[axis]
                
                # For orientation, both 0 and 180 should be '0' in merged labels
                if axis == 'orientation':
                    expected_gt = '0' if gt in ['0', '180'] else gt
                    is_correct = (pred == expected_gt)
                else:
                    is_correct = (pred == gt)
                
                if is_correct:
                    metrics[axis]['correct'] += 1
                metrics[axis]['total'] += 1
                metrics[axis]['conf_sum'] += conf
            
            total_processed += 1
            if total_processed % 200 == 0:
                print(f"Processed {total_processed} images...")
                
        except Exception as e:
            continue
    
    # Print results
    print("\n" + "="*60)
    print("📊 CORRECTED PERFORMANCE REPORT")
    print("="*60)
    
    for axis in metrics:
        if metrics[axis]['total'] > 0:
            acc = metrics[axis]['correct'] / metrics[axis]['total']
            avg_conf = metrics[axis]['conf_sum'] / metrics[axis]['total']
            print(f"\n{axis.upper()}:")
            print(f"  Accuracy: {acc:.3f} ({metrics[axis]['correct']}/{metrics[axis]['total']})")
            print(f"  Avg Confidence: {avg_conf:.3f}")

if __name__ == "__main__":
    main()