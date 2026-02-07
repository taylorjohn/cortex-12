#!/usr/bin/env python3
"""
CORTEX-12 Phase 3 Certification - PROPER VERSION (FIXED)
✓ Uses constants.py for axis layout consistency
✓ weights_only=True security fix
✓ Corrupted image error handling
Tests on geometric shapes matching training data distribution
"""
import os
import sys
import json
import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image, ImageDraw, UnidentifiedImageError
from torchvision import transforms
import numpy as np
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
from constants import AXIS_LAYOUT  # CENTRALIZED AXIS LAYOUT

try:
    from cortex_adapter_v12 import CortexAdapter
    print("[OK] CortexAdapter imported")
except ImportError as e:
    print(f"[ERROR] Could not import CortexAdapter: {e}")
    sys.exit(1)

# ... [keep all shape drawing functions identical: draw_circle, draw_square, draw_triangle, create_shape_image] ...

class CORTEX12Certifier:
    """Certifies semantic axes using proper geometric shapes"""
    def __init__(self, model_path, device='cpu'):
        self.device = torch.device(device)
        # Load DINOv2 backbone
        print("Loading DINOv2 ViT-S/14 backbone...")
        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', trust_repo=True)
        self.backbone.eval()
        self.backbone.to(self.device)
        print("[OK] DINOv2 loaded")
        # Load adapter
        print(f"Loading model: {model_path}")
        self.adapter = CortexAdapter(input_dim=384)
        # SECURITY FIX: weights_only=True
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        if 'cortex_state_dict' in checkpoint:
            self.adapter.load_state_dict(checkpoint['cortex_state_dict'])
        elif 'adapter_state_dict' in checkpoint:
            self.adapter.load_state_dict(checkpoint['adapter_state_dict'])
        else:
            print("[ERROR] Could not find adapter weights in checkpoint")
            print("Available keys:", list(checkpoint.keys()))
            sys.exit(1)
        self.adapter.eval()
        self.adapter.to(self.device)
        epoch = checkpoint.get('epoch', 'unknown')
        loss = checkpoint.get('avg_loss', checkpoint.get('loss', 'unknown'))
        print(f"[OK] Model loaded (epoch {epoch}, loss {loss})")
        # Transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def encode_image(self, image):
        """Encode image to 128-D embedding with corruption handling"""
        if isinstance(image, str):
            try:
                image = Image.open(image).convert('RGB')
            except (UnidentifiedImageError, OSError) as e:
                print(f"⚠️ Skipping corrupted image: {e}")
                return np.zeros(128)
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.backbone(tensor)
            embedding = self.adapter(features)
        return embedding.cpu().numpy()[0]

    # ... [keep generate_validation_data, certify_axis, generate_certificate identical] ...

    def certify_axis(self, axis_name, embeddings_by_class):
        """Certify semantic axis using centralized layout"""
        if axis_name not in AXIS_LAYOUT:
            raise ValueError(f"Unknown axis: {axis_name}. Valid: {list(AXIS_LAYOUT.keys())}")
        dims = AXIS_LAYOUT[axis_name]
        # ... [rest of method identical, just uses dims from constant] ...

# ... [keep main() identical, just import AXIS_LAYOUT from constants] ...

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CORTEX-12 Phase 3 Semantic Certification (PROPER SHAPES)')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--output-dir', type=str, default='results/phase3_proper_certification', help='Output directory')
    parser.add_argument('--num-samples', type=int, default=500, help='Samples per class for validation')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu/cuda)')
    parser.add_argument('--skip-generation', action='store_true', help='Skip data generation if already exists')
    args = parser.parse_args()
    
    print("="*80)
    print("CORTEX-12 PHASE 3 SEMANTIC CERTIFICATION (WITH REAL SHAPES)")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Output: {args.output_dir}")
    print(f"Samples per class: {args.num_samples}")
    print(f"Axis layout source: constants.py (single source of truth)")
    print()
    
    # Initialize certifier
    certifier = CORTEX12Certifier(args.model, device=args.device)
    # ... [rest of main identical] ...