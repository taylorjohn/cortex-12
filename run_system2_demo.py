"""
CORTEX-12 v14: System 2 Reasoning Demo
Run this script to test the new Reasoning Engine and Adaptive Size.
"""

import torch
import torch.nn as nn
from cortex_adapter_v13 import CortexJEPA
from reasoning_predictor import ReasoningPredictor
from adaptive_size import AdaptiveSizeHead

def setup_v14_model(device='cpu'):
    print(f"🔧 Initializing CORTEX-12 v14 on {device}...")
    
    # 1. Load the base v13 Architecture (predictor disabled — we replace it below)
    model = CortexJEPA(enable_predictor=False, device='cpu')  # build on CPU first
    
    # 2. UPGRADE: Inject System 2 Reasoning Predictor BEFORE moving to device.
    #    Assigning an nn.Module to an attribute on an nn.Module registers it as a
    #    submodule, so it will appear in model.parameters() and move with .to().
    print("   👉 Upgrading to ReasoningPredictor (Beam Search enabled)")
    model.predictor = ReasoningPredictor(
        embedding_dim=128,
        num_beams=3,       # Keep 3 active hypotheses
        num_samples=5      # Generate 5 variations per step
    )
    
    # 3. UPGRADE: Inject Adaptive Size Head (same principle — assign before .to())
    print("   👉 Upgrading to AdaptiveSizeHead (Entropy Refinement enabled)")
    model.encoder.size_head = AdaptiveSizeHead(
        input_dim=384,
        num_classes=5,
        hidden_dim=128
    )
    
    # 4. Move everything to target device in one shot
    model = model.to(device)
    
    return model

def demo_reasoning_trajectory(model, device):
    print(f"\n🧠 [DEMO 1] System 2 Latent Reasoning")
    print("-" * 50)
    
    # Mock Start State (e.g., a "Red Circle")
    # In a real run, this comes from: start_emb = model.encode(image)
    start_emb = torch.randn(1, 128, device=device)
    
    # Define a complex plan
    actions = ['rotate_90', 'scale_up', 'recolor_blue']
    print(f"   Plan: {actions}")
    
    # Execute Reasoning (System 2)
    # The model will explore multiple futures and pick the most 'physically valid' one
    trajectory = model.predictor.plan_reasoned_trajectory(start_emb, actions)
    
    print(f"   ✅ Trajectory generated successfully!")
    print(f"   Steps: {len(trajectory)}")
    print(f"   Final State Norm: {torch.norm(trajectory[-1]).item():.4f}")
    print("   (Note: High validity scores in logs indicate 'Physics' checks passed)")

def demo_adaptive_size(model, device):
    print(f"\n📏 [DEMO 2] Adaptive Size Refinement")
    print("-" * 50)
    
    # Mock ambiguous input (Simulating a DINOv2 feature vector)
    # We use batch_size=5
    inputs = torch.randn(5, 384, device=device)
    
    # 1. Force the model to be "confused" for demonstration
    # We do this by ensuring the weights output high entropy initially
    with torch.no_grad():
        model.encoder.size_head.ordinal.feature_extractor[0].weight.fill_(0.01)
    
    # 2. Run Adaptive Prediction
    print("   Thinking...")
    final_logits, confusion_mask = model.encoder.size_head.predict_with_refinement(
        inputs,
        refinement_steps=5,  # Think 5 times if confused
        noise_scale=0.02
    )
    
    # 3. Report
    num_confused = confusion_mask.sum().item()
    print(f"   Input batch size: {inputs.size(0)}")
    print(f"   Items requiring System 2 thought: {num_confused}")
    
    if num_confused > 0:
        print(f"   ✅ Successfully refined {num_confused} ambiguous predictions.")
    else:
        print("   ✅ All inputs were clear (System 1 used).")

if __name__ == "__main__":
    # Detect Hardware (Prioritize M4 Pro)
    if torch.backends.mps.is_available():
        device = 'mps'
        print("✅ Using Metal Performance Shaders (Apple Silicon)")
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
        
    # Run Demos
    model = setup_v14_model(device)
    demo_reasoning_trajectory(model, device)
    demo_adaptive_size(model, device)
    
    print("\n🎉 CORTEX-12 v14 System 2 is operational.")