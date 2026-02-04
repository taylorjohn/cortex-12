"""
CORTEX-12 v14: RL Reasoning Fine-Tuning
Implements 'Scaling Laws for Reasoning' (Tan et al., 2025).

Fine-tunes the Predictor using Reinforcement Learning (REINFORCE/PPO)
to maximize 'Structural Integrity' and 'Semantic Match' rewards.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from reasoning_predictor import ReasoningPredictor

def compute_reward(predicted_emb, target_emb, integrity_score):
    """
    Reward = (Semantic Similarity) + (Physical Validity)
    """
    # 1. Semantic Match (Cosine Similarity)
    # Range: [-1, 1]
    cosine_sim = F.cosine_similarity(predicted_emb, target_emb)
    
    # 2. Integrity (SIGReg Score)
    # Range: Negative to 0. We map it to [0, 1] roughly
    # We clip extremely bad scores
    integrity_reward = torch.clamp(integrity_score + 10.0, 0, 10.0) / 10.0
    
    # Combined Reward
    # We prioritize being CORRECT (cosine) over just being VALID (integrity)
    total_reward = cosine_sim + 0.2 * integrity_reward
    
    return total_reward

def train_reasoning_rl(
    model, 
    dinov2,
    dataloader, 
    device='mps', 
    epochs=5,
    lr=1e-5
):
    """
    RL Fine-tuning Phase.
    
    Args:
        model: CORTEX-12 JEPA model (with ReasoningPredictor)
        dinov2: Frozen DINOv2 backbone (produces 384-D features)
        dataloader: Dataset providing (start_img, target_img, action_name)
    """
    # Maps string action names to the integer IDs used by TransformationPredictor.TRANSFORMS
    ACTION_MAP = {
        'rotate_90': 0, 'rotate_180': 1, 'rotate_270': 2,
        'scale_up': 3, 'scale_down': 4,
        'flip_h': 5, 'flip_v': 6,
        'recolor_red': 7, 'recolor_blue': 8, 'recolor_green': 9,
        'translate_x': 10, 'translate_y': 11,
    }

    print(f"\n🚀 Starting RL Reasoning Fine-Tuning ({device})...")
    
    # Only train the Predictor (Policy Network). Encoder + DINOv2 frozen.
    optimizer = optim.AdamW(model.predictor.parameters(), lr=lr)
    dinov2.eval()
    model.eval()
    model.predictor.train()  # dropout active for sampling diversity
    
    for epoch in range(epochs):
        total_reward = 0.0
        
        for batch in dataloader:
            start_imgs, target_imgs, action_names = batch
            start_imgs = start_imgs.to(device)
            target_imgs = target_imgs.to(device)
            
            # --- Feature extraction (DINOv2) then CORTEX encoding ---
            with torch.no_grad():
                start_features = dinov2(start_imgs)   # (batch, 384)
                target_features = dinov2(target_imgs) # (batch, 384)
                start_emb = model.encode(start_features)[0]   # (batch, 128)
                target_emb = model.encode(target_features)[0] # (batch, 128)
            
            # --- RL Step (Reward-Weighted Regression) ---
            optimizer.zero_grad()
            
            num_samples = 5
            
            # Replicate for sampling: each input gets num_samples hypotheses
            start_rep = start_emb.repeat_interleave(num_samples, dim=0)
            target_rep = target_emb.repeat_interleave(num_samples, dim=0)
            
            # Map string action names → integer IDs, then replicate to match samples
            action_ids = torch.tensor(
                [ACTION_MAP.get(a, 0) for a in action_names],
                dtype=torch.long, device=device
            )
            action_id_tensor = action_ids.repeat_interleave(num_samples)  # (batch*samples,)
            
            # Forward pass — predictor generates hypotheses
            pred_emb, _ = model.predictor(start_rep, transform_id=action_id_tensor)
            
            # --- Compute Rewards ---
            integrity = model.predictor._evaluate_structural_integrity(pred_emb)
            rewards = compute_reward(pred_emb, target_rep, integrity)
            
            # --- Reward-Weighted Regression Loss ---
            # Normalize rewards to [0, 1] within each original sample's group
            # so weighting is relative. High reward → low loss weight (we want to
            # KEEP good predictions, not push them away).
            rewards_grouped = rewards.view(-1, num_samples)           # (batch, samples)
            baseline = rewards_grouped.mean(dim=1, keepdim=True)      # per-sample baseline
            advantage = rewards_grouped - baseline                    # (batch, samples)
            advantage_flat = advantage.view(-1)                       # (batch*samples,)
            
            # Normalize advantage to stable [0,1] weights via sigmoid
            weights = torch.sigmoid(advantage_flat)   # positive advantage → weight > 0.5
            
            # MSE per hypothesis
            mse = F.mse_loss(pred_emb, target_rep, reduction='none').mean(dim=1)  # (batch*samples,)
            
            # Weight MSE by reward: GOOD hypotheses (high weight) contribute more gradient
            # toward their target. Bad hypotheses are down-weighted.
            loss = (mse * weights).mean()
            
            loss.backward()
            optimizer.step()
            
            total_reward += rewards.mean().item()
            
        print(f"  Epoch {epoch+1}: Avg Reward = {total_reward / len(dataloader):.4f}")
        
    print("✅ RL Fine-Tuning Complete.")