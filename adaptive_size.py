"""
CORTEX-12 v14: Adaptive Size Regression
Implements 'Inference-Time Compute' for ambiguous size inputs.

If the model is 'confused' (high entropy), it runs multiple passes
with slight noise (Test-Time Augmentation) to refine the estimate.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ordinal_size_regression import HybridSizeHead

class AdaptiveSizeHead(HybridSizeHead):
    """
    Smart Size Head that uses extra compute for hard examples.
    """
    
    def predict_with_refinement(self, x, refinement_steps=5, noise_scale=0.01):
        """
        Adaptive prediction loop.
        
        Args:
            x: (batch, input_dim)
            refinement_steps: Number of 'thoughts' to average if confused
            noise_scale: Magnitude of thought noise
            
        Returns:
            final_logits: Refined threshold logits
            uncertainty_mask: Boolean mask of which items needed refinement
        """
        # 1. Fast Pass (System 1)
        logits, class_probs, continuous = self.forward(x)
        
        # 2. Check Confidence (Entropy)
        # High entropy = I don't know = Need to think
        entropy = -(class_probs * torch.log(class_probs + 1e-8)).sum(dim=1)
        
        # Threshold for "Confusion" (e.g., > 1.0 means distribution is flat)
        # Max entropy for 5 classes is ln(5) ≈ 1.6
        confusion_mask = entropy > 0.8 
        
        if not confusion_mask.any():
            return logits, confusion_mask
            
        # 3. Slow Pass (System 2) - Only for confused items
        # print(f"  🤔 Refining {confusion_mask.sum().item()} ambiguous size predictions...")
        
        indices = torch.nonzero(confusion_mask).squeeze()
        if indices.ndim == 0: indices = indices.unsqueeze(0)
        
        confused_inputs = x[indices]
        accumulated_logits = torch.zeros_like(logits[indices])
        
        # Run multiple noisy views ("looking closer")
        for _ in range(refinement_steps):
            noise = torch.randn_like(confused_inputs) * noise_scale
            noisy_logits, _, _ = self.forward(confused_inputs + noise)
            accumulated_logits += noisy_logits
            
        # Average the thoughts
        refined_logits = accumulated_logits / refinement_steps
        
        # Update original logits with refined ones
        # We need to clone to avoid in-place modification errors if gradients needed
        final_logits = logits.clone()
        final_logits[indices] = refined_logits
        
        return final_logits, confusion_mask