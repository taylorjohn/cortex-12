"""
CortexAdapter for CORTEX-12 Phase 3
- Projects DINOv2 features into 128-D semantic space
- Enforces fixed axis layout for 6 attributes
- Lightweight (~680 KB when saved)
"""

import torch
import torch.nn as nn


class CortexAdapter(nn.Module):
    def __init__(self, input_dim=384):
        super().__init__()
        # Input: DINOv2 ViT-S/14 [CLS] token (384-D)
        
        # Semantic heads (total = 128D)
        self.shape_proj = nn.Linear(input_dim, 32)      # dims 0-31
        self.size_proj = nn.Linear(input_dim, 16)       # dims 32-47
        self.material_proj = nn.Linear(input_dim, 16)   # dims 48-63
        self.color_proj = nn.Linear(input_dim, 16)      # dims 64-79
        self.location_proj = nn.Linear(input_dim, 8)    # dims 80-87
        self.orientation_proj = nn.Linear(input_dim, 16) # dims 88-103
        # dims 104-127: reserved (16D)

    def forward(self, x):
        """
        x: [B, 384] DINOv2 features
        Returns: [B, 128] semantic embedding
        """
        shape = self.shape_proj(x)
        size = self.size_proj(x)
        material = self.material_proj(x)
        color = self.color_proj(x)
        location = self.location_proj(x)
        orientation = self.orientation_proj(x)
        
        # Concatenate into fixed layout
        embedding = torch.cat([
            shape,        # 32
            size,         # 16 → 48
            material,     # 16 → 64
            color,        # 16 → 80
            location,     # 8  → 88
            orientation,  # 16 → 104
            torch.zeros(x.size(0), 24, device=x.device)  # reserved (128 - 104 = 24)
        ], dim=1)
        
        return embedding  # [B, 128]