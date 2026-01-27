"""
CORTEX-12 Phase 3 Main Runtime
- Loads DINOv2 backbone via torch.hub (external)
- Uses lightweight CortexAdapter for 128-D semantic projection
- Supports compositional imagination and explicit memory
- CPU-only, deterministic
"""

import os
import json
import torch
from torchvision import transforms
from PIL import Image

# Local modules
from cortex_adapter_v12 import CortexAdapter


class Cortex12Runtime:
    def __init__(self, checkpoint_path=None, memory_path="memory_vector_v12.json"):
        self.device = torch.device("cpu")
        
        # Load DINOv2 ViT-S/14 (backbone is NOT stored in checkpoint)
        print("Loading DINOv2 ViT-S/14 backbone (via torch.hub)...")
        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        self.backbone.eval()
        self.backbone.to(self.device)
        
        # Load adapter
        self.adapter = CortexAdapter()
        if checkpoint_path:
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            self.adapter.load_state_dict(ckpt['cortex_state_dict'], strict=False)
        self.adapter.eval()
        self.adapter.to(self.device)
        
        # Load explicit memory
        with open(memory_path, 'r') as f:
            self.memory = json.load(f)
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def perceive(self, image_path):
        """Extract 128-D semantic embedding from image."""
        img = Image.open(image_path).convert("RGB")
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.backbone(tensor)
            embedding = self.adapter(features)
        return embedding.squeeze().cpu().numpy()  # [128]

    def imagine(self, concept_key):
        """
        Generate a synthetic image from memory (for demo/curriculum).
        In real use, this would call a renderer (e.g., render_utils.py).
        For now, returns a placeholder path.
        """
        if concept_key not in self.memory:
            raise ValueError(f"Concept '{concept_key}' not in memory")
        # In full system, this would render using render_utils.create_object_image()
        return f"[IMAGINED: {concept_key}]"

    def get_concept_attributes(self, concept_key):
        """Return structured attributes for a known concept."""
        return self.memory.get(concept_key, None)


# Example usage
if __name__ == "__main__":
    runtime = Cortex12Runtime(
        checkpoint_path="runs/phase3/cortex_step_phase3_0050.pt",
        memory_path="memory_vector_v12.json"
    )
    
    # Perceive
    emb = runtime.perceive("data/curriculum/images/red_square_medium_0deg_matte_0_25_0_25.png")
    print(f"Embedding shape: {emb.shape}")  # (128,)
    
    # Query memory
    attrs = runtime.get_concept_attributes("red_square_medium_0deg_matte_0_25_0_25")
    print("Attributes:", attrs)