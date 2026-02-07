"""
CORTEX-12 Phase 3 Main Runtime (FIXED)
✓ weights_only=True security fix
✓ Corrupted image error handling
✓ Centralized axis constants
✓ Syntax errors fixed (no spaces in identifiers/literals)
CPU-only, deterministic, production-ready
"""
import os
import json
import torch
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')

# Centralized constants
from constants import AXIS_LAYOUT

# Local modules
from cortex_adapter_v12 import CortexAdapter


class Cortex12Runtime:
    def __init__(self, checkpoint_path=None, memory_path="memory_vector_v12.json"):
        self.device = torch.device("cpu")
        
        # Load DINOv2 ViT-S/14 (backbone is NOT stored in checkpoint)
        print("Loading DINOv2 ViT-S/14 backbone (via torch.hub)...")
        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', trust_repo=True)
        self.backbone.eval()
        self.backbone.to(self.device)
        
        # Load adapter
        self.adapter = CortexAdapter()
        if checkpoint_path:
            # SECURITY FIX: weights_only=True prevents arbitrary code execution
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            # Handle both checkpoint formats
            if 'cortex_state_dict' in ckpt:
                self.adapter.load_state_dict(ckpt['cortex_state_dict'], strict=False)
            elif 'adapter_state_dict' in ckpt:
                self.adapter.load_state_dict(ckpt['adapter_state_dict'], strict=False)
            else:
                raise KeyError(f"Checkpoint missing weights. Keys: {list(ckpt.keys())}")
        self.adapter.eval()
        self.adapter.to(self.device)
        
        # Load explicit memory (handle legacy JSON with trailing spaces)
        with open(memory_path, 'r') as f:
            raw_memory = json.load(f)
        
        # Handle legacy format with trailing spaces in keys
        if "concepts " in raw_memory:
            self.memory = raw_memory["concepts "]
            # Strip trailing spaces from all concept keys/values
            self.memory = {
                k.strip(): {kk.strip(): (vv.strip() if isinstance(vv, str) else vv) 
                           for kk, vv in v.items()}
                for k, v in self.memory.items()
            }
        else:
            self.memory = raw_memory.get("concepts", raw_memory)
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def perceive(self, image_path):
        """Extract 128-D semantic embedding from image with corruption handling"""
        try:
            # ERROR HANDLING: Skip corrupted images gracefully
            img = Image.open(image_path).convert("RGB")
        except (UnidentifiedImageError, OSError) as e:
            print(f"⚠️ WARNING: Corrupted image '{image_path}': {e}")
            # Return neutral embedding (zeros) to avoid crashing pipeline
            return torch.zeros(128).numpy()
        
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
        clean_key = concept_key.strip()
        if clean_key not in self.memory:
            raise ValueError(f"Concept '{concept_key}' not in memory")
        # In full system, this would render using render_utils.create_object_image()
        return f"[IMAGINED: {clean_key}]"

    def get_concept_attributes(self, concept_key):
        """Return structured attributes for a known concept (handles trailing spaces)"""
        clean_key = concept_key.strip()
        return self.memory.get(clean_key, None)

    def get_axis_subspace(self, embedding, axis_name):
        """Extract subspace for a specific semantic axis"""
        if axis_name not in AXIS_LAYOUT:
            raise ValueError(f"Unknown axis: {axis_name}. Valid axes: {list(AXIS_LAYOUT.keys())}")
        start, end = AXIS_LAYOUT[axis_name]
        return embedding[start:end+1]


# Example usage
if __name__ == "__main__":
    runtime = Cortex12Runtime(
        checkpoint_path="runs/cortex_v13_supervised/cortex_v13_supervised_best.pt",
        memory_path="memory_vector_v12.json"
    )
    
    # Perceive
    emb = runtime.perceive("data/enhanced_5sizes/images/red_circle_small_0deg_matte_0_25_0_25.png")
    print(f"Embedding shape: {emb.shape}")  # (128,)
    
    # Query memory
    attrs = runtime.get_concept_attributes("red_circle_small_0deg_matte_0_25_0_25")
    print("Attributes:", attrs)
    
    # Extract color subspace
    color_emb = runtime.get_axis_subspace(emb, "color")
    print(f"Color subspace shape: {color_emb.shape}")  # (16,)