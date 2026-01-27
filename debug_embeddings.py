import os
import json
import torch
import torch.hub
from PIL import Image
from torchvision import transforms
import numpy as np
from cortex_adapter_v12 import CortexAdapter

# Load model
device = torch.device("cpu")
dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
dinov2.eval(); dinov2.to(device)
model = CortexAdapter()
ckpt = torch.load("runs/phase3/cortex_step_phase3_0050.pt", map_location=device, weights_only=True)
model.load_state_dict(ckpt['cortex_state_dict'], strict=False)
model.eval(); model.to(device)

def load_image(img_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(Image.open(img_path).convert("RGB")).unsqueeze(0)

# Use images that definitely exist
img1 = "data/curriculum/images/red_square_medium_0deg_matte_0_25_0_25.png"
img2 = "data/curriculum/images/chartreuse_circle_small_180deg_glossy_0_25_0_75.png"  # From your earlier output

with torch.no_grad():
    emb1 = model(dinov2(load_image(img1).to(device))).squeeze().cpu().numpy()
    emb2 = model(dinov2(load_image(img2).to(device))).squeeze().cpu().numpy()

print("Total embedding distance:", np.linalg.norm(emb1 - emb2))
print("Shape subspace distance:", np.linalg.norm(emb1[0:32] - emb2[0:32]))
print("Color subspace distance:", np.linalg.norm(emb1[64:80] - emb2[64:80]))