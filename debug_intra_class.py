import os
import json
import torch
import torch.hub
from PIL import Image
from torchvision import transforms
import numpy as np
from cortex_adapter_v12 import CortexAdapter

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

# Find all "square" images
image_dir = "data/curriculum/images"
square_images = [f for f in os.listdir(image_dir) if "square" in f][:3]  # First 3 squares

embeddings = []
with torch.no_grad():
    for img_name in square_images:
        img_path = os.path.join(image_dir, img_name)
        emb = model(dinov2(load_image(img_path).to(device))).squeeze().cpu().numpy()
        embeddings.append(emb[0:32])  # Shape subspace

embeddings = np.array(embeddings)
centroid = np.mean(embeddings, axis=0)
distances = [np.linalg.norm(emb - centroid) for emb in embeddings]

print("Intra-class shape distances for 'square':", distances)
print("Max intra-class distance:", max(distances))