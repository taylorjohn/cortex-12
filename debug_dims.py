import torch
from cortex_adapter_v12 import CortexAdapter

model = CortexAdapter()
dummy_input = torch.randn(1, 384)
output = model(dummy_input)
print("Total embedding dim:", output.shape[1])  # Should be 128

# Print axis dims
shape = model.shape_proj(dummy_input)
size = model.size_proj(dummy_input)
material = model.material_proj(dummy_input)
color = model.color_proj(dummy_input)
location = model.location_proj(dummy_input)
orientation = model.orientation_proj(dummy_input)

print("Shape dim:", shape.shape[1])      # 32
print("Size dim:", size.shape[1])        # 16
print("Material dim:", material.shape[1]) # 16
print("Color dim:", color.shape[1])      # 16
print("Location dim:", location.shape[1]) # 8
print("Orientation dim:", orientation.shape[1]) # 16