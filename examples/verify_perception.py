# examples/verify_perception.py
"""
Demo: Verifiable perception with CORTEX-12.
"""

from cortex12 import CORTEX12
from cortex12.semantic_axes import probe, validate_embedding
import numpy as np
from PIL import Image

model = CORTEX12()
img = np.array(Image.open("cert_data/sample_0042.png"))
embedding = model.encode(img)

# Interpret
print("Color:", probe(embedding, "hue"))
print("Circularity:", f"{probe(embedding, 'shape_circularity'):.2f}")
print("Size (norm):", f"{probe(embedding, 'size'):.2f}")

# Validate safety conditions
valid = validate_embedding(embedding, {
    "hue": "red",
    "shape_circularity": ">0.8"
})
print("Validation passed:", valid)