# tools/generate_certification_data.py
"""
Generate synthetic dataset for semantic axis certification.
Produces 64x64 RGB images of colored shapes with ground-truth labels.
"""

import os
import numpy as np
from PIL import Image, ImageDraw
import json

OUTPUT_DIR = "cert_data"
NUM_SAMPLES = 1200
IMAGE_SIZE = 64
SEED = 42

COLORS = {
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "purple": (128, 0, 128),
    "cyan": (0, 255, 255)
}

SHAPES = ["circle", "square", "triangle"]
SIZES = {"small": 8, "medium": 14, "large": 20}

np.random.seed(SEED)

def draw_shape(draw, shape, center, size, color):
    x, y = center
    if shape == "circle":
        draw.ellipse((x-size, y-size, x+size, y+size), fill=color)
    elif shape == "square":
        draw.rectangle((x-size, y-size, x+size, y+size), fill=color)
    elif shape == "triangle":
        points = [(x, y-size), (x-size, y+size), (x+size, y+size)]
        draw.polygon(points, fill=color)

def generate_sample():
    color_name = np.random.choice(list(COLORS.keys()))
    shape = np.random.choice(SHAPES)
    size_name = np.random.choice(list(SIZES.keys()))
    size_val = SIZES[size_name]
    circularity = 1.0 if shape == "circle" else 0.0 if shape == "square" else 0.3
    
    img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    margin = 24
    center = (
        np.random.randint(margin, IMAGE_SIZE - margin),
        np.random.randint(margin, IMAGE_SIZE - margin)
    )
    draw_shape(draw, shape, center, size_val, COLORS[color_name])
    
    labels = {
        "hue": color_name,
        "shape": shape,
        "size_scalar": float(size_val) / max(SIZES.values()),
        "shape_circularity": circularity
    }
    return np.array(img), labels

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    metadata = []
    for i in range(NUM_SAMPLES):
        img, labels = generate_sample()
        path = os.path.join(OUTPUT_DIR, f"sample_{i:04d}.png")
        Image.fromarray(img).save(path)
        metadata.append({"image": os.path.basename(path), "labels": labels})
    with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ Generated {NUM_SAMPLES} samples in '{OUTPUT_DIR}'")

if __name__ == "__main__":
    main()