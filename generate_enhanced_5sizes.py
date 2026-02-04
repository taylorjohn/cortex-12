"""
CORTEX-12: Enhanced Data Generation with 5 Size Classes
Creates training data with:
- 5 size classes (tiny, small, medium, large, huge)
- Clear size boundaries (no overlap)
- Optional size references
- Better within-class variation
"""

import os
import json
import random
from PIL import Image, ImageDraw
import argparse
from pathlib import Path

random.seed(42)

# 5 SIZE CLASSES with NO OVERLAP
SIZE_CLASSES = {
    'tiny':   {'min': 20, 'max': 30},   # 20-30 pixels
    'small':  {'min': 35, 'max': 48},   # 35-48 pixels (5px gap)
    'medium': {'min': 53, 'max': 68},   # 53-68 pixels (5px gap)
    'large':  {'min': 73, 'max': 88},   # 73-88 pixels (5px gap)
    'huge':   {'min': 93, 'max': 108}   # 93-108 pixels (5px gap)
}

COLORS = {
    'red':     (255, 0, 0),
    'blue':    (0, 0, 255),
    'green':   (0, 255, 0),
    'yellow':  (255, 255, 0),
    'magenta': (255, 0, 255),
    'cyan':    (0, 255, 255),
    'orange':  (255, 165, 0),
    'purple':  (128, 0, 128),
    'white':   (255, 255, 255),
    'gray':    (192, 192, 192),
    'brown':   (165, 42, 42),
    'pink':    (255, 192, 203)
}

SHAPES = ['circle', 'square', 'triangle', 'hexagon', 'star', 'cross']

MATERIALS = ['matte', 'glossy', 'metallic', 'transparent', 'textured']

ORIENTATIONS = ['0', '45', '90']


def draw_circle(draw, center, radius, color):
    """Draw a circle"""
    x, y = center
    draw.ellipse(
        [x - radius, y - radius, x + radius, y + radius],
        fill=color,
        outline=None
    )


def draw_square(draw, center, size, color, rotation=0):
    """Draw a square"""
    x, y = center
    half = size // 2
    draw.rectangle(
        [x - half, y - half, x + half, y + half],
        fill=color,
        outline=None
    )


def draw_triangle(draw, center, size, color):
    """Draw an equilateral triangle"""
    x, y = center
    height = int(size * 0.866)  # sqrt(3)/2
    points = [
        (x, y - height // 2),
        (x - size // 2, y + height // 2),
        (x + size // 2, y + height // 2)
    ]
    draw.polygon(points, fill=color, outline=None)


def draw_hexagon(draw, center, size, color):
    """Draw a hexagon"""
    x, y = center
    r = size // 2
    points = []
    for i in range(6):
        angle = i * 60
        px = x + int(r * cos(radians(angle)))
        py = y + int(r * sin(radians(angle)))
        points.append((px, py))
    draw.polygon(points, fill=color, outline=None)


def draw_star(draw, center, size, color):
    """Draw a 5-pointed star"""
    import math
    x, y = center
    outer_r = size // 2
    inner_r = size // 4
    points = []
    for i in range(10):
        angle = i * 36 - 90
        r = outer_r if i % 2 == 0 else inner_r
        px = x + int(r * math.cos(math.radians(angle)))
        py = y + int(r * math.sin(math.radians(angle)))
        points.append((px, py))
    draw.polygon(points, fill=color, outline=None)


def draw_cross(draw, center, size, color):
    """Draw a cross/plus"""
    x, y = center
    width = size // 3
    # Vertical bar
    draw.rectangle(
        [x - width // 2, y - size // 2, x + width // 2, y + size // 2],
        fill=color
    )
    # Horizontal bar
    draw.rectangle(
        [x - size // 2, y - width // 2, x + size // 2, y + width // 2],
        fill=color
    )


def draw_shape(draw, shape, center, size, color):
    """Draw any shape"""
    if shape == 'circle':
        draw_circle(draw, center, size // 2, color)
    elif shape == 'square':
        draw_square(draw, center, size, color)
    elif shape == 'triangle':
        draw_triangle(draw, center, size, color)
    elif shape == 'hexagon':
        draw_hexagon(draw, center, size, color)
    elif shape == 'star':
        draw_star(draw, center, size, color)
    elif shape == 'cross':
        draw_cross(draw, center, size, color)


def create_enhanced_image(
    shape, 
    color_name, 
    size_class, 
    material='matte', 
    orientation='0',
    add_reference=True,
    image_size=224,
    background=(128, 128, 128)
):
    """
    Create enhanced training image with:
    - Clear size boundaries
    - Optional reference object
    - Random variation within class
    """
    
    img = Image.new('RGB', (image_size, image_size), background)
    draw = ImageDraw.Draw(img)
    
    # Get random size within class boundaries
    size_info = SIZE_CLASSES[size_class]
    actual_size = random.randint(size_info['min'], size_info['max'])
    
    # Random position (some variation, but centered)
    center_x = random.randint(image_size // 3, 2 * image_size // 3)
    center_y = random.randint(image_size // 3, 2 * image_size // 3)
    center = (center_x, center_y)
    
    # Get color
    color_rgb = COLORS[color_name]
    
    # Draw main shape
    draw_shape(draw, shape, center, actual_size, color_rgb)
    
    # Add size reference (small fixed-size circle in corner)
    if add_reference:
        ref_size = 10  # Always 10 pixels
        ref_color = (200, 200, 200)  # Light gray
        draw_circle(draw, (15, 15), ref_size, ref_color)
    
    return img


def generate_balanced_dataset(
    output_dir,
    samples_per_combination=10,
    add_reference=True
):
    """
    Generate balanced dataset with all combinations
    
    Args:
        output_dir: Where to save images
        samples_per_combination: Images per (shape, color, size) combo
        add_reference: Add size reference dot
    """
    
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    labels = {}
    count = 0
    
    print(f"Generating enhanced dataset with 5 size classes...")
    print(f"  Shapes: {len(SHAPES)}")
    print(f"  Colors: {len(COLORS)}")
    print(f"  Sizes: {len(SIZE_CLASSES)} (tiny, small, medium, large, huge)")
    print(f"  Materials: {len(MATERIALS)}")
    print(f"  Orientations: {len(ORIENTATIONS)}")
    print(f"  Samples per combination: {samples_per_combination}")
    print(f"  Reference dot: {add_reference}")
    print()
    
    # Size boundaries
    print("Size Boundaries (no overlap):")
    for size_name, info in SIZE_CLASSES.items():
        print(f"  {size_name:8s}: {info['min']:3d}-{info['max']:3d} pixels")
    print()
    
    total_combos = (len(SHAPES) * len(COLORS) * len(SIZE_CLASSES) * 
                   len(MATERIALS) * len(ORIENTATIONS))
    total_images = total_combos * samples_per_combination
    
    print(f"Total combinations: {total_combos:,}")
    print(f"Total images: {total_images:,}")
    print()
    
    for shape in SHAPES:
        for color in COLORS:
            for size in SIZE_CLASSES:
                for material in MATERIALS:
                    for orientation in ORIENTATIONS:
                        for sample in range(samples_per_combination):
                            
                            # Create image
                            img = create_enhanced_image(
                                shape, color, size, material, orientation,
                                add_reference=add_reference
                            )
                            
                            # Save
                            filename = f"{shape}_{color}_{size}_{material}_{orientation}_{sample}.png"
                            img.save(images_dir / filename)
                            
                            # Record label
                            labels[filename] = {
                                'shape': shape,
                                'color': color,
                                'size': size,
                                'material': material,
                                'orientation': orientation
                            }
                            
                            count += 1
                            if count % 1000 == 0:
                                print(f"  Generated {count:,}/{total_images:,} images...")
    
    # Save labels
    labels_path = output_dir / "labels_5sizes.json"
    with open(labels_path, 'w') as f:
        json.dump(labels, f, indent=2)
    
    print(f"\n✅ Dataset generation complete!")
    print(f"  Total images: {count:,}")
    print(f"  Images directory: {images_dir}")
    print(f"  Labels file: {labels_path}")
    
    # Statistics
    print(f"\nDataset Statistics:")
    print(f"  Images per size class: {count // len(SIZE_CLASSES):,}")
    print(f"  Images per shape: {count // len(SHAPES):,}")
    print(f"  Images per color: {count // len(COLORS):,}")
    
    return labels_path


def main():
    parser = argparse.ArgumentParser(
        description='Generate enhanced dataset with 5 size classes'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/enhanced_5sizes',
        help='Output directory for dataset'
    )
    parser.add_argument(
        '--samples_per_combo',
        type=int,
        default=10,
        help='Images per combination'
    )
    parser.add_argument(
        '--no_reference',
        action='store_true',
        help='Do not add size reference dot'
    )
    
    args = parser.parse_args()
    
    generate_balanced_dataset(
        args.output_dir,
        samples_per_combination=args.samples_per_combo,
        add_reference=not args.no_reference
    )


if __name__ == "__main__":
    # Need math imports
    from math import cos, sin, radians
    main()