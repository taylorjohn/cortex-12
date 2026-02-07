"""
CORTEX-12 Semantic Constants
Single source of truth for axis layouts and dimensions
Prevents drift between training/certification/runtime
"""
# Semantic axis layout (fixed 128-D embedding)
AXIS_LAYOUT = {
    "shape": (0, 31),        # 32 dimensions
    "size": (32, 47),        # 16 dimensions
    "material": (48, 63),    # 16 dimensions
    "color": (64, 79),       # 16 dimensions
    "location": (80, 87),    # 8 dimensions
    "orientation": (88, 103) # 16 dimensions
    # dims 104-127: reserved (24 dimensions)
}

# Size vocabulary mappings (handle both naming conventions)
SIZE_VOCAB = {
    'tiny': 'extra-small',
    'small': 'small',
    'medium': 'medium',
    'large': 'large',
    'huge': 'extra-large',
    'extra-small': 'extra-small',
    'extra-large': 'extra-large'
}

VALID_SIZES = set(SIZE_VOCAB.keys())
VALID_COLORS = {'red', 'blue', 'green', 'yellow', 'magenta', 'cyan', 'orange', 'purple', 'brown', 'pink', 'gray', 'black'}
VALID_SHAPES = {'circle', 'square', 'triangle', 'rectangle', 'star', 'hexagon', 'cross', 'diamond'}