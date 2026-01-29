# tools/correct_merge.py
import json

# Load original labels
with open("data/balanced_images/labels.json", "r") as f:
    labels = json.load(f)

# Convert ALL orientation values to strings and merge 180→0
fixed_count = 0
for img_name, attrs in labels.items():
    if "orientation" in attrs:
        # Convert to string if it's an integer
        orient_str = str(attrs["orientation"])
        # Merge 180° → 0°
        if orient_str == "180":
            attrs["orientation"] = "0"
            fixed_count += 1
        else:
            attrs["orientation"] = orient_str

# Save corrected merged labels
with open("data/balanced_images/labels_merged.json", "w") as f:
    json.dump(labels, f, indent=2)

print(f"✅ Fixed {fixed_count} orientation labels")
print("✅ All orientations are now strings with 180° merged to 0°")