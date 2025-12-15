import trimesh
import os

INPUT_DIR = "/Users/divyanshgarg/Documents/Python/DSG /editable-3d-shapes/data/mugs"
OUTPUT_DIR = "/Users/divyanshgarg/Documents/Python/DSG /editable-3d-shapes/data/mugs_normalized"

os.makedirs(OUTPUT_DIR, exist_ok=True)

for fname in os.listdir(INPUT_DIR):
    if not fname.endswith(".obj"):
        continue
    mesh = trimesh.load(os.path.join(INPUT_DIR, fname))
    mesh.apply_translation(-mesh.centroid)
    mesh.apply_scale(1.0 / max(mesh.extents))
    mesh.export(os.path.join(OUTPUT_DIR, fname))
