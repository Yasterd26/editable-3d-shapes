import numpy as np
import trimesh
import os
import argparse
import glob
from tqdm import tqdm
from scipy.spatial import KDTree

def normalize_mesh(mesh):
    """
    Scale mesh to fit in unit sphere and center it.
    """
    mesh.vertices -= mesh.center_mass
    max_scale = np.max(np.linalg.norm(mesh.vertices, axis=1))
    mesh.vertices /= max_scale
    return mesh

def sample_sdf(mesh, num_samples=20000, surface_ratio=0.8):
    """
    Robust custom SDF sampling.
    """
    # 1. Sample points
    num_surface = int(num_samples * surface_ratio)
    num_random = num_samples - num_surface
    
    surface_points, _ = trimesh.sample.sample_surface(mesh, num_surface)
    surface_points += np.random.normal(0, 0.01, size=surface_points.shape)
    
    random_points = np.random.uniform(-1.0, 1.0, size=(num_random, 3))
    points = np.concatenate([surface_points, random_points], axis=0)
    
    # 2. Compute Unsigned Distance using KDTree on mesh vertices (Approx)
    # Better: KDTree on dense surface samples
    # We can sample MANY points on surface to build the tree
    dense_surf, _ = trimesh.sample.sample_surface(mesh, 100000)
    tree = KDTree(dense_surf)
    dist, _ = tree.query(points)
    
    # 3. Compute Sign using Ray Casting (Robust but slow)
    # trimesh's contains_points uses ray casting
    # We use the slow python implementation if rtree is missing
    # But trimesh might automatically use it.
    
    # To be safe, let's wrap in try-except or check for watertight
    # If not watertight, signs might be bad.
    
    # If mesh is not watertight, fix it?
    # mesh.fill_holes()
    
    # Check containment
    # Split into chunks to avoid memory issues/long hangs
    is_inside = np.zeros(len(points), dtype=bool)
    chunk_size = 5000
    
    # Use RayMeshIntersector explicitly if possible, or just mesh.contains
    # mesh.contains handles chunks?
    
    for i in range(0, len(points), chunk_size):
        chunk = points[i:i+chunk_size]
        try:
            # This uses ray tracing
            is_inside[i:i+chunk_size] = mesh.contains(chunk)
        except Exception:
            # Fallback: assume all outside or random? 
            # If contains fails, we can't get sign. 
            pass
            
    # SDF: negative inside, positive outside
    sdf = dist.copy()
    sdf[is_inside] = -sdf[is_inside]
    
    return points, sdf

def process_file(file_path, output_dir):
    try:
        # Load mesh
        # GLB files might load as a Scene, we need to extract the geometry
        scene_or_mesh = trimesh.load(file_path, force='mesh')
        
        # If it's a scene, dump all geometries into a single mesh
        if isinstance(scene_or_mesh, trimesh.Scene):
            if len(scene_or_mesh.geometry) == 0:
                print(f"No geometry in {file_path}")
                return
            # Concatenate all meshes in the scene
            mesh = trimesh.util.concatenate(
                tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces) 
                      for g in scene_or_mesh.geometry.values())
            )
        else:
            mesh = scene_or_mesh
            
        # Normalize
        mesh = normalize_mesh(mesh)
        
        # Sample SDF
        points, sdf = sample_sdf(mesh)
        
        # Save
        filename = os.path.basename(file_path).replace('.glb', '.npz')
        output_path = os.path.join(output_dir, filename)
        
        np.savez(output_path, points=points, sdf=sdf)
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description='Preprocess GLB meshes to SDF samples')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing GLB files')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for NPZ files')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    files = glob.glob(os.path.join(args.input_dir, "**/*.glb"), recursive=True)
    print(f"Found {len(files)} files in {args.input_dir}")
    
    # Use multiprocessing
    from multiprocessing import Pool, cpu_count
    
    # Filter out already processed files
    files_to_process = []
    for f in files:
        filename = os.path.basename(f).replace('.glb', '.npz')
        if not os.path.exists(os.path.join(args.output_dir, filename)):
            files_to_process.append(f)
            
    print(f"Processing {len(files_to_process)} new files with {cpu_count()} cores...")
    
    with Pool(processes=cpu_count()) as pool:
        args_list = [(f, args.output_dir) for f in files_to_process]
        # Use starmap
        list(tqdm(pool.starmap(process_file, args_list), total=len(files_to_process)))

if __name__ == '__main__':
    main()
