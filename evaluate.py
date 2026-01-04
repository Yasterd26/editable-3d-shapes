import torch
import numpy as np
import trimesh
from skimage import measure
import os
from models.deep_sdf import DeepSDF
import argparse
from tqdm import tqdm
from scipy.spatial import KDTree

def create_mesh(model, latent, resolution=64, max_batch=128**3):
    """
    Generate a mesh from the DeepSDF model for a given latent code.
    """
    model.eval()
    
    # Create a grid
    overall_index = np.arange(0, resolution ** 3, 1, dtype=np.int64)
    samples = np.zeros([resolution ** 3, 4], dtype=np.float32) # (x, y, z, val)
    
    # Transform index to (x, y, z) coordinates in [-1, 1]
    # (0, 0, 0) -> (-1, -1, -1)
    # (res-1, res-1, res-1) -> (1, 1, 1)
    
    # stride = 2 / (resolution - 1)
    
    # Vectorized grid generation
    x = np.linspace(-1, 1, resolution)
    y = np.linspace(-1, 1, resolution)
    z = np.linspace(-1, 1, resolution)
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    points = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3) # (N, 3)
    
    points_tensor = torch.from_numpy(points).float().to(next(model.parameters()).device)
    latent_tensor = latent.unsqueeze(0).to(next(model.parameters()).device) # (1, latent_dim)
    
    sdf_values = []
    
    with torch.no_grad():
        # Process in batches to avoid OOM
        head = 0
        while head < points.shape[0]:
            tail = min(head + max_batch, points.shape[0])
            batch_points = points_tensor[head:tail]
            
            # Forward
            # (1, N_batch, 3) and (1, latent_dim)
            # model forward expects (B, N, 3) or (B, 3)
            # here B=1, N=batch_size
            pred = model(batch_points.unsqueeze(0), latent_tensor) # (1, N_batch, 1)
            sdf_values.append(pred.squeeze().cpu().numpy())
            
            head += max_batch
            
    sdf_values = np.concatenate(sdf_values)
    print(f"Predicted SDF Range: {np.min(sdf_values):.4f} to {np.max(sdf_values):.4f}")
    
    sdf_grid = sdf_values.reshape(resolution, resolution, resolution)
    
    # Marching Cubes
    # Only if there is a zero crossing
    # Marching Cubes
    level = 0.0
    if np.min(sdf_grid) > 0:
        print(f"Warning: Surface level not within SDF range (Min: {np.min(sdf_grid):.4f}). attempting fallback.")
        if np.min(sdf_grid) < 0.05:
            # Fallback: render the "core" of the shape
            level = np.min(sdf_grid) + 0.0001
        else:
            return None
    elif np.max(sdf_grid) < 0:
        return None

        
    verts, faces, normals, values = measure.marching_cubes(sdf_grid, level=level)
    
    # Scale verts back to [-1, 1]
    # verts are in [0, resolution-1]
    verts = verts / (resolution - 1) # [0, 1]
    verts = verts * 2 - 1 # [-1, 1]
    
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    
    # CRITICAL FIX: Remove disconnected fragments, keep only largest component
    # This fixes the "broken pieces" issue
    if not mesh.is_empty:
        components = mesh.split(only_watertight=False)
        if len(components) > 1:
            # Keep only the largest component by vertex count
            mesh = max(components, key=lambda m: len(m.vertices))
    
    # Apply smoothing to reduce blocky artifacts (1 iteration for speed)
    try:
        mesh = trimesh.smoothing.filter_laplacian(mesh, iterations=1)
    except Exception as e:
        print(f"Smoothing failed: {e}")  # Debug output
    
    return mesh

def compute_chamfer_distance(mesh_pred, mesh_gt, num_samples=30000):
    """
    Compute Chamfer Distance between two meshes.
    CD = mean(dist(pred, gt)) + mean(dist(gt, pred))
    """
    # Sample points
    pts_pred, _ = trimesh.sample.sample_surface(mesh_pred, num_samples)
    pts_gt, _ = trimesh.sample.sample_surface(mesh_gt, num_samples)
    
    # KDTree for nearest neighbor
    tree_pred = KDTree(pts_pred)
    tree_gt = KDTree(pts_gt)
    
    dist_pred_to_gt, _ = tree_gt.query(pts_pred) # for each pred point, dist to nearest gt
    dist_gt_to_pred, _ = tree_pred.query(pts_gt) # for each gt point, dist to nearest pred
    
    chamfer_dist = np.mean(dist_pred_to_gt) + np.mean(dist_gt_to_pred)
    return chamfer_dist

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--full_eval', action='store_true', help='Evaluate on a set of shapes')
    parser.add_argument('--output_dir', type=str, default='results/reconstructions')
    parser.add_argument('--resolution', type=int, default=64)
    parser.add_argument('--latent_dim', type=int, default=128)
    
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    # Load Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    model = DeepSDF(args.latent_dim, hidden_dim=256, num_layers=8).to(device) # Ensure dims match training
    model.load_state_dict(checkpoint['model_state_dict'])
    
    latents = checkpoint['latents'].to(device) # (Num_shapes, Latent_dim)
    
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    
    # Just reconstruct the first few shapes as a test
    for idx_to_recon in range(min(5, len(latents))):
        print(f"Reconstructing shape (idx {idx_to_recon})...")
        latent_vec = latents[idx_to_recon]
        
        mesh = create_mesh(model, latent_vec, resolution=args.resolution)
        if mesh:
            out_name = os.path.join(args.output_dir, f'reconstruction_{idx_to_recon}.glb')
            mesh.export(out_name)
            print(f"Saved {out_name}")
        else:
            print(f"Failed to reconstruct mesh {idx_to_recon} (no zero level set).")

if __name__ == '__main__':
    main()
