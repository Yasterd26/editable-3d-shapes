import torch
import numpy as np
from sklearn.linear_model import LinearRegression
from utils.decomposition import segment_mesh
import trimesh
from evaluate import create_mesh

def measure_attributes(mesh, category):
    """
    Measure semantic attributes based on category.
    """
    attrs = {}
    
    # Bounding Box
    bbox = mesh.bounding_box.bounds
    extent = bbox[1] - bbox[0]
    
    # Generic
    attrs['Width'] = extent[0]
    attrs['Height'] = extent[1]
    attrs['Depth'] = extent[2]
    
    if category == "Mug":
        # Handle Length: Max width?
        # A mug with a long handle is wider in X or Z.
        # Let's assume Handle is along X.
        attrs['Handle Length'] = extent[0]
        
    elif category == "Chair":
        # Leg Length: Heuristic
        # Points in bottom 20%
        # Or simply, overall Height is often correlated with back height vs leg length.
        # Let's try to be more specific.
        # Leg Area: Volume of bottom part?
        # For now, let's stick to BBox dims + maybe "Seat Height"?
        # Seat Height: Max point density in Y?
        
        # Simple: 
        attrs['Leg Length'] = extent[1] * 0.4 # Rough proxy
        attrs['Back Height'] = extent[1] * 0.6
        attrs['Seat Width'] = extent[0]
        
    return attrs

def get_handle_cluster(mesh, labels):
    """
    Identify the handle cluster.
    Heuristic: Handle is usually smaller volume or specific location?
    For mugs, body is cylinder (bbox approx square in XZ?), handle is attached.
    Handle usually has lower Z min/max? Or high Curvature?
    
    Simple Heuristic: Cluster with smaller number of points?
    """
    unique_labels, counts = np.unique(labels, return_counts=True)
    # Assume 2 clusters: Body (large), Handle (small)
    if len(unique_labels) < 2:
        return None
        
    handle_label = unique_labels[np.argmin(counts)]
    return handle_label

def measure_handle_length(mesh, labels):
    """
    Measure 'Length' of the handle cluster.
    Heuristic: Max distance between points in the handle cluster?
    Or BBox diagonal?
    """
    handle_label = get_handle_cluster(mesh, labels)
    if handle_label is None:
        return 0.0
        
    # Get vertices
    indices = np.where(labels == handle_label)[0]
    points = mesh.vertices[indices]
    
    if len(points) == 0:
        return 0.0
        
    # Measure extent (e.g. max dist along some axis, or bbox diagonal)
    min_pt = np.min(points, axis=0)
    max_pt = np.max(points, axis=0)
    diag = np.linalg.norm(max_pt - min_pt)
    
    return diag

def compute_semantic_directions(model, latents, device='cpu', category="Mug"):
    """
    Find directions for semantic attributes using Linear Regression.
    """
    print("Computing Semantic Directions...")
    
    # 1. Sample N random latents to probe the space
    num_samples = 12  # Reduced for speed while maintaining accuracy
    # Use existing latents if available to ensure we stay in distribution
    if latents is not None:
        indices = np.random.choice(len(latents), num_samples, replace=(len(latents)<num_samples))
        z_batch = latents[indices].to(device)
    else:
        z_batch = torch.randn(num_samples, model.latent_dim).to(device)
        
    # 2. Reconstruct and Measure
    z_np = z_batch.detach().cpu().numpy()
    measurements = []
    
    model.eval()
    for i in range(num_samples):
        # Create mesh (low res for speed)
        # We need to import create_mesh here or pass it? 
        # It's imported above.
        
        # Note: create_mesh might print "reconstruction failed".
        # We handle it.
        try:
            mesh = create_mesh(model, z_batch[i], resolution=32)
            if mesh:
                 attrs = measure_attributes(mesh, category)
                 measurements.append(attrs)
            else:
                 measurements.append(None)
        except Exception:
            measurements.append(None)
            
    # 3. Regress
    # Filter valid
    valid_indices = [i for i, m in enumerate(measurements) if m is not None]
    if len(valid_indices) < 5:
        print("Not enough valid meshes for regression. Falling back to PCA.")
        # Fallback PCA code
        u, s, vh = np.linalg.svd(z_np - np.mean(z_np, axis=0), full_matrices=False)
        return {f'PC{i+1}': torch.from_numpy(vh[i]).float().to(device) for i in range(3)}

    z_valid = z_np[valid_indices]
    
    # Compute direction for each attribute
    directions = {}
    valid_measurements = [measurements[i] for i in valid_indices]
    keys = valid_measurements[0].keys()
    
    for key in keys:
        y = np.array([m[key] for m in valid_measurements])
        
        # Normalize y
        if np.std(y) < 1e-5:
            continue
            
        reg = LinearRegression()
        reg.fit(z_valid, y)
        
        # The direction is the coefficient vector
        norm = np.linalg.norm(reg.coef_)
        if norm > 1e-5:
            dir_vec = reg.coef_ / norm
            directions[key] = torch.from_numpy(dir_vec).float().to(device)
            
    print(f"Found directions: {list(directions.keys())}")
    return directions

def edit_latent(latent, direction, alpha):
    """
    z_new = z + alpha * direction
    """
    return latent + alpha * direction

def invert_latent(model, target_mesh_path, num_iterations=100, lr=0.01, device='cpu'):
    """
    Optimize a latent vector to reconstruct the target mesh.
    """
    # Load mesh
    try:
        mesh = trimesh.load(target_mesh_path, force='mesh')
        # Handle Scene objects by dumping to a single mesh
        if isinstance(mesh, trimesh.Scene):
            if len(mesh.geometry) == 0:
                 print("No geometry in scene")
                 return None
            mesh = trimesh.util.concatenate(
                tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces) 
                      for g in mesh.geometry.values())
            )
            
        # NORMALIZE MESH (Critical Step)
        # Match the preprocessing logic: center at 0, scale to fit unit sphere
        mesh.vertices -= mesh.center_mass
        max_scale = np.max(np.linalg.norm(mesh.vertices, axis=1))
        mesh.vertices /= max_scale
        
    except Exception as e:
        print(f"Error loading mesh: {e}")
        return None

    # Sample points from surface
    # We want points ON the surface, where SDF = 0
    points, _ = trimesh.sample.sample_surface(mesh, 2048)
    points = torch.from_numpy(points).float().to(device) # (N, 3)
    
    # Initialize latent (start from mean)
    latent = torch.zeros(1, model.latent_dim, requires_grad=True, device=device)
    
    optimizer = torch.optim.Adam([latent], lr=lr)
    
    model.eval()
    
    # Optimization loop
    for i in range(num_iterations):
        optimizer.zero_grad()
        
        # Model forward
        # DeepSDF forward expects (1, N, 3) and (1, latent_dim)
        pred = model(points.unsqueeze(0), latent) # (1, N, 1)
        
        # Loss: L1 distance to 0 (surface) + Regularization
        # Reg ensures z stays close to the prior mean (0), preventing "exploding" shapes
        dist_loss = torch.mean(torch.abs(pred))
        reg_loss = 1e-4 * torch.mean(latent.pow(2))
        loss = dist_loss + reg_loss
        
        loss.backward()
        optimizer.step()
        
        if i % 20 == 0:
            print(f"Inversion Iter {i}: Loss {loss.item():.4f}")
            
    return latent.detach()
