import numpy as np
import trimesh
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def compute_features(mesh, sample_indices=None):
    """
    Compute features for clustering.
    Features: Position (x,y,z), Normal (nx,ny,nz).
    Optionally others like Curvature (if available).
    """
    if sample_indices is None:
        vertices = mesh.vertices
        normals = mesh.vertex_normals
    else:
        vertices = mesh.vertices[sample_indices]
        normals = mesh.vertex_normals[sample_indices]
        
    # Weight position vs normal?
    # Mugs: Handle is spatially distinct (x,z) but also normal orientation is complex.
    # Simple concatenation often works for rigid parts.
    
    features = np.concatenate([vertices, normals], axis=1)
    return features

def segment_mesh(mesh, n_clusters=2):
    """
    Segment mesh into n_clusters parts using K-Means.
    Returns: labels (N_vertices,)
    """
    features = compute_features(mesh)
    
    # Normalize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features_scaled)
    
    return labels

def visualize_segmentation(mesh, labels):
    """
    Colorize mesh based on labels.
    """
    # Create a copy
    mesh_vis = mesh.copy()
    
    # Generate colors
    unique_labels = np.unique(labels)
    colors = np.random.randint(0, 255, size=(len(unique_labels), 4))
    colors[:, 3] = 255 # Alpha
    
    vertex_colors = np.zeros((len(mesh.vertices), 4), dtype=np.uint8)
    for i, label in enumerate(unique_labels):
        vertex_colors[labels == label] = colors[i]
        
    mesh_vis.visual.vertex_colors = vertex_colors
    return mesh_vis
