import gradio as gr
import torch
import numpy as np
import os
from models.deep_sdf import DeepSDF
from utils.edit import compute_semantic_directions, edit_latent, invert_latent
from evaluate import create_mesh
import trimesh

# Global State
MODEL = None
LATENTS = None
DIRECTIONS = None
CURRENT_Z = None # Current base latent (1, 128)
DEVICE = 'cpu'
CURRENT_CATEGORY = "Mug"

CATEGORIES = {
    "Mug": {
        "checkpoints_dir": "models/checkpoints",
        "fallback_checkpoint": "models/checkpoints/model_epoch_200.pth", 
        "latent_dim": 128,
        "initial_msg": "Loaded Mug Model."
    },
    "Chair": {
        "checkpoints_dir": "models/checkpoints/chairs", 
        "fallback_checkpoint": "models/checkpoints/chairs/model_epoch_50.pth",
        "latent_dim": 128,
        "initial_msg": "Loaded Chair Model."
    }
}


def get_mug_checkpoint():
    p = CATEGORIES["Mug"]["fallback_checkpoint"]
    if os.path.exists(p):
        return p
    d = CATEGORIES["Mug"].get("checkpoints_dir")
    if d and os.path.exists(d):
        files = [f for f in os.listdir(d) if f.endswith('.pth') and 'chairs' not in f]
        if files:
            try:
                files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
            except:
                files.sort()
            return os.path.join(d, files[-1])
    return p

def get_chair_checkpoint():
    p = CATEGORIES["Chair"]["fallback_checkpoint"]
    if os.path.exists(p):
        return p
    if d and os.path.exists(d):
        files = [f for f in os.listdir(d) if f.endswith('.pth')]
        if files:
            # Sort by epoch number
            # valid pattern: model_epoch_X.pth
            try:
                files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
            except:
                files.sort() # Fallback
            return os.path.join(d, files[-1])
    return p

def load_model(category):
    global MODEL, LATENTS, DIRECTIONS, DEVICE, CURRENT_CATEGORY, CURRENT_Z
    
    CURRENT_CATEGORY = category
    config = CATEGORIES[category]
    
    if category == "Chair":
        checkpoint_path = get_chair_checkpoint()
    elif category == "Mug":
        checkpoint_path = get_mug_checkpoint()
    else:
        checkpoint_path = config.get("checkpoint", config.get("fallback_checkpoint"))
    
    if not os.path.exists(checkpoint_path):
         return f"Error: Checkpoint not found at {checkpoint_path}", *[gr.update(visible=False) for _ in range(6)]

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading {category} model from {checkpoint_path} on {DEVICE}...")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        latent_dim = checkpoint['latents'].shape[1]
        
        # Infer hidden_dim from checkpoint (supports both 256 and 512)
        # Check first layer bias size
        try:
            hidden_dim = checkpoint['model_state_dict']['layers.0.bias'].shape[0]
        except:
            hidden_dim = 256  # fallback
        
        print(f"Detected hidden_dim={hidden_dim}, latent_dim={latent_dim}")
        
        MODEL = DeepSDF(latent_dim, hidden_dim=hidden_dim, num_layers=8).to(DEVICE)
        MODEL.load_state_dict(checkpoint['model_state_dict'])
        MODEL.eval()
        
        LATENTS = checkpoint['latents'].to(DEVICE)
        CURRENT_Z = LATENTS[0].unsqueeze(0)
        
        # Compute Semantic Directions
        # This now measures attributes and regresses
        DIRECTIONS = compute_semantic_directions(MODEL, LATENTS, device=DEVICE, category=category)
        
        # Sort keys to ensure consistent order
        keys = sorted(list(DIRECTIONS.keys()))
        
        # Update Slider Visibility and Labels
        updates = []
        for i in range(6): 
            if i < len(keys):
                updates.append(gr.Slider(visible=True, value=0.0, label=keys[i]))
            else:
                updates.append(gr.Slider(visible=False, value=0.0))
                
        return f"Loaded {category}. Found controls: {', '.join(keys)}", *updates
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error loading model: {e}", *[gr.update(visible=False) for _ in range(6)]

def update_mesh(s1, s2, s3, s4, s5, s6):
    global MODEL, CURRENT_Z, DIRECTIONS
    
    if MODEL is None or CURRENT_Z is None:
        return None
        
    z = CURRENT_Z.clone() 
    vals = [s1, s2, s3, s4, s5, s6]
    
    if DIRECTIONS:
        keys = sorted(list(DIRECTIONS.keys()))
        for i, val in enumerate(vals):
            if i < len(keys):
                key = keys[i]
                if val != 0:
                     z = edit_latent(z, DIRECTIONS[key], val)
    
    try:
        # Resolution 128: Optimal balance of speed and quality
        mesh = create_mesh(MODEL, z.squeeze(0), resolution=128)
        
        if mesh is None:
            print("Reconstruction returned None.")
            return None
            
        output_path = "temp_mesh.glb"
        mesh.export(output_path)
        return output_path
    except Exception as e:
        print(f"Reconstruction failed: {e}")
        return None

def randomize_base():
    global LATENTS, CURRENT_Z
    if LATENTS is None:
        return "Load model first", None
    
    idx = np.random.randint(0, len(LATENTS))
    CURRENT_Z = LATENTS[idx].unsqueeze(0)
    
    return f"Base Shape Index: {idx}", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 

def invert_and_update(file):
    global MODEL, CURRENT_Z, DEVICE
    if MODEL is None:
        return "Load model first", None
    
    if file is None:
        return "No file uploaded", None
        
    print(f"Inverting {file.name}...")
    try:
        z_opt = invert_latent(MODEL, file.name, num_iterations=100, lr=0.01, device=DEVICE)
        if z_opt is not None:
             CURRENT_Z = z_opt 
             mesh = create_mesh(MODEL, z_opt.squeeze(0), resolution=128)
             if mesh:
                 mesh.export("temp_mesh.glb")
                 return "Inversion Successful", "temp_mesh.glb"
             else:
                 return "Inversion failed to generate surface", None
        else:
            return "Inversion failed", None
    except Exception as e:
        return f"Inversion error: {e}", None


# Layout
with gr.Blocks() as demo:
    gr.Markdown("# Editable 3D Neural Shape Priors (Semantic)")
    
    with gr.Row():
        category_dropdown = gr.Dropdown(choices=["Mug", "Chair"], value="Mug", label="Category")
        load_btn = gr.Button("Load Model")
        status = gr.Textbox(label="Status", value="Not Loaded")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Edit Controls")
            # Create 6 sliders (max)
            sliders = []
            for i in range(6):
                s = gr.Slider(-5.0, 5.0, value=0.0, step=0.1, label=f"Param {i+1}", visible=False)
                sliders.append(s)
            
            gr.Markdown("---")
            rand_btn = gr.Button("Randomize Base Shape")
            base_lbl = gr.Label("Base Shape Index: 0")
            
            gr.Markdown("### Upload Custom Shape")
            upload_btn = gr.File(label="Upload .obj/.glb", file_count="single")
            invert_status = gr.Label(value="")

        with gr.Column(scale=2):
            output_model = gr.Model3D(label="3D Preview", interactive=True)

    # Event Wiring
    load_btn.click(
        fn=load_model, 
        inputs=[category_dropdown], 
        outputs=[status, *sliders]
    ).then(
        fn=update_mesh,
        inputs=sliders,
        outputs=output_model
    )
    
    rand_btn.click(
        fn=randomize_base,
        inputs=[],
        outputs=[base_lbl, *sliders] 
    ).then(
        fn=update_mesh,
        inputs=sliders,
        outputs=output_model
    )
    
    for s in sliders:
        s.change(fn=update_mesh, inputs=sliders, outputs=output_model)
        
    upload_btn.upload(
        fn=invert_and_update,
        inputs=[upload_btn],
        outputs=[invert_status, output_model]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
