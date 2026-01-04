import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import argparse
from tqdm import tqdm
from models.deep_sdf import DeepSDF
import glob

class SDFDataset(Dataset):
    def __init__(self, data_dir, num_samples_per_item=10000):
        self.files = glob.glob(os.path.join(data_dir, "*.npz"))
        print(f"Loading {len(self.files)} shapes from {data_dir}...")
        
        self.data = []
        for f in tqdm(self.files):
            try:
                npz = np.load(f)
                points = npz['points'] # (N, 3)
                sdf = npz['sdf']       # (N,)
                self.data.append({'points': points, 'sdf': sdf})
            except Exception as e:
                print(f"Error loading {f}: {e}")
                
        self.num_samples_per_item = num_samples_per_item
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Randomly sample points from the shape data
        item = self.data[idx]
        points = item['points']
        sdf = item['sdf']
        
        # Ensure we have enough points
        indices = np.random.choice(len(points), self.num_samples_per_item, replace=True)
        
        return {
            'xyz': torch.from_numpy(points[indices]).float(),
            'sdf': torch.from_numpy(sdf[indices]).float(),
            'idx': idx
        }

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Dataset
    dataset = SDFDataset(args.data_dir, num_samples_per_item=args.num_samples)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0) # workers=0 for simplicity
    
    num_shapes = len(dataset)
    
    # Model
    model = DeepSDF(args.latent_dim, hidden_dim=args.hidden_dim, num_layers=args.num_layers, dropout_prob=args.dropout).to(device)
    
    # Latent Codes
    # We optimize latent codes ALONG WITH the network
    # Initialize with normal distribution
    latents = torch.nn.Embedding(num_shapes, args.latent_dim).to(device)
    torch.nn.init.normal_(latents.weight.data, 0.0, 1.0 / np.sqrt(args.latent_dim))
    
    # Optimizer
    optimizer_net = optim.Adam(model.parameters(), lr=args.lr)
    optimizer_latent = optim.Adam(latents.parameters(), lr=args.lr)
    
    # Loss
    l1_loss = nn.L1Loss()
    
    print("Starting training...")
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        total_recon = 0
        total_reg = 0
        
        for batch in tqdm(dataloader, leave=False):
            # Inputs
            xyz = batch['xyz'].to(device) # (B, N, 3)
            sdf_gt = batch['sdf'].to(device).unsqueeze(-1) # (B, N, 1)
            indices = batch['idx'].to(device) # (B,)
            
            # Latent vectors for this batch
            batch_latents = latents(indices) # (B, latent_dim)
            
            # Forward
            # deep_sdf forward handles broadcasting
            sdf_pred = model(xyz, batch_latents) # (B, N, 1) or (B, N)
            
            # Clamp SDF for stability (DeepSDF typically uses clamp distance 0.1)
            # We clamp both prediction and GT for loss calculation if needed, 
            # but usually just clamping the loss is enough or clamping GT.
            # DeepSDF paper: clamp(sdf, -delta, delta)
            clamp_dist = args.clamp_dist
            
            sdf_pred_clamped = torch.clamp(sdf_pred, -clamp_dist, clamp_dist)
            sdf_gt_clamped = torch.clamp(sdf_gt, -clamp_dist, clamp_dist)
            
            recon_loss = l1_loss(sdf_pred_clamped, sdf_gt_clamped)
            
            # Latent Regularization
            reg_loss = torch.mean(torch.sum(torch.square(batch_latents), dim=1)) * 1e-4
            
            loss = recon_loss + reg_loss
            
            # Backward
            optimizer_net.zero_grad()
            optimizer_latent.zero_grad()
            loss.backward()
            optimizer_net.step()
            optimizer_latent.step()
            
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_reg += reg_loss.item()
            
        avg_loss = total_loss / len(dataloader)
        avg_recon = total_recon / len(dataloader)
        avg_reg = total_reg / len(dataloader)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{args.epochs} | Loss: {avg_loss:.5f} | Recon: {avg_recon:.5f} | Reg: {avg_reg:.5f}")
            
        if (epoch + 1) % args.save_freq == 0:
            save_path = os.path.join(args.save_dir, f"model_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'latents': latents.weight.data,
                'optimizer_net': optimizer_net.state_dict(),
                'optimizer_latent': optimizer_latent.state_dict(),
            }, save_path)
            print(f"Saved checkpoint to {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/processed/mugs')
    parser.add_argument('--save_dir', type=str, default='models/checkpoints')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4) # standard DeepSDF uses 1e-4
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--hidden_dim', type=int, default=256) # 256 for faster training? paper used 512
    parser.add_argument('--num_layers', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--num_samples', type=int, default=2048) # points per shape per batch
    parser.add_argument('--clamp_dist', type=float, default=0.1)
    parser.add_argument('--save_freq', type=int, default=100)
    
    args = parser.parse_args()
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        
    train(args)

if __name__ == '__main__':
    main()
