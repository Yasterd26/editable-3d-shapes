import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepSDF(nn.Module):
    def __init__(
        self,
        latent_dim,
        hidden_dim=512,
        num_layers=8,
        skip_connection=[4],
        dropout_prob=0.2,
        use_weight_norm=True
    ):
        super(DeepSDF, self).__init__()
        
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.skip_connection = skip_connection
        self.dropout_prob = dropout_prob
        
        # Input dimension: 3 (xyz) + latent_dim
        input_dim = 3 + latent_dim
        
        layers = []
        for i in range(num_layers + 1):
            if i == 0:
                in_dim = input_dim
                out_dim = hidden_dim
            elif i in skip_connection:
                in_dim = hidden_dim + input_dim
                out_dim = hidden_dim
            else:
                in_dim = hidden_dim
                out_dim = hidden_dim
            
            # Last layer outputs 1 value (SDF)
            if i == num_layers:
                out_dim = 1
                
            linear = nn.Linear(in_dim, out_dim)
            
            if use_weight_norm:
                linear = nn.utils.weight_norm(linear)
                
            layers.append(linear)
            
        self.layers = nn.ModuleList(layers)
        
    def forward(self, input_xyz, latent_code):
        """
        input_xyz: (B, N, 3) or (B, 3)
        latent_code: (B, latent_dim)
        """
        # Handle shapes
        # If input is (B, N, 3), we need to replicate latent code
        if input_xyz.dim() == 3:
            B, N, _ = input_xyz.shape
            # latent_code: (B, L) -> (B, N, L)
            latent_expanded = latent_code.unsqueeze(1).expand(-1, N, -1)
            x = torch.cat([input_xyz, latent_expanded], dim=2) # (B, N, 3+L)
        else:
            x = torch.cat([input_xyz, latent_code], dim=1)
            
        input_vec = x.clone() # Keep original input for skip connections
        
        for i, layer in enumerate(self.layers):
            if i in self.skip_connection:
                x = torch.cat([x, input_vec], dim=-1)
                
            x = layer(x)
            
            # Activation and Dropout (except for last layer)
            if i < self.num_layers:
                x = F.relu(x)
                if self.dropout_prob > 0.0:
                    x = F.dropout(x, p=self.dropout_prob, training=self.training)
                    
        return x
