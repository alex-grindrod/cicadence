import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiheadAttBlock(nn.Module):
    def __init__(self, embed_dims, num_heads, hidden_channels=0.5, dropout=0.1):
        super(MultiheadAttBlock, self).__init__()
        org_dims = embed_dims #Used for linear layer resizing
        print(num_heads)
        if org_dims % num_heads != 0:
            embed_dims = ((org_dims + num_heads - 1) // num_heads) * num_heads
        if not hidden_channels or hidden_channels >= embed_dims:
            hidden_channels = int(embed_dims * 0.5)
        else:
            hidden_channels = int(embed_dims * hidden_channels)

        self.linear_input = nn.Linear(org_dims, embed_dims)
        
        self.mha = nn.MultiheadAttention(embed_dims, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dims, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, embed_dims),
            nn.Dropout(dropout),
        )
        
        self.linear_output = nn.Linear(embed_dims, org_dims)
 
    def forward(self, x):
        x = self.linear_input(x)
        
        att_output, _ = self.mha(x, x, x)
        x = self.norm1(x + att_output)

        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)  
        
        return self.linear_output(x)
        
        
class CicadaUNetAttModel(nn.Module):
    def __init__(self, n_encoders=None, s=None, k=None, num_heads=None, hidden_channels=None, d=None, embed_dims=None):
        super(CicadaUNetAttModel, self).__init__()

        #Setting Hyperparameter Defaults
        n_encoders = 6 if n_encoders is None else n_encoders
        k = [7 for i in range(n_encoders)] if k is None else k
        s = ([1, 8, 16, 23, 38, 70, 134] if s is None else s)[:n_encoders + 1]
        num_heads = 4 if num_heads is None else num_heads
        hidden_channels = 0.5281775897621598 if hidden_channels is None else hidden_channels
        d = 0.25625027800774713 if d is None else d
        embed_dims = 262144 if embed_dims is None else embed_dims
        
        print(f"ENCODERS: {n_encoders}")
        print(f"S and K {s}, {k}")
        print(f"Att Heads: {num_heads}")
        print(f"Hidden Channels: {hidden_channels}")
        print(f"DROPOUT: {d}")

        assert n_encoders+1 == len(s)
        assert n_encoders == len(k)
        
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()

        for i in range(n_encoders):
            self.encoders.append(nn.Sequential(
                nn.Conv1d(s[i], s[i+1], kernel_size=k[i], stride=2, padding=k[i]//2),
                nn.ReLU()
            ))
            embed_dims //= 2

        self.bottleneck = MultiheadAttBlock(embed_dims=embed_dims, num_heads=num_heads, hidden_channels=hidden_channels, dropout=d)
        
        for i in range(n_encoders, 0, -1):
            self.decoders.append( nn.Sequential(
                nn.ConvTranspose1d(s[i] + s[i], s[i-1], kernel_size=k[i-1], stride=2, padding=(k[i-1] - 1) // 2, output_padding=1),
                nn.ReLU()
            ))
        
        self.output = nn.ConvTranspose1d(2, 1, kernel_size=k[i-1], stride=1, padding=(k[i-1] - 1) // 2)
    

    def forward(self, x):
        first_skip = x
        skip_connections = [x]
        for layer in self.encoders:
            # print(x.shape)
            x = layer(x)
            skip_connections.append(x)

        # print(f"BOTTLENECK: {x.shape}")
        x = self.bottleneck(x)

        skip_ptr = len(skip_connections) - 1
        for layer in self.decoders:
            skip = skip_connections[skip_ptr]
            # print(f"SKIP: {skip.shape}")
            # print(f"X: {x.shape}")
            combine = torch.cat([skip, x], dim=1)
            # print(f"COMBINE: {combine.shape}")
            x = layer(combine)
            skip_ptr -= 1
            
        # print(f"Pre-end X: {x.shape}")
        combine = torch.cat([first_skip, x], dim=1)
        x = self.output(combine)
        # print(f"ENDING X: {x.shape}")
        return x
        

