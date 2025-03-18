import torch
import torch.nn as nn
import torch.nn.functional as F

class CicadaUNetModel(nn.Module):
    def __init__(self, n_encoders=None, s=None, k=None):
        super(CicadaUNetModel, self).__init__()

        # Setting Hyperparameter Defaults
        n_encoders = 5 if n_encoders is None else n_encoders
        s = ([1, 14, 27, 45, 84, 164] if s is None else s)[:n_encoders + 1]
        k = [7 for i in range(n_encoders)] if k is None else k

        print(f"ENCODERS: {n_encoders}")
        print(f"S and K {s}, {k}")

        assert n_encoders+1 == len(s)
        assert n_encoders == len(k)
        
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        
        for i in range(n_encoders):
            self.encoders.append(nn.Sequential(
                nn.Conv1d(s[i], s[i+1], kernel_size=k[i], stride=2, padding=k[i]//2),
                nn.ReLU()
            ))
        
        self.bottleneck = nn.ConvTranspose1d(s[n_encoders], s[n_encoders], kernel_size=k[n_encoders-1], stride=1, padding=((k[n_encoders-1] - 1) // 2))
        
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