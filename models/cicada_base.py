import torch
import torch.nn as nn
# import torch.nn.functional as F

class CicadaBaseAutoencoder(nn.Module):
    def __init__(self):
        super(CicadaBaseAutoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # (1, 257, 291) -> (16, 257, 291)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), # (32, 257, 291)
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # (32, 128, 145)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),  # (16, 257, 291)
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),  # Output (1, 257, 291)
            nn.Sigmoid()  # Normalize output between 0-1
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x