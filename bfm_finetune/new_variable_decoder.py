import torch.nn as nn


class NewVariableHead(nn.Module):
    def __init__(self, latent_dim, out_channels=10000, upsample_factor=4):
        """
        Projects the latent representation (from the backbone) to the high-dimensional output.

        latent_dim: Number of channels in the latent representation.
        out_channels: Number of output channels (e.g. 10,000 species).
        upsample_factor: Factor to upsample the latent grid to match the output spatial dimensions.
        """
        super(NewVariableHead, self).__init__()
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(
                latent_dim,
                latent_dim,
                kernel_size=upsample_factor,
                stride=upsample_factor,
            ),
            nn.ReLU(),
        )
        self.out_proj = nn.Conv2d(latent_dim, out_channels, kernel_size=1)

    def forward(self, latent):
        # latent: (B, latent_dim, H_lat, W_lat)
        x = self.upsample(latent)  # Upsample to (B, latent_dim, H, W)
        out = self.out_proj(x)  # (B, out_channels, H, W)
        return out
