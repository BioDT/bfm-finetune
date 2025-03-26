from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from datetime import datetime
from aurora.batch import Batch, Metadata

class NewVariableHead(nn.Module):
    def __init__(
        self,
        latent_dim,
        out_channels: int,
        target_size=Tuple[int],
    ):
        """
        Projects the latent representation to the desired output.

        Args:
            latent_dim (int): Number of input channels (should match backbone output channels).
            out_channels (int): Desired number of output channels.
            target_size (tuple): Desired spatial dimensions (H, W) for the final output.
        """
        super(NewVariableHead, self).__init__()
        # 1x1 convolution to project the latent features to out_channels.
        self.out_proj = nn.Conv2d(latent_dim, out_channels, kernel_size=1)
        self.target_size = target_size

    def forward(self, latent):
        """
        Expects latent tensor in either 3D (N, C, L) or 4D (N, C, H, W) format.
        If latent is 3D, we assume it is (N, C, W) and insert a dummy height dimension.
        """
        if latent.dim() == 3:
            # Assume latent shape is (N, C, W). Insert a dummy dimension to get (N, C, 1, W).
            latent = latent.unsqueeze(2)
        elif latent.dim() != 4:
            raise ValueError("Expected latent tensor to be 3D or 4D.")

        x = self.out_proj(latent)  # (N, out_channels, H_lat, W_lat)
        # Interpolate spatially to the target size.
        x = F.interpolate(
            x, size=self.target_size, mode="bilinear", align_corners=False
        )
        return x

class Encoder(nn.Module):
    """
    U-Net-like encoder with convolution blocks.
    Here we adapt the in_channels to 1000 to match T=2 x S=500.
    """
    def __init__(self, in_channels=1000, base_channels=64):
        super().__init__()
        # You can tune base_channels as needed
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels*2, base_channels*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels*2),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # Typical U-Net flow
        enc1 = self.encoder1(x)     # [B, base_channels, H, W]
        out1 = self.pool(enc1)      # [B, base_channels, H/2, W/2]

        enc2 = self.encoder2(out1)  # [B, 2*base_channels, H/2, W/2]
        out2 = self.pool(enc2)      # [B, 2*base_channels, H/4, W/4]

        return enc1, enc2, out2


class Decoder(nn.Module):
    """
    U-Net-like decoder that upsamples and concatenates encoder outputs.
    We'll produce 500 output channels for S=500 species at the next time step.
    """
    def __init__(self, out_channels=500, base_channels=64):
        super().__init__()
        self.upconv2 = nn.ConvTranspose2d(base_channels*2, base_channels*2, 2, stride=2)
        self.decoder2 = nn.Sequential(
            nn.Conv2d(base_channels*4, base_channels*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels*2, base_channels*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels*2),
            nn.ReLU(inplace=True),
        )

        self.upconv1 = nn.ConvTranspose2d(base_channels*2, base_channels, 2, stride=2)
        self.decoder1 = nn.Sequential(
            nn.Conv2d(base_channels*2, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )

        self.final_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, enc1, enc2, bottleneck):
        """
        enc1: [B, base_channels, H, W]
        enc2: [B, base_channels*2, H/2, W/2]
        bottleneck: [B, base_channels*2, H/4, W/4] (transformed by Aurora or a skip)
        """
        # up + concat with enc2
        x = self.upconv2(bottleneck)  # -> [B, base_channels*2, H/2, W/2]
        x = torch.cat([x, enc2], dim=1)  # [B, base_channels*4, H/2, W/2]
        x = self.decoder2(x)            # [B, base_channels*2, H/2, W/2]

        # up + concat with enc1
        x = self.upconv1(x)            # -> [B, base_channels, H, W]
        x = torch.cat([x, enc1], dim=1) # [B, base_channels*2, H, W]
        x = self.decoder1(x)           # [B, base_channels, H, W]

        # final
        out = self.final_conv(x)       # [B, out_channels=500, H, W]
        return out


class NewModalityEncoder(nn.Module):
    def __init__(self, species_channels: int = 500, target_spatial: Tuple[int, int] = (152, 320)):
        """
        Args:
            species_channels: Number of channels in the new modality (S=500).
            target_spatial: Target (H, W) resolution required by the Aurora backbone.
        """
        super().__init__()
        self.target_spatial = target_spatial

        # A two-layer convolutional network to map 500 species channels -> 4 channels.
        # Using kernel size 3 and padding=1 preserves spatial dimensions.
        self.adapter = nn.Sequential(
            nn.Conv2d(species_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 4, kernel_size=3, padding=1)
        )

    def forward(self, x: torch.Tensor) -> Batch:
        """
        Args:
            x: Input tensor with shape [B, T, S, H, W] where
               T=2 timesteps, S=500 species, H=152, W=320.
               
        Returns:
            A Batch object with:
              - surf_vars: keys "2t", "10u", "10v", "msl" each of shape [B, T, H_target, W_target]
              - static_vars: keys "lsm", "z", "slt" each of shape [H_target, W_target] (defaulted to zeros)
              - atmos_vars: keys "z", "u", "v", "t", "q" each of shape [B, T, 4, H_target, W_target] (defaulted to zeros)
              - metadata: Contains spatial coordinates and other info.
        """
        B, T, C, H, W = x.shape  # C=500 #num_species
        # Merge batch and time dims for per-frame processing.
        x_reshaped = x.view(B * T, C, H, W)  # [B*T, 500, 152, 320]
        
        # Apply the adapter network.
        adapted = self.adapter(x_reshaped)  # [B*T, 4, 152, 320]
        
        # OPTIONAL Downsample to the target spatial resolution if necessary.
        if (H, W) != self.target_spatial:
            adapted = F.interpolate(adapted, size=self.target_spatial, mode='bilinear', align_corners=False)
        H_target, W_target = self.target_spatial

        # Reshape back to [B, T, 4, H_target, W_target]
        adapted = adapted.view(B, T, 4, H_target, W_target)

        keys = ("2t", "10u", "10v", "msl")
        surf_vars = {k: adapted[:, :, i, :, :] for i, k in enumerate(keys)}
        # Each surf_vars[k] has shape [B, T, H_target, W_target]

        # As this new modality is the only input, we create default static_vars and atmos_vars.
        # For static_vars, use zeros of shape [H_target, W_target]
        static_vars = {
            "lsm": torch.zeros(H_target, W_target, device=x.device),
            "z": torch.zeros(H_target, W_target, device=x.device),
            "slt": torch.zeros(H_target, W_target, device=x.device)
        }
        # For atmos_vars, use zeros of shape [B, T, 4, H_target, W_target]
        atmos_vars = {}
        for k in ("z", "u", "v", "t", "q"):
            atmos_vars[k] = torch.zeros(B, T, 4, H_target, W_target, device=x.device)

        lat = torch.linspace(90, -90, H_target, device=x.device)
        lon = torch.linspace(0, 360, W_target + 1, device=x.device)[:-1]
        metadata = Metadata(
            lat=lat,
            lon=lon,
            time=(datetime(2020, 6, 1, 12, 0),),  # This can be updated with actual timestamps.
            atmos_levels=(100, 250, 500, 850)
        )

        return Batch(
            surf_vars=surf_vars,
            static_vars=static_vars,
            atmos_vars=atmos_vars,
            metadata=metadata
        )

# TODO adapt to map back from Batch to our modalities shapes
class VectorDecoder(nn.Module):
    def __init__(self, latent_dim: int = 128, out_channels: int = 500, final_resolution: Tuple[int, int] = (152, 320), hidden_dim: int = 256):
        """
        Decodes a latent vector (from the frozen backbone) into a spatial output.
        Args:
            latent_dim: Dimension of the latent vector.
            out_channels: Number of output channels (e.g., 500 species).
            final_resolution: Desired output resolution (H, W) = (152, 320).
            hidden_dim: Dimension of the intermediate feature map.
        """
        super().__init__()
        # Choose an intermediate spatial resolution; here, we pick (19, 40) so that 19*8=152 and 40*8=320.
        inter_res = (19, 40)
        self.fc = nn.Linear(latent_dim, hidden_dim * inter_res[0] * inter_res[1])
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, kernel_size=4, stride=2, padding=1),  # (19,40) -> (38,80)
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_dim // 2, hidden_dim // 4, kernel_size=4, stride=2, padding=1),  # (38,80) -> (76,160)
            nn.BatchNorm2d(hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_dim // 4, out_channels, kernel_size=4, stride=2, padding=1)  # (76,160) -> (152,320)
        )
        self.inter_res = inter_res

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Latent vector of shape [B, latent_dim].
        Returns:
            Output tensor of shape [B, out_channels, 152, 320].
        """
        B = x.size(0)
        x = self.fc(x)  # [B, hidden_dim * 19 * 40]
        x = x.view(B, -1, self.inter_res[0], self.inter_res[1])  # [B, hidden_dim, 19, 40]
        x = self.conv(x)  # [B, out_channels, 152, 320]
        return x