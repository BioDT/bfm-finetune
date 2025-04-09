from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from datetime import datetime
from aurora.batch import Batch, Metadata
from bfm_finetune.utils import get_supersampling_target_lat_lon

class NewVariableHead(nn.Module):
    def __init__(
        self,
        latent_dim,
        out_channels: int,
        target_size: Tuple[int],
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
        # print("new_variable_decoder.__init__: latent_dim", latent_dim)
        # print("new_variable_decoder.__init__: out_channels", out_channels)
        # print("new_variable_decoder.__init__: target_size", target_size)

    def forward(self, latent):
        """
        Expects latent tensor in either 3D (N, C, L) or 4D (N, C, H, W) format.
        If latent is 3D, we assume it is (N, C, W) and insert a dummy height dimension.
        """
        # print("new_variable_decoder.forward: latent.shape (before)", latent.shape)
        if latent.dim() == 3:
            # Assume latent shape is (N, C, W). Insert a dummy dimension to get (N, C, 1, W).
            latent = latent.unsqueeze(2)
        elif latent.dim() != 4:
            raise ValueError("Expected latent tensor to be 3D or 4D.")
        # print("new_variable_decoder.forward: latent.shape (after)", latent.shape)

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
    def __init__(self, species_channels: int = 500, hidden_channels: int = 160, target_spatial: Tuple[int, int] = (152, 320)):
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
            nn.Conv2d(species_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 4, kernel_size=3, padding=1)
        )
        print("Finished Init Encoder")

    def forward(self, batch: torch.Tensor) -> Batch:
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
        # Expect the new modality variable to be stored under "species_distribution" in surf_vars.
        if "species_distribution" not in batch.surf_vars:
            raise ValueError("Input Batch must contain 'species_distribution' in surf_vars.")
        
        x = batch.surf_vars["species_distribution"]  # Expected shape: [B, T, 500, H, W]
        B, T, C, H, W = x.shape  # Here, C should equal species_channels (500).
        

        # B, T, C, H, W = x.shape  # C=500 #num_species
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
        # TODO REMOVE and use the Batche's values
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

class VectorDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int = 128,
        out_channels: int = 500,
        hidden_dim: int = 256,
        final_resolution: Tuple[int, int] = (152, 320)
    ):
        """
        Decodes a Batch structure (with surf_vars and atmos_vars at a low resolution)
        into a compact latent representation and then upsamples it to produce an output tensor of shape [B, out_channels, 152, 320].
        
        The input Batch is expected to have:
          - surf_vars: keys "2t", "10u", "10v", "msl" with shape [B, 4, 17, 32]
          - atmos_vars: keys "z", "u", "v", "t", "q" with shape [B, 5, 4, 17, 32]
        
        This decoder performs the following steps:
          1. Extracts and flattens the surf_vars into a vector of size 4*17*32 = 2176.
          2. Extracts and flattens the atmos_vars into a vector of size 5*4*17*32 = 10880.
          3. Projects each to a latent vector of size latent_dim.
          4. Fuses the two latent vectors (by concatenation followed by a linear layer).
          5. Maps the fused latent representation to an intermediate feature map.
          6. Upsamples via transposed convolutions to reach final resolution (152, 320).
        """
        super().__init__()
        ### V1
        # Constants: low resolution expected in the Batch.
        # Expected low resolution from the Batch.
        low_res = (17, 32)
        # low_res = (152, 320)
        # Compute the number of input features for surface and atmospheric variables.
        self.surf_in_dim = 4 * low_res[0] * low_res[1]         # 4*17*32 = 2176
        self.atmos_in_dim = 5 * 4 * low_res[0] * low_res[1]      # 5*4*17*32 = 10880
        
        ### V2
        # Use the full resolution available.
        # H, W = final_resolution  # (152, 320)
        # self.surf_in_dim = 4 * H * W         # 4 * 152 * 320 = 194560
        # self.atmos_in_dim = 5 * 4 * H * W      # 5 * 4 * 152 * 320 = 972800

        # Linear layers to project flattened features to latent_dim.
        self.surf_linear = nn.Linear(self.surf_in_dim, latent_dim)
        self.atmos_linear = nn.Linear(self.atmos_in_dim, latent_dim)
        # Fuse the two latent representations (concatenation -> 2*latent_dim) and map to latent_dim.
        self.fusion = nn.Linear(2 * latent_dim, latent_dim)
        
        # Define an intermediate spatial resolution for upsampling.
        # We choose (19, 40) so that after three upsampling steps we reach (152, 320) since 19*8=152 and 40*8=320.
        self.inter_res = (19, 40)
        # Map fused latent to a hidden feature map.
        self.fc = nn.Linear(latent_dim, hidden_dim * self.inter_res[0] * self.inter_res[1])
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, kernel_size=4, stride=2, padding=1),  # (19,40) -> (38,80)
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_dim // 2, hidden_dim // 4, kernel_size=4, stride=2, padding=1),  # (38,80) -> (76,160)
            nn.BatchNorm2d(hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_dim // 4, out_channels, kernel_size=4, stride=2, padding=1)  # (76,160) -> (152,320)
        )
        self.final_resolution = final_resolution
        self.out_channels = out_channels
        print("Finsihed init Decoder")

    def forward(self, batch: Batch) -> torch.Tensor:
        """
        Args:
            batch: A Batch structure whose surf_vars and atmos_vars are at low resolution.
                   Expected:
                     - Each surf_vars[key] has shape [B, 1, 17, 32].
                     - Each atmos_vars[key] has shape [B, 5, 4, 17, 32].
        Returns:
            A tensor of shape [B, out_channels, 152, 320] representing the final decoded output.
        """
        B = next(iter(batch.surf_vars.values())).size(0)
        # Process surface variables.
        surf_features = []
        surf_keys = ("2t", "10u", "10v", "msl")
        for key in surf_keys:
            # Extract the first (and only) timestep: shape [B, 17, 32]
            feat = batch.surf_vars[key][:, 0, ...]
            print("Surface shape", feat.shape)
            # print("Surf keys dim", batch.surf_vars[key].shape)
            # Add a channel dimension if necessary (should be [B, 1, 17, 32]).
            if feat.dim() == 3:
                feat = feat.unsqueeze(1)
            # Downsample to expected resolution using adaptive average pooling.
            # feat_ds = F.adaptive_avg_pool2d(feat, self.expected_low_res)  # [B, 1, 17, 32]
            surf_features.append(feat)
        # Concatenate along channel dimension → [B, 4, 17, 32].
        surf_concat = torch.cat(surf_features, dim=1)
        # Flatten: [B, 4*17*32] = [B, 2176].
        surf_flat = surf_concat.view(B, -1)
        # Project to latent.
        # print("Surf_flat dim", surf_flat.shape)
        latent_surf = self.surf_linear(surf_flat)  # [B, latent_dim]
        
        # Process atmospheric variables.
        atmos_features = []
        atmos_keys = ("z", "u", "v", "t", "q")
        for key in atmos_keys:
            # Each atmos_vars[key] is expected to have shape [B, 1, 4, 17, 32].
            feat = batch.atmos_vars[key][:, 0, ...]  # now shape: [B, 4, 17, 32]
            # Downsample spatially to expected resolution.
            # feat_ds = F.adaptive_avg_pool2d(feat, self.expected_low_res)  # [B, 4, 17, 32]
            atmos_features.append(feat)
        # Concatenate along channel dimension → [B, 5*4, 17, 32] = [B, 20, 17, 32].
        atmos_concat = torch.cat(atmos_features, dim=1)
        # Flatten: [B, 20*17*32] = [B, 10880].
        atmos_flat = atmos_concat.view(B, -1)
        latent_atmos = self.atmos_linear(atmos_flat)  # [B, latent_dim]
        fused_latent = self.fusion(torch.cat([latent_surf, latent_atmos], dim=1))  # [B, latent_dim]
        
        # Map fused latent to intermediate feature map.
        x_fc = self.fc(fused_latent)  # [B, hidden_dim * 19 * 40]
        x_fc = x_fc.view(B, -1, self.inter_res[0], self.inter_res[1])  # [B, hidden_dim, 19, 40]
        
        # Upsample to final resolution.
        x_out = self.conv(x_fc)  # [B, out_channels, 152, 320]
        return x_out
    


class VectorDecoderSimple(nn.Module):
    def __init__(
        self,
        latent_dim: int = 128,
        out_channels: int = 1000,
        hidden_dim: int = 256,  # not used in this simplified version
        final_resolution: Tuple[int, int] = (152, 320)
    ):
        """
        Simplified decoder that takes a Batch (with surf_vars and atmos_vars at full resolution)
        and produces an output tensor of shape [B, out_channels, 152, 320].
        
        Steps:
          1. Flatten the surface variables (4 keys) into a vector of size 4*152*320.
          2. Flatten the atmospheric variables (5 keys, each with 4 channels) into a vector of size 5*4*152*320.
          3. Project each to a latent vector of size latent_dim.
          4. Fuse the two latent vectors (concatenation followed by a linear layer) into one latent vector.
          5. Decode the fused latent via a fully connected layer to output the final tensor.
        """
        super().__init__()
        H, W = final_resolution  # (152, 320)
        self.surf_in_dim = 4 * H * W         # 4 * 152 * 320 = 194560
        # For atmospheric variables (5 keys, each with 4 channels):
        self.atmos_in_dim = 5 * 4 * H * W      # 5 * 4 * 152 * 320 = 972800

        self.surf_linear = nn.Linear(self.surf_in_dim, latent_dim)
        self.atmos_linear = nn.Linear(self.atmos_in_dim, latent_dim)
        # Fusion of the two latent vectors:
        self.fusion = nn.Linear(2 * latent_dim, latent_dim)
        
        # Decoder: project fused latent vector to final output.
        # We simply use a linear layer and then reshape.
        self.fc_out = nn.Linear(latent_dim, out_channels * H * W)
        self.final_resolution = final_resolution
        self.out_channels = out_channels

    def forward(self, batch: Batch) -> torch.Tensor:
        """
        Args:
            batch: A Batch structure with:
                - surf_vars: each key ("2t", "10u", "10v", "msl") of shape [B, 1, 152, 320]
                - atmos_vars: each key ("z", "u", "v", "t", "q") of shape [B, 1, 4, 152, 320]
        Returns:
            A tensor of shape [B, out_channels, 152, 320].
        """
        B = next(iter(batch.surf_vars.values())).size(0)
        H, W = self.final_resolution
        
        surf_keys = ("2t", "10u", "10v", "msl")
        surf_feats = []
        for key in surf_keys:
            # Extract the single timestep: shape [B, 152, 320]
            feat = batch.surf_vars[key][:, 0, ...]
            # Add a channel dimension if needed: shape [B, 1, 152, 320]
            if feat.dim() == 3:
                feat = feat.unsqueeze(1)
            surf_feats.append(feat)
        # Concatenate along channel dimension: [B, 4, 152, 320]
        surf_concat = torch.cat(surf_feats, dim=1)
        # Flatten: [B, 4*152*320] = [B, 194560]
        surf_flat = surf_concat.view(B, -1)
        latent_surf = self.surf_linear(surf_flat)  # [B, latent_dim]
        
        atmos_keys = ("z", "u", "v", "t", "q")
        atmos_feats = []
        for key in atmos_keys:
            # Each atmos_vars[key] has shape [B, 1, 4, 152, 320]. Take the first timestep:
            feat = batch.atmos_vars[key][:, 0, ...]  # [B, 4, 152, 320]
            atmos_feats.append(feat)
        # Concatenate along channel dimension: [B, 5*4, 152, 320] = [B, 20, 152, 320]
        atmos_concat = torch.cat(atmos_feats, dim=1)
        # Flatten: [B, 20*152*320] = [B, 972800]
        atmos_flat = atmos_concat.view(B, -1)
        latent_atmos = self.atmos_linear(atmos_flat)  # [B, latent_dim]
        
        # Fuse the two latent representations.
        fused_latent = self.fusion(torch.cat([latent_surf, latent_atmos], dim=1))  # [B, latent_dim]
        
        fc_out = self.fc_out(fused_latent)  # [B, out_channels * 152 * 320]
        x_out = fc_out.view(B, self.out_channels, H, W)
        return x_out
    
class Conv3dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm3d(out_channels)
    
    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class Upsampler(nn.Module):
    """
    Upsamples a feature map from (H, W) = (152,320) to (721,1440)
    using only learned transposed convolution layers with output_size control.
    Input is expected with shape [B, C, T, 152,320].
    """
    def __init__(self, in_channels, out_H: int, out_W: int):
        super(Upsampler, self).__init__()
        ## V1
        self.up1 = nn.ConvTranspose3d(in_channels, in_channels, kernel_size=(1,2,2), 
                                      stride=(1,2,2))
        self.up2 = nn.ConvTranspose3d(in_channels, in_channels, kernel_size=(1,2,2), 
                                      stride=(1,2,2))
        self.conv_adjust = nn.Conv3d(in_channels, in_channels, kernel_size=(1,3,3), padding=(0,1,1))
        ## V2
        # self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.out_H = out_H
        self.out_W = out_W

    
    def forward(self, x):
        ## V1
        # x: [B, C, T, H, W] with H,W ~ (152,320)
        x = self.up1(x)  # now approx (152*2, 320*2) = (304,640)
        x = self.up2(x)  # now approx (304*2,640*2) = (608,1280)
        x = self.conv_adjust(x)  # features refined at (608,1280)
        x = F.interpolate(x, size=(x.shape[2], self.out_H, self.out_W), mode='trilinear', align_corners=False)

        ## V2
        # x = F.interpolate(x, size=(x.shape[2], 721, 1440), mode='trilinear', align_corners=False)
        # # Refine features with a learned convolution:
        # x = self.conv(x)
        return x


class Downsampler(nn.Module):
    """
    Downsamples a feature map from (H, W) = (721,1440) to (152,320)
    using only learned convolution layers with output_size control.
    Input is expected with shape [B, C, T, 721,1440].
    """
    def __init__(self, in_channels, out_H: int, out_W: int):
        super(Downsampler, self).__init__()
        ## V1
        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=(1,2,2), stride=(1,2,2))
        self.conv2 = nn.Conv3d(in_channels, in_channels, kernel_size=(1,2,2), stride=(1,2,2))
        self.conv_adjust = nn.Conv3d(in_channels, in_channels, kernel_size=(1,3,3), padding=(0,1,1))
        ## V2
        # self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=(1,5,5), padding=1)
        self.out_H = out_H
        self.out_W = out_W
    
    
    def forward(self, x):
        ## V1
        # x: [B, in_channels, T, 721,1440]
        x = self.conv1(x)  # approx (721/2,1440/2) = (360,720)
        x = self.conv2(x)  # approx (360/2,720/2) = (180,360)
        x = self.conv_adjust(x)  # refine features at (180,360)
        # Final adjustment: interpolate to exactly (152,320)
        x = F.interpolate(x, size=(x.shape[2], self.out_H, self.out_W), mode='trilinear', align_corners=False)
        ## V2
        # x = self.conv(x) 
        # x = F.interpolate(x, size=(x.shape[2], 152, 320), mode='trilinear', align_corners=False)

        return x

class InputMapper(nn.Module):
    def __init__(self, in_channels=500, timesteps=2, base_channels=64, atmos_levels=(100, 250, 500, 850), upsampling=None):
        """
        Args:
            in_channels (int): Number of input channels.
            timesteps (int): Number of time steps (e.g. 2).
            base_channels (int): Base number of channels for the network.
            geo_size (tuple): Tuple with (H, W) for geographic dimensions.
            num_atmos_levels (int): Number of levels for each atmospheric variable.
        """
        super(InputMapper, self).__init__()
        self.num_atmos_levels = len(atmos_levels)
        self.atmos_levels = atmos_levels
        self.upsampling = upsampling
        self.supersampling_target_lat_lon = get_supersampling_target_lat_lon(supersampling_config=self.upsampling)
        out_channels = 7 + 5 * self.num_atmos_levels
        self.init_conv = nn.Conv3d(in_channels, base_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.encoder_conv = nn.Conv3d(base_channels, base_channels * 2, kernel_size=(1, 3, 3),
                                      stride=(1, 2, 2), padding=(0, 1, 1))
        self.decoder_conv = nn.ConvTranspose3d(base_channels * 2, base_channels,
                                               kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.final_conv = nn.Conv3d(base_channels, out_channels,
                                    kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.relu = nn.ReLU(inplace=True)
        if self.supersampling_target_lat_lon:
            self.out_H = self.supersampling_target_lat_lon[0].shape[0]
            self.out_W = self.supersampling_target_lat_lon[1].shape[0]
            self.upsample_net = Upsampler(out_channels, out_H=self.out_H, out_W=self.out_W)

    def forward(self, batch):
        """
        Args:
            batch: An object with batch.surf_vars["species_distribution"] of shape [B, T, C_in, H, W].
        Returns:
            A Batch structure with surf_vars, static_vars, atmos_vars, and metadata.
        """
        if self.supersampling_target_lat_lon:
            # it means we are upscaling to this
            aurora_lat = torch.Tensor(self.supersampling_target_lat_lon[0])
            aurora_lon = torch.Tensor(self.supersampling_target_lat_lon[1])
        else:
            # otherwise take the original
            aurora_lat = batch.metadata.lat
            aurora_lon = batch.metadata.lon
        x = batch.surf_vars["species_distribution"]
        B, T, C_in, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4)
        x = self.relu(self.init_conv(x))          # [B, base_channels, T, H, W]
        x_enc = self.relu(self.encoder_conv(x))     # [B, base_channels*2, T, H/2, W/2]
        x_dec = self.relu(self.decoder_conv(x_enc)) # [B, base_channels, T, H, W]
        # Optional skip connection.
        x_dec = x_dec + x
        x_out = self.final_conv(x_dec)              # [B, 7+5*num_atmos_levels, T, H, W]
        if self.supersampling_target_lat_lon:
            x_out = self.upsample_net(x_out)
        # Surf_vars: first 4 channels → [B, 4, T, H, W] then permute to [B, T, 4, H, W]
        surf = x_out[:, :4, :, :, :].permute(0, 2, 1, 3, 4)
        keys_surf = ("2t", "10u", "10v", "msl")
        surf_vars = {k: surf[:, :, i, :, :] for i, k in enumerate(keys_surf)}

        # Static_vars: next 3 channels → [B, 3, T, H, W]
        static = x_out[:, 4:7, :, :, :]
        # For static, take the first batch element and first time step → [3, H, W]
        static = static[0, :, 0, :, :]
        keys_static = ("lsm", "slt", "z")
        static_vars = {k: static[i, :, :] for i, k in enumerate(keys_static)}

        # Atmos_vars: remaining channels → [B, 5*num_atmos_levels, T, H, W]
        atmos = x_out[:, 7:, :, :, :]
        # Reshape to [B, 5, num_atmos_levels, T, H, W] then permute to [B, T, 5, num_atmos_levels, H, W]
        atmos = atmos.view(B, 5, self.num_atmos_levels, T, aurora_lat.shape[0], aurora_lon.shape[0]).permute(0, 3, 1, 2, 4, 5)
        keys_atmos = ("t", "u", "v", "q", "z")
        atmos_vars = {k: atmos[:, :, i, :, :, :] for i, k in enumerate(keys_atmos)}

        # time_stamp = batch.metadata.time[0]
        # print("Input mappers batch timestamp :", time_stamp)
        metadata = Metadata(
            lat=aurora_lat,
            lon=aurora_lon,
            time=(datetime(2020, 6, 1, 12, 0),), # batch.metadata.time, # TODO
            atmos_levels=self.atmos_levels
        )
        return Batch(surf_vars=surf_vars, static_vars=static_vars, atmos_vars=atmos_vars, metadata=metadata)


class OutputMapper(nn.Module):
    def __init__(self, out_channels=1000, atmos_levels=(100, 250, 500, 850), lat_lon=Tuple[np.ndarray, np.ndarray], downsampling = None):
        """
        Args:
            num_atmos_levels (Tuple): The levels for each atmospheric variable.
            out_channels (int): Number of output channels.
        """
        super(OutputMapper, self).__init__()
        in_channels = 7 + 5 * len(atmos_levels)
        self.downsampling = downsampling
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.conv2 = nn.Conv3d(64, 128, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.conv3 = nn.Conv3d(128, out_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.downsample_net = Downsampler(out_channels, out_H=lat_lon[0].shape[0], out_W=lat_lon[1].shape[0])
    
    def forward(self, batch: Batch) -> torch.Tensor:
        """
        Reconstructs the prediction back to a tensor of shape [B, T, C_out, H, W] where T=1.
        """
        b, t, H, W = next(iter(batch.surf_vars.values())).shape
        
        surf_list = [v.unsqueeze(2) for v in batch.surf_vars.values()]
        surf = torch.cat(surf_list, dim=2)  # [B, T, 4, H, W]
        
        static_list = [v.unsqueeze(0).unsqueeze(0).expand(b, t, 1, H, W) for v in batch.static_vars.values()]
        static = torch.cat(static_list, dim=2)  # [B, T, 3, H, W]
        
        atmos_list = [v for v in batch.atmos_vars.values()]
        atmos = torch.cat(atmos_list, dim=2)  # [B, T, 5*num_atmos_levels, H, W]
        
        # Concatenate all along channel dimension: [B, T, 7 + 5*num_atmos_levels, H, W]
        x = torch.cat([surf, static, atmos], dim=2)
        # Rearrange to [B, channels, T, H, W] for convolution.
        x = x.permute(0, 2, 1, 3, 4)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        if self.downsampling:
            x = self.downsample_net(x)
        # Rearrange back to [B, T, out_channels, H, W]
        x = x.permute(0, 2, 1, 3, 4)
        return x