from typing import Tuple

import torch.nn as nn
import torch.nn.functional as F


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
