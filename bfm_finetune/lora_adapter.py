import torch
import torch.nn as nn


class LoRAAdapter(nn.Module):
    def __init__(self, in_channels, out_channels, rank=4):
        """
        Maps an input tensor with 'in_channels' to one with 'out_channels'
        using a frozen base mapping plus low-rank updates via LoRA.
        Supports both 4D (B, C, H, W) and 5D (B, T, C, H, W) inputs.
        """
        super(LoRAAdapter, self).__init__()
        # Base mapping: a 1x1 convolution (frozen).
        self.base_mapping = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        for param in self.base_mapping.parameters():
            param.requires_grad = False

        # LoRA parameters: low-rank factors.
        self.lora_A = nn.Conv2d(in_channels, rank, kernel_size=1, bias=False)
        self.lora_B = nn.Conv2d(rank, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        # If x is 5D (B, T, C, H, W), merge batch and time.
        if x.dim() == 5:
            B, T, C, H, W = x.size()
            x_reshaped = x.view(B * T, C, H, W)
            base = self.base_mapping(x_reshaped)
            lora_update = self.lora_B(self.lora_A(x_reshaped))
            out = base + lora_update
            # Reshape back to (B, T, out_channels, H, W)
            return out.view(B, T, -1, H, W)
        else:
            # x is assumed to be 4D: (B, C, H, W)
            base = self.base_mapping(x)
            lora_update = self.lora_B(self.lora_A(x))
            return base + lora_update
