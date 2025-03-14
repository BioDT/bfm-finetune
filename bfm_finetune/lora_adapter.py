import torch.nn as nn

class LoRAAdapter(nn.Module):
    def __init__(self, in_channels, out_channels, rank=4):
        """
        Maps an input tensor with 'in_channels' to one with 'out_channels'
        using a base mapping (frozen) plus low-rank updates via LoRA.
        """
        super(LoRAAdapter, self).__init__()
        self.base_mapping = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        for param in self.base_mapping.parameters():
            param.requires_grad = False
        
        self.lora_A = nn.Conv2d(in_channels, rank, kernel_size=1, bias=False)
        self.lora_B = nn.Conv2d(rank, out_channels, kernel_size=1, bias=False)
        
    def forward(self, x):
        # x: (B, in_channels, H, W)
        base = self.base_mapping(x)
        lora_update = self.lora_B(self.lora_A(x))
        return base + lora_update
