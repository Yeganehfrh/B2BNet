from torch import nn
import math
from .tcn_block import TCNBlock


class TCN(nn.Module):
    def __init__(self, n_timesteps, output_length, n_features, kernel_size, dilation_base, dropout=0.2):
        super().__init__()

        n_layers = math.ceil(
            math.log(
                (n_timesteps - 1) * (dilation_base - 1) / (kernel_size - 1) / 2 + 1,
                dilation_base,
            )
        )

        self.blocks = []
        for i in range(n_layers):
            dilation = dilation_base ** i
            block = TCNBlock(
                n_timesteps,
                n_features,
                kernel_size,
                dilation,
            )
            self.blocks.append(block)
        self.blocks = nn.Sequential(*self.blocks)

    def forward(self, x):
        return self.blocks(x)
