import torch  # noqa 
from torch import nn
import torch.nn.functional as F


class TCNBlock(nn.Module):
    def __init__(self,
                 n_timesteps: int,
                 n_features: int,
                 kernel_size: int,
                 dilation: int,
                 stride: int):
        super().__init__()

        self.n_timesteps = n_timesteps
        self.n_features = n_features
        self.kernel_size = kernel_size
        self.dilation = dilation

        self.conv1 = nn.Sequential(
            nn.Conv1d(n_features, n_features, kernel_size, dilation=self.dilation, stride=stride),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, x):

        # first step
        left_padding = self.dilation * (self.kernel_size - 1)
        x = F.pad(x, (left_padding, 0))
        x = self.conv1(x)
        return x
