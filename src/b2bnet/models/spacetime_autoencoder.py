import pytorch_lightning as pl # noqa
from torch import nn
import torch.nn.functional as F  # noqa


class SpaceTimeAutoEncoder(nn.Module):
    def __init__(self,
                 n_channels, n_features, hidden_size, kernel_size=1):
        super().__init__()

        # spatial encoder
        self.space_encoder = nn.Sequential(nn.Conv1d(n_channels, n_features, kernel_size),
                                           nn.ReLU(),
                                        # nn.Conv1d(n_features, 16, kernel_size),  # TODO: change 16 to a variable
                                        # nn.ReLU()
                                        )

        # temporal auto-encoder
        self.time_encoder = nn.LSTM(n_features,
                                    hidden_size,
                                    batch_first=True)
        self.time_decoder = nn.LSTM(hidden_size,
                                    n_features,
                                    batch_first=True)

        # spatial decoder
        self.space_decoder = nn.Sequential(nn.ConvTranspose1d(n_features, n_channels, 1, stride=1),
                                           nn.ReLU(),
                                        #  nn.ConvTranspose1d(n_features, n_channels, 1, stride=1)
                                           )

    def forward(self, x):
        # spatial encoding
        x_space_enc = self.space_encoder(x.permute(0, 2, 1))

        # temporal encoding
        x_time_enc, (h_enc, c_enc) = self.time_encoder(x_space_enc.permute(0, 2, 1))
        x_time_dec, _ = self.time_decoder(x_time_enc, (h_enc, c_enc))

        # spatial decoding
        x_space_dec = self.space_decoder(x_time_dec.permute(0, 2, 1))

        return h_enc[-1, :, :], x_space_dec.permute(0, 2, 1)
