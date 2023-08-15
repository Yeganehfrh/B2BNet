import torch
import pytorch_lightning as pl # noqa
from torch import nn
import torch.nn.functional as F  # noqa


class SpaceTimeAutoEncoder(nn.Module):
    """Spatio-temporal auto-encoder.

    """

    def __init__(self,
                 n_channels, space_embedding_dim, time_embedding_dim, kernel_size=1):
        super().__init__()

        self.space_embedding_dim = space_embedding_dim
        self.time_embedding_dim = time_embedding_dim

        # spatial encoder
        self.space_encoder = nn.Sequential(
            nn.Conv1d(n_channels, space_embedding_dim * 2, kernel_size),
            nn.ReLU(),
            nn.Conv1d(space_embedding_dim * 2, space_embedding_dim, kernel_size),
            nn.ReLU())

        # temporal auto-encoder
        self.time_encoder = nn.LSTM(
            self.space_embedding_dim,
            self.time_embedding_dim,
            batch_first=True)

        self.time_decoder = nn.LSTM(
            self.time_embedding_dim,
            self.space_embedding_dim,
            batch_first=True)

        # spatial decoder
        self.space_decoder = nn.Sequential(
            nn.ConvTranspose1d(space_embedding_dim, space_embedding_dim * 2, 1, stride=1),
            nn.ReLU(),
            nn.ConvTranspose1d(space_embedding_dim * 2, n_channels, 1, stride=1),
            nn.ReLU())

    def forward(self, x):

        batch_size = x.shape[0]
        n_timesteps = x.shape[1]

        # spatial encoding
        y_space_enc = self.space_encoder(x.permute(0, 2, 1))

        # temporal encoding
        y_time_enc, (h_enc, c_enc) = self.time_encoder(y_space_enc.permute(0, 2, 1))
        embedding = h_enc[-1, :, :]
        h_enc = h_enc.permute(1, 0, 2).repeat(1, n_timesteps, 1)
        y_time_dec, (_, _) = self.time_decoder(h_enc)

        # spatial decoding
        x_space_dec = self.space_decoder(y_time_dec.permute(0, 2, 1))

        y_dec = x_space_dec.permute(0, 2, 1)

        return embedding, y_dec
