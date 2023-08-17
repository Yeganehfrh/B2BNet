from torch import nn  # noqa
import pytorch_lightning as pl


class B2BHead(pl.LightningModule):
    """B2B head for the B2BNet model"""

    def __init__(self, time_embedding_dim: int, space_embedding_dim: int, n_channels: int):
        super().__init__()

        # b2b head
        self.b2b_time_decoder = nn.LSTM(time_embedding_dim,
                                        space_embedding_dim,
                                        batch_first=True)
        self.b2b_space_decoder = nn.Sequential(nn.ConvTranspose1d(space_embedding_dim,
                                                                  n_channels,
                                                                  1,
                                                                  stride=1),
                                               nn.ReLU(),
                                               nn.ConvTranspose1d(space_embedding_dim * 2,
                                                                  n_channels,
                                                                  1,
                                                                  stride=1),
                                               nn.ReLU())

    def forward(self, embedding, n_timesteps):
        h_enc = embedding.permute(1, 0, 2).repeat(1, n_timesteps, 1)
        y_b2b_hat, (_, _) = self.b2b_time_decoder(h_enc)
        y_b2b_hat = self.b2b_space_decoder(y_b2b_hat.permute(0, 2, 1))
        return y_b2b_hat.permute(0, 2, 1)
