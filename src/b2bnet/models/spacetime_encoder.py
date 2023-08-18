from torch import nn
import pytorch_lightning as pl


class SpaceTimeEncoder(pl.LightningModule):
    """B2B head for the B2BNet model, but returns the embedding to compare with the 
    participants embeddings.
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

    def forward(self, x):
        # spatial encoding
        y_space_enc = self.space_encoder(x.permute(0, 2, 1))

        # temporal encoding
        y_time_enc, (h_enc, c_enc) = self.time_encoder(y_space_enc.permute(0, 2, 1))

        return h_enc
