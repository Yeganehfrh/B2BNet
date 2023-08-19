import pytorch_lightning as pl
import torch  # noqa
from torch import nn
from .spacetime_autoencoder import SpaceTimeAutoEncoder
from .spacetime_encoder import SpaceTimeEncoder
import torchmetrics.functional as tmf
from .b2b_head import B2BHead
from typing import Literal


class B2BNetSpaceTimeModel(pl.LightningModule):
    def __init__(self,
                 n_channels, space_embedding_dim, time_embedding_dim,
                 n_subjects, kernel_size=1,
                 base: Literal['autoencoder', 'encoder'] = 'encoder',
                 b2b: Literal['decoder', 'embedding'] | None = None):
        super().__init__()
        self.b2b = b2b
        self.base = base

        # baseline
        if self.base == 'autoencoder':
            self.n_subjects = n_subjects
            self.b2b = b2b

            self.base = SpaceTimeAutoEncoder(
                n_channels, space_embedding_dim, time_embedding_dim,
                kernel_size=kernel_size)

        if self.base == 'encoder':
            self.base = SpaceTimeEncoder(
                n_channels, space_embedding_dim, time_embedding_dim,
                kernel_size=kernel_size)

        # b2b head
        if self.b2b == 'decoder':
            self.b2b_head = B2BHead(n_channels, space_embedding_dim, time_embedding_dim)

        if self.b2b == 'embedding':
            self.b2b_head = SpaceTimeEncoder(n_channels, space_embedding_dim, time_embedding_dim,
                                             kernel_size=kernel_size)

        # classifier head
        self.cls = nn.Linear(time_embedding_dim, 2)

    def forward(self, x, x_b2b):

        # base
        embedding, y_hat = self.base(x)

        # b2b
        y_b2b_hat = embedding_b2b = None  # default values for b2b head

        if self.b2b == 'decoder':
            y_b2b_hat = self.b2b_head(embedding, n_timesteps=x.shape[1])

        if self.b2b == 'embedding':
            embedding_b2b, _ = self.b2b_head(x_b2b)

        # classifier
        y_cls = self.cls(embedding[-1, :, :])

        return embedding, y_hat, y_b2b_hat, embedding_b2b, y_cls

    def training_step(self, batch, batch_idx):
        x, y, subject_ids, x_b2b, y_b2b, y_cls = batch
        embedding, y_hat, y_b2b_hat, embedding_b2b, y_cls_hat = self(x, x_b2b)

        # loss & logging
        loss_cls = nn.functional.cross_entropy(y_cls_hat, y_cls)
        self.log('train/loss_cls', loss_cls)
        loss = loss_cls

        accuracy = tmf.accuracy(y_cls_hat, y_cls, task='multiclass', num_classes=2)
        self.log('train/accuracy', accuracy)

        if self.base == 'autoencoder':
            loss_reconn = nn.functional.mse_loss(y_hat, y)
            loss += loss_reconn
            self.log('train/loss_reconn', loss_reconn)

        if self.b2b == 'decoder':
            loss_b2b = nn.functional.mse_loss(y_b2b_hat, y_b2b)
            loss += loss_b2b
            self.log('train/loss_b2b', loss_b2b)

        if self.b2b == 'embedding':
            loss_b2b = nn.functional.mse_loss(embedding_b2b, embedding)
            loss += loss_b2b
            self.log('train/loss_b2b', loss_b2b)

        self.log('train/loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, subject_ids, x_b2b, y_b2b, y_cls = batch
        embedding, y_hat, y_b2b_hat, embedding_b2b, y_cls_hat = self(x, x_b2b)

        # losses & logging
        loss_cls = nn.functional.cross_entropy(y_cls_hat, y_cls)
        self.log('val/loss_cls', loss_cls)
        loss = loss_cls
        accuracy = tmf.accuracy(y_cls_hat, y_cls, task='multiclass', num_classes=2)
        self.log('val/accuracy', accuracy)

        if self.base == 'autoencoder':
            loss_reconn = nn.functional.mse_loss(y_hat, y)
            loss += loss_reconn
            self.log('train/loss_reconn', loss_reconn)

        if self.b2b == 'decoder':
            loss_b2b = nn.functional.mse_loss(y_b2b_hat, y_b2b)
            loss += loss_b2b
            self.log('train/loss_b2b', loss_b2b)

        if self.b2b == 'embedding':
            loss_b2b = nn.functional.mse_loss(embedding_b2b, embedding)
            loss += loss_b2b
            self.log('train/loss_b2b', loss_b2b)

        self.log('val/loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-2)
