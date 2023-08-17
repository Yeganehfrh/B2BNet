import pytorch_lightning as pl
import torch  # noqa
from torch import nn
from .spacetime_autoencoder import SpaceTimeAutoEncoder
import torchmetrics.functional as tmf
from .b2b_head import B2BHead


class B2BNetSpaceTimeModel(pl.LightningModule):
    def __init__(self,
                 n_channels, space_embedding_dim, time_embedding_dim,
                 n_subjects, kernel_size=1, b2b_head=True):
        super().__init__()

        self.n_subjects = n_subjects

        self.autoencoder = SpaceTimeAutoEncoder(
            n_channels, space_embedding_dim, time_embedding_dim,
            kernel_size=kernel_size)

        # classifier head
        self.cls = nn.Linear(time_embedding_dim, 2)

        if b2b_head:
            self.b2b_head = B2BHead(time_embedding_dim, space_embedding_dim, n_channels)

    def forward(self, x):

        # autoencoder
        embedding, y_hat = self.autoencoder(x)
        y_cls = self.cls(embedding[-1, :, :])

        if self.b2b_head:
            y_b2b_hat = self.b2b_head(embedding, n_timesteps=x.shape[1])
            return y_cls, y_hat[:, -1, :], y_b2b_hat[:, -1, :]

        else:
            return y_cls, y_hat[:, -1, :], None

    def training_step(self, batch, batch_idx):
        x, y, subject_ids, y_b2b, y_cls = batch
        y_cls_hat, y_hat, y_b2b_hat = self(x)

        # loss
        loss_reconn = nn.functional.mse_loss(y_hat, y)
        loss_cls = nn.functional.cross_entropy(y_cls_hat, y_cls)
        loss = loss_cls + loss_reconn

        if self.b2b_head:
            loss_b2b = nn.functional.mse_loss(y_b2b_hat, y_b2b)
            loss = loss + loss_b2b
            self.log('train/loss_b2b', loss_b2b)

        accuracy = tmf.accuracy(y_cls_hat, y_cls, task='multiclass', num_classes=2)
        # DEBUG: accuracy = (y_cls_hat.argmax(dim=1) == y_cls).float().mean()

        # logging
        self.log('train/loss_reconn', loss_reconn)
        self.log('train/loss_cls', loss_cls)
        self.log('train/accuracy', accuracy)
        self.log('train/loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y, subject_ids, y_b2b, y_cls = batch
        y_cls_hat, y_hat, y_b2b_hat = self(x)

        # loss
        loss_reconn = nn.functional.mse_loss(y_hat, y)
        loss_cls = nn.functional.cross_entropy(y_cls_hat, y_cls)
        loss = loss_cls + loss_reconn

        if self.b2b_head:
            loss_b2b = nn.functional.mse_loss(y_b2b_hat, y_b2b)
            loss = loss + loss_b2b
            self.log('train/loss_b2b', loss_b2b)

        accuracy = tmf.accuracy(y_cls_hat, y_cls, task='multiclass', num_classes=2)
        # DEBUG: accuracy = (y_cls_hat.argmax(dim=1) == y_cls).float().mean()

        # logging
        self.log('val/loss_reconn', loss_reconn)
        self.log('val/loss_cls', loss_cls)
        self.log('val/accuracy', accuracy)
        self.log('val/loss', loss)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-2)
