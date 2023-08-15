import pytorch_lightning as pl
import torch  # noqa
from torch import nn
from .spacetime_autoencoder import SpaceTimeAutoEncoder
import torchmetrics.functional as tmf


class B2BNetSpaceTimeModel(pl.LightningModule):
    def __init__(self,
                 n_channels, space_embedding_dim, time_embedding_dim,
                 n_subjects, kernel_size=1):
        super().__init__()

        self.n_subjects = n_subjects

        self.autoencoder = SpaceTimeAutoEncoder(
            n_channels, space_embedding_dim, time_embedding_dim,
            kernel_size=kernel_size)

        # classifier head
        self.cls = nn.Linear(time_embedding_dim, 2)

    def forward(self, x):

        # autoencoder
        h, x_reconn = self.autoencoder(x)
        y_cls = self.cls(h)

        return y_cls, x_reconn

    def training_step(self, batch, batch_idx):
        x, y, subject_ids, x_b2b, y_b2b, y_cls = batch
        y_cls_hat, y_hat = self(x)

        # loss
        loss_reconn = nn.functional.mse_loss(y_hat[:, -1, :], y)
        loss_cls = nn.functional.cross_entropy(y_cls_hat, y_cls)

        # total loss
        loss = loss_cls + loss_reconn
        accuracy = tmf.accuracy(y_cls_hat, y_cls, task='multiclass', num_classes=2)
        # DEBUG: accuracy = (y_cls_hat.argmax(dim=1) == y_cls).float().mean()

        # logging
        self.log('train/loss_reconn', loss_reconn)
        self.log('train/loss_cls', loss_cls)
        self.log('train/accuracy', accuracy)
        self.log('train/loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y, subject_ids, x_b2b, y_b2b, y_cls = batch
        y_cls_hat, y_hat = self(x)

        # loss
        loss_reconn = nn.functional.mse_loss(y_hat[:, -1, :], y)
        loss_cls = nn.functional.cross_entropy(y_cls_hat, y_cls)

        # total loss
        loss = loss_cls + loss_reconn
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
