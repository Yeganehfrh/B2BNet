import pytorch_lightning as pl
import torch  # noqa
from torch import nn
from .spacetime_autoencoder import SpaceTimeAutoEncoder


class B2BNetSpaceTimeModel(pl.LightningModule):
    def __init__(self,
                 n_channels, n_features, hidden_size,
                 n_subjects, kernel_size=1):
        super().__init__()

        self.n_subjects = n_subjects

        self.autoencoder = SpaceTimeAutoEncoder(n_channels, n_features, hidden_size, kernel_size=kernel_size)

        # classifier head
        self.cls = nn.Linear(hidden_size, 2)

    def forward(self, x):

        # autoencoder
        h, x_reconn = self.autoencoder(x)
        y_cls = self.cls(h)

        return y_cls, x_reconn

    def training_step(self, batch, batch_idx):
        x, subject_ids, y_b2b, y_cls = batch
        y_cls_hat, x_reconn = self(x)

        # loss
        loss_reconn = nn.functional.mse_loss(x_reconn, x)
        loss_cls = nn.functional.cross_entropy(y_cls_hat, y_cls)

        # total loss
        loss = loss_cls + loss_reconn

        # logging
        self.log('train/loss_reconn', loss_reconn)
        self.log('train/loss_cls', loss_cls)
        self.log('train/accuracy', (y_cls_hat.argmax(dim=1) == y_cls).float().mean())
        self.log('train/loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, subject_ids, y_b2b, y_cls = batch
        y_cls_hat, x_reconn = self(x)

        # loss
        loss_reconn = nn.functional.mse_loss(x_reconn, x)
        loss_cls = nn.functional.cross_entropy(y_cls_hat, y_cls)

        # total loss
        loss = loss_cls + loss_reconn

        # logging
        self.log('val/loss_reconn', loss_reconn)
        self.log('val/loss_cls', loss_cls)
        self.log('val/accuracy', (y_cls_hat.argmax(dim=1) == y_cls).float().mean())
        self.log('val/loss', loss)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-2)
