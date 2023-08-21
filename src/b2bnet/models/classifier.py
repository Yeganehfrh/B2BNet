import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from .b2bnet_model import B2BNetSpaceTimeModel
import torchmetrics.functional as tmf


class Classifier(pl.LightningModule):
    def __init__(self,
                 pretrained_encoder_checkpoint_path,
                 n_labels,
                 with_b2b_head=False):
        super().__init__()

        self.with_b2b_head = with_b2b_head

        self.encoder = B2BNetSpaceTimeModel.load_from_checkpoint(pretrained_encoder_checkpoint_path)
        self.embeddings_dim = self.encoder.baseline.time_embedding_dim
        self.model = nn.Sequential(
            nn.Linear(self.embeddings_dim, self.embeddings_dim // 2),
            nn.ReLU(),
            nn.Linear(self.embeddings_dim // 2, n_labels),
            nn.Sigmoid()
        )

        if self.with_b2b_head:
            self.b2b_head = nn.Sequential(
                nn.Linear(self.embeddings_dim, self.embeddings_dim),
                nn.ReLU(),
                nn.Linear(self.embeddings_dim, self.embeddings_dim),
            )

    def forward(self, x):
        _, h = self.encoder.baseline(x)
        y_cls = self.model(h.squeeze(0))
        if self.with_b2b_head:
            h_b2b = self.b2b_head(h.squeeze(0))
        else:
            h_b2b = None

        return y_cls, h_b2b

    def training_step(self, batch, batch_idx):
        x, y, subject_ids, x_b2b, y_b2b, y_cls = batch
        _, h_b2b = self.encoder.baseline(x_b2b)  # pretrain
        y_cls_hat, h_b2b_hat = self.forward(x)

        loss_cls = F.cross_entropy(y_cls_hat, y_cls)
        accuracy = tmf.accuracy(y_cls_hat, y_cls, task='multiclass', num_classes=2)
        self.log('train/accuracy', accuracy)
        self.log('train/loss_cls', loss_cls)
        loss = loss_cls
        if self.with_b2b_head:
            loss_b2b = F.mse_loss(h_b2b_hat, h_b2b)
            self.log('train/loss_b2b', loss_b2b)
            loss += loss_b2b

        self.log('train/loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, subject_ids, x_b2b, y_b2b, y_cls = batch
        _, h_b2b = self.encoder.baseline(x_b2b)
        y_cls_hat, h_b2b_hat = self.forward(x)
        loss_cls = F.cross_entropy(y_cls_hat, y_cls)
        accuracy = tmf.accuracy(y_cls_hat, y_cls, task='multiclass', num_classes=2)
        self.log('val/accuracy', accuracy)
        self.log('val/loss_cls', loss_cls)
        loss = loss_cls
        if self.with_b2b_head:
            loss_b2b = F.mse_loss(h_b2b_hat, h_b2b)
            self.log('val/loss_b2b', loss_b2b)
            loss += loss_b2b

        self.log('val/loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
