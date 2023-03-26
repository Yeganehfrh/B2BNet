import pytorch_lightning as pl
import torch
from torch import nn
# from torchmetrics import Accuracy


class B2BNetModel(pl.LightningModule):
    """B2BNet model.
    """

    def __init__(self, input_size=58, n_timesteps=128, hidden_size=2, n_cls_labels=3):
        super().__init__()
        self.encoder = nn.RNN(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=n_timesteps, batch_first=True)

        self.fc = nn.Linear(hidden_size, n_cls_labels)  # classifier

        self.decoder = nn.RNN(
            input_size=hidden_size, hidden_size=input_size,
            num_layers=n_timesteps, batch_first=True)

        self.b2b = nn.RNN(
            input_size=hidden_size, hidden_size=input_size,
            num_layers=n_timesteps, batch_first=True)
        
        self.fc_text = nn.Linear(hidden_size, 1024)

        self.loss_cls = nn.CrossEntropyLoss()
        self.loss_reconn = nn.MSELoss()
        self.loss_b2b = nn.MSELoss()
        self.loss_text = nn.MSELoss()

    def forward(self, x):
        y_enc, h_enc = self.encoder(x)
        y_cls = self.fc(h_enc[-1, :, :])  # select last state
        x_reconn, h_dec = self.decoder(h_enc.permute(1, 0, 2))  # permute to batch first
        y_b2b, h_b2b = self.b2b(h_enc.permute(1, 0, 2))  # permute to batch first
        y_text = self.fc_text(h_enc[-1, :, :])  # select last state
        return y_cls, x_reconn, y_b2b, y_text

    def training_step(self, batch, batch_idx):
        X_input, y_b2b, y_cls, y_text = batch
        y_cls_hat, X_reconn, y_b2b_hat, y_text_hat = self(X_input)

        loss_cls = self.loss_cls(y_cls_hat, y_cls)
        loss_reconn = self.loss_reconn(X_reconn, X_input)
        loss_b2b = self.loss_reconn(y_b2b_hat, y_b2b)
        loss_text = self.loss_reconn(y_text_hat, y_text)
        loss = loss_cls + loss_reconn + loss_b2b + loss_text

        accuracy = (y_cls_hat.argmax(dim=1) == y_cls).float().mean()

        self.log('train/loss_cls', loss_cls)
        self.log('train/loss_reconn', loss_reconn)
        self.log('train/loss_b2b', loss_b2b)
        self.log('train/loss_text', loss_text)
        self.log('train/loss', loss)
        self.log('train/accuracy', accuracy)

        return loss_cls

    def validation_step(self, batch, batch_idx):
        X_input, y_b2b, y_cls, y_text = batch
        y_cls_hat, X_reconn, y_b2b_hat, y_text_hat = self(X_input)

        loss_cls = self.loss_cls(y_cls_hat, y_cls)
        loss_reconn = self.loss_reconn(X_reconn, X_input)
        loss_b2b = self.loss_reconn(y_b2b_hat, y_b2b)
        loss_text = self.loss_reconn(y_text_hat, y_text)
        loss = loss_cls + loss_reconn + loss_b2b + loss_text

        accuracy = (y_cls_hat.argmax(dim=1) == y_cls).float().mean()

        self.log('val/loss_cls', loss_cls)
        self.log('val/loss_reconn', loss_reconn)
        self.log('val/loss_b2b', loss_b2b)
        self.log('val/loss_text', loss_text)
        self.log('val/loss', loss)
        self.log('val/accuracy', accuracy)

        return loss_cls

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
