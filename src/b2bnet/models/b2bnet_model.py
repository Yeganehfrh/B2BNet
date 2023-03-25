import pytorch_lightning as pl
import torch
from torch import nn
# from torchmetrics import Accuracy


class B2BNetModel(pl.LightningModule):
    """B2BNet model.
    """

    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(input_size=58, hidden_size=2, num_layers=1, batch_first=True)
        self.fc = nn.Linear(2, 3)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        h, y = self.rnn(x)
        y = self.fc(y)
        y = y.permute(1, 0, 2).squeeze(1)
        return y

    def training_step(self, batch, batch_idx):
        X_input, X_output, y_class, y_text = batch
        y_hat = self(X_input)
        loss = self.loss(y_hat, y_class)
        accuracy = (y_hat.argmax(dim=1) == y_class).float().mean()
        self.log('train/loss', loss)
        self.log('train/accuracy', accuracy)
        return loss

    def validation_step(self, batch, batch_idx):
        X_input, X_output, y_class, y_text = batch
        y_hat = self(X_input)
        loss = self.loss(y_hat, y_class)
        accuracy = (y_hat.argmax(dim=1) == y_class).float().mean()
        self.log('val/loss', loss)
        self.log('val/accuracy', accuracy)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
