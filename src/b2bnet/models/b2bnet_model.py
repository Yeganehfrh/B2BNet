import pytorch_lightning as pl
import torch
from torch import nn


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
        # y = torch.argmax(y, dim=2).to(torch.float)
        return y

    def training_step(self, batch, batch_idx):
        X_input, X_output, y_class, y_text = batch
        y_hat = self(X_input)
        loss = self.loss(y_hat, y_class)
        self.log('train/loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        X_input, X_output, y_class, y_text = batch
        y_hat = self(X_input)
        loss = self.loss(y_hat, y_class)
        self.log('val/loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
