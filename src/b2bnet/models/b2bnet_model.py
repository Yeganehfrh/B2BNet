import pytorch_lightning as pl
import torch  # noqa
from torch import nn
# from src.b2bnet import TCN
# from torchmetrics import Accuracy


class B2BNetModel(pl.LightningModule):
    """B2BNet model.
    """

    def __init__(self, input_size, n_timesteps, n_cls_labels,
                 n_subjects, hidden_size=8, decoder_hidden_size=8,
                 subject_embedding_dim=4,
                 example_input_array=None):
        super().__init__()
        self.example_input_array = example_input_array

        # # feature extractor
        # self.feature_extractor = TCN(n_timesteps=n_timesteps, output_length=output_length,
        #                              n_features=input_size, kernel_size=kernel_size,
        #                              dilation_base=dilation_base)

        # subject embedding
        self.subject_embedding = nn.Embedding(n_subjects, subject_embedding_dim)

        # encoder
        self.encoder_fc = nn.Linear(input_size + subject_embedding_dim, hidden_size)

        self.encoder = nn.RNN(
            input_size=hidden_size, hidden_size=hidden_size,
            num_layers=n_timesteps, batch_first=True)

        # decoder
        self.decoder = nn.RNN(
            input_size=hidden_size, hidden_size=hidden_size,
            num_layers=n_timesteps, batch_first=True)

        self.decoder_fc = nn.Linear(hidden_size, input_size)

        # classification output model
        self.fc = nn.Linear(hidden_size, n_cls_labels)

        # B2B output model
        self.b2b = nn.RNN(
            input_size=hidden_size, hidden_size=hidden_size,
            num_layers=n_timesteps, batch_first=True)

        self.loss_cls = nn.CrossEntropyLoss()
        self.loss_reconn = nn.MSELoss()
        self.loss_b2b = nn.MSELoss()

    def forward(self, x, subject_ids=None):

        # append subject embedding
        if subject_ids is not None:
            subject_features = self.subject_embedding(subject_ids)
            subject_features = subject_features.repeat(1, x.shape[1], 1)
            x = torch.cat([x, subject_features], dim=2)

        # autoencoder
        x_enc = self.encoder_fc(x)
        y_enc, h_enc = self.encoder(x_enc)
        x_reconn, h_dec = self.decoder(h_enc.permute(1, 0, 2))  # permute to batch first
        x_reconn = self.decoder_fc(x_reconn)

        # classification output
        y_cls = self.fc(h_enc[-1, :, :])  # select last state

        # b2b output
        y_b2b, h_b2b = self.b2b(h_enc.permute(1, 0, 2))  # permute to batch first
        y_b2b = self.decoder_fc(y_b2b)

        return y_cls, x_reconn, y_b2b

    def training_step(self, batch, batch_idx):
        X_input, subject_ids, y_b2b, y_cls = batch
        y_cls_hat, X_reconn, y_b2b_hat = self(X_input, subject_ids)

        loss_cls = self.loss_cls(y_cls_hat, y_cls)
        loss_reconn = self.loss_reconn(X_reconn, X_input)
        loss_b2b = self.loss_reconn(y_b2b_hat, y_b2b)
        loss = loss_cls + loss_reconn + loss_b2b

        accuracy = (y_cls_hat.argmax(dim=1) == y_cls).float().mean()

        self.log('train/loss_cls', loss_cls)
        self.log('train/loss_reconn', loss_reconn)
        self.log('train/loss_b2b', loss_b2b)
        self.log('train/loss', loss)
        self.log('train/accuracy', accuracy)

        return loss_cls

    def validation_step(self, batch, batch_idx):
        X_input, subject_ids, y_b2b, y_cls = batch
        y_cls_hat, X_reconn, y_b2b_hat = self(X_input, subject_ids)

        loss_cls = self.loss_cls(y_cls_hat, y_cls)
        loss_reconn = self.loss_reconn(X_reconn, X_input)
        loss_b2b = self.loss_reconn(y_b2b_hat, y_b2b)
        loss = loss_cls + loss_reconn + loss_b2b

        accuracy = (y_cls_hat.argmax(dim=1) == y_cls).float().mean()

        self.log('val/loss_cls', loss_cls)
        self.log('val/loss_reconn', loss_reconn)
        self.log('val/loss_b2b', loss_b2b)
        self.log('val/loss', loss)
        self.log('val/accuracy', accuracy)

        return loss_cls

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
