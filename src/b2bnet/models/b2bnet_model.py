import pytorch_lightning as pl
import torch  # noqa
from torch import nn
import torch.nn.functional as F
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

        self.encoder = nn.LSTM(
            input_size=input_size + subject_embedding_dim, hidden_size=hidden_size,
            num_layers=n_timesteps, batch_first=True)

        # decoder
        self.decoder = nn.LSTM(
            input_size=hidden_size, hidden_size=input_size + subject_embedding_dim,
            num_layers=n_timesteps, batch_first=True)

        # classification output model
        self.fc = nn.Linear(hidden_size, n_cls_labels)

        # B2B output model
        self.b2b = nn.RNN(
            input_size=hidden_size, hidden_size=input_size,
            num_layers=n_timesteps, batch_first=True)

        self.loss_cls = nn.CrossEntropyLoss()
        self.loss_reconn = nn.MSELoss()
        self.loss_b2b = nn.MSELoss()

    def forward(self, x, subject_ids=None):

        # append subject embedding
        if subject_ids is not None:
            subject_features = self.subject_embedding(subject_ids)
            subject_features = subject_features.repeat(1, x.shape[1], 1)
            x_enc = torch.cat([x, subject_features], dim=2)

        # autoencoder
        # x_enc = self.encoder_fc(x)
        y_enc, (h_enc, _) = self.encoder(x_enc)
        x_reconn, (h_dec, _) = self.decoder(h_enc.permute(1, 0, 2))  # permute to batch first
        # x_reconn = self.decoder_fc(x_reconn)

        # classification output
        y_cls = self.fc(h_enc[-1, :, :])  # select last state

        # b2b output
        y_b2b, h_b2b = self.b2b(h_enc.permute(1, 0, 2))  # permute to batch first
        # y_b2b = self.decoder_fc(y_b2b)

        return x_enc, x_reconn, y_cls, y_b2b

    def training_step(self, batch, batch_idx):
        X_input, subject_ids, y_b2b, y_cls = batch
        X_enc, X_reconn, y_cls_hat, y_b2b_hat = self(X_input, subject_ids)

        loss_cls = self.loss_cls(y_cls_hat, y_cls)
        loss_reconn = self.loss_reconn(X_reconn, X_enc)
        loss_b2b = self.loss_reconn(y_b2b_hat, y_b2b)
        loss = loss_reconn  # + loss_cls + loss_b2b

        accuracy = (y_cls_hat.argmax(dim=1) == y_cls).float().mean()

        self.log('train/loss_cls', loss_cls)
        self.log('train/loss_reconn', loss_reconn)
        self.log('train/loss_b2b', loss_b2b)
        self.log('train/loss', loss)
        self.log('train/accuracy', accuracy)

        return loss

    def validation_step(self, batch, batch_idx):
        X_input, subject_ids, y_b2b, y_cls = batch
        X_enc, X_reconn, y_cls_hat, y_b2b_hat = self(X_input, subject_ids)

        loss_cls = self.loss_cls(y_cls_hat, y_cls)
        loss_reconn = self.loss_reconn(X_reconn, X_enc)
        loss_b2b = self.loss_reconn(y_b2b_hat, y_b2b)
        loss = loss_reconn  # + loss_cls + loss_b2b

        accuracy = (y_cls_hat.argmax(dim=1) == y_cls).float().mean()

        self.log('val/loss_cls', loss_cls)
        self.log('val/loss_reconn', loss_reconn)
        self.log('val/loss_b2b', loss_b2b)
        self.log('val/loss', loss)
        self.log('val/accuracy', accuracy)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


class Autoencoder(pl.LightningModule):
    def __init__(self, n_features, hidden_size, n_subjects):
        super().__init__()

        self.hidden_size = hidden_size
        self.n_subjects = n_subjects

        # autoencoder
        self.encoder = nn.LSTM(n_features, hidden_size, batch_first=True)
        self.relu = nn.ReLU()
        self.decoder = nn.LSTM(hidden_size, n_features, batch_first=True)
        self.fc_decoder = nn.Linear(n_features, n_features)

        # # b2b head
        # self.b2b_decoder = nn.LSTM(hidden_size, n_features, batch_first=True)
        # self.fc_b2b_decoder = nn.Linear(n_features, n_features)

        # classifier head
        self.cls = nn.Linear(hidden_size, 2)

        # subject classifier head
        self.clf_subject = nn.Linear(hidden_size, n_subjects)

    def forward(self, x):
        batch_size = x.size(0)
        n_timesteps = x.size(1)

        # autoencoder
        y_enc, (h_enc, c_enc) = self.encoder(x)
        x_enc = torch.zeros(batch_size, n_timesteps, self.hidden_size)
        h_enc, c_enc = self.relu(h_enc), self.relu(c_enc)
        y_dec, (h_dec, c_dec) = self.decoder(x_enc, (h_enc, c_enc))
        y_dec = self.relu(y_dec)
        y_dec = self.fc_decoder(y_dec)

        # # b2b head
        # x_enc_b2b = torch.zeros(batch_size, n_timesteps, self.hidden_size)
        # y_b2b, (h_b2b, c_b2b) = self.b2b_decoder(x_enc_b2b, (h_enc, c_enc))
        # y_b2b = self.fc_b2b_decoder(y_b2b)

        # classifier head
        y_cls = self.cls(h_enc[-1, :, :])  # last hidden state of encoder

        # subject classifier head
        y_sub_cls = self.clf_subject(h_enc[-1, :, :])  # last hidden state of encoder

        return y_cls, y_dec, y_sub_cls  # , y_b2b

    def training_step(self, batch, batch_idx):
        X, subject_ids, y_b2b, y_cls = batch
        y_cls_hat, X_recon, y_sub = self(X)

        # loss
        loss_reconn = nn.functional.mse_loss(X_recon, X)
        # loss_b2b = nn.functional.mse_loss(y_b2b_hat, y_b2b)
        loss_cls = nn.functional.cross_entropy(y_cls_hat, y_cls)

        # subject ids classification (for sanity check)
        subject_ids_oh = F.one_hot(subject_ids.squeeze(dim=1), num_classes=self.n_subjects).float()
        loss_cls_sub = nn.functional.cross_entropy(y_sub, subject_ids_oh)

        # total loss
        loss = loss_cls + loss_reconn  # + loss_b2b

        # logging
        self.log('train/loss_reconn', loss_reconn)
        # self.log('train/loss_b2b', loss_b2b)
        self.log('train/loss_cls', loss_cls)
        self.log('train/loss_cls_sub', loss_cls_sub)
        self.log('train/accuracy', (y_cls_hat.argmax(dim=1) == y_cls).float().mean())
        self.log('train/sub_accuracy', (y_sub.argmax(dim=1) == subject_ids).float().mean())
        self.log('train/loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        X, subject_ids, y_b2b, y_cls = batch
        y_cls_hat, X_recon, y_sub = self(X)

        # loss
        loss_reconn = nn.functional.mse_loss(X_recon, X)
        # loss_b2b = nn.functional.mse_loss(y_b2b_hat, y_b2b)
        loss_cls = nn.functional.cross_entropy(y_cls_hat, y_cls)

        # subject ids classification (for sanity check)
        subject_ids_oh = F.one_hot(subject_ids.squeeze(dim=1), num_classes=self.n_subjects).float()
        loss_cls_sub = nn.functional.cross_entropy(y_sub, subject_ids_oh)

        # total loss
        loss = loss_cls + loss_reconn  # + loss_b2b

        # logging
        self.log('val/loss_reconn', loss_reconn)
        # self.log('val/loss_b2b', loss_b2b)
        self.log('val/loss_cls', loss_cls)
        self.log('val/loss_cls_sub', loss_cls_sub)
        self.log('val/accuracy', (y_cls_hat.argmax(dim=1) == y_cls).float().mean(), prog_bar=True)
        self.log('val/sub_accuracy', (y_sub.argmax(dim=1) == subject_ids).float().mean())
        self.log('val/loss', loss)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-2)

    def on_fit_start(self) -> None:
        pl.seed_everything(42)
        torch.nn.init.zeros_(self.fc_decoder.weight)
        torch.nn.init.zeros_(self.fc_decoder.bias)
        torch.nn.init.zeros_(self.cls.weight)
        torch.nn.init.zeros_(self.cls.bias)
