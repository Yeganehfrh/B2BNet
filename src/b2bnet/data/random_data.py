import pytorch_lightning as pl
import torch


class RandomDataModule(pl.LightningDataModule):
    """Data module to generate random data for testing purposes.
    """

    def prepare_data(self):
        X_input = torch.randn(50, 128, 58)
        y_b2b = torch.randn(1, 128, 58).repeat(50, 1, 1)
        y_class = torch.randint(low=0, high=3, size=(50,))
        y_text = torch.randn(50, 1024)

        self.train_dataset = torch.utils.data.TensorDataset(X_input, y_b2b, y_class, y_text)
        self.val_dataset = torch.utils.data.TensorDataset(X_input, y_b2b, y_class, y_text)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=32)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=32)
