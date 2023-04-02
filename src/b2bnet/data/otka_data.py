import pytorch_lightning as pl
import torch
import xarray as xr
import torch.utils.data as data
from pathlib import Path


class OtkaDataModule(pl.LightningDataModule):
    """Data module to upload input data and split it into train and validation sets.
    """
    def __init__(self, data_dir: Path = Path('data/'), train_ratio: float = 0.8):
        super().__init__()
        self.data_dir = data_dir
        self.train_ratio = train_ratio

    def setup(self, stage: str):
        if stage == 'fit':
            train_size = int(len(self.dataset) * self.train_ratio)
            val_size = len(self.dataset) - train_size

        self.train_dataset, self.val_dataset = data.random_split(self.dataset, [train_size, val_size])

    def prepare_data(self):
        ds = xr.open_dataset(self.data_dir / 'otka.nc5')
        X_input = torch.from_numpy(ds['hypnotee'].values).float().permute(0, 2, 1)
        y_b2b = torch.from_numpy(ds['hypnotist'].values).float().repeat(X_input.shape[0], 1, 1).permute(0, 2, 1)
        y_class = torch.from_numpy(ds['y_class'].values)
        y_text = torch.zeros_like(y_class)

        # X_input = X_input[:, :100, :]
        # y_b2b = y_b2b[:, :100, :]

        # X_input = torch.split(X_input[:, :39700, :], split_size_or_sections=100, dim=1)
        # y_b2b = torch.split(y_b2b, split_size_or_sections=100, dim=1)

        self.dataset = torch.utils.data.TensorDataset(X_input, y_b2b, y_class, y_text)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=32)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=32)
