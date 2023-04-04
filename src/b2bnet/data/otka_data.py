import pytorch_lightning as pl
import torch
import xarray as xr
import torch.utils.data as data
from pathlib import Path


class OtkaDataModule(pl.LightningDataModule):
    """Data module to upload input data and split it into train and validation sets.
    """
    def __init__(self,
                 data_dir: Path = Path('data/'),
                 train_ratio: float = 0.8,
                 segment_size: int = 128,
                 batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.train_ratio = train_ratio
        self.segment_size = segment_size
        self.batch_size = batch_size

    def setup(self, stage: str):
        if stage == 'fit':
            train_size = int(len(self.dataset) * self.train_ratio)
            val_size = len(self.dataset) - train_size

        self.train_dataset, self.val_dataset = data.random_split(self.dataset, [train_size, val_size])

    def prepare_data(self):
        # read data from file
        ds = xr.open_dataset(self.data_dir / 'otka.nc5')
        X_input = torch.from_numpy(ds['hypnotee'].values).float().permute(0, 2, 1)
        y_b2b = torch.from_numpy(ds['hypnotist'].values).float().repeat(X_input.shape[0], 1, 1).permute(0, 2, 1)
        y_class = torch.from_numpy(ds['y_class'].values)

        # cleanups
        # truncate y_b2b to match X_input
        y_b2b = y_b2b[:, :X_input.shape[1], :]

        # segment
        X_input = X_input.unfold(1, self.segment_size, self.segment_size).permute(0, 1, 3, 2)
        y_b2b = y_b2b.unfold(1, self.segment_size, self.segment_size).permute(0, 1, 3, 2)

        # create subject ids
        subject_ids = torch.arange(0, X_input.shape[0]).reshape(-1, 1, 1).repeat(1, X_input.shape[1], 1)

        # repeat y_class to match segmentation
        y_class = y_class.reshape(-1, 1, 1).repeat(1, X_input.shape[1], 1)

        # remove subject dimension
        X_input = X_input.flatten(0, 1)
        subject_ids = subject_ids.flatten(0, 1)
        y_b2b = y_b2b.flatten(0, 1)
        y_class = y_class.flatten(0, 1).squeeze(dim=1)

        self.dataset = torch.utils.data.TensorDataset(X_input, subject_ids, y_b2b, y_class)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size)
