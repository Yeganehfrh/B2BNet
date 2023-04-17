import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import xarray as xr
import torch.utils.data as data  # noqa
from pathlib import Path
from sklearn.model_selection import train_test_split


class OtkaDataModule(pl.LightningDataModule):
    """Data module to upload input data and split it into train and validation sets.
    """
    def __init__(self,
                 data_dir: Path = Path('data/'),
                 train_ratio: float = 0.7,
                 segment_size: int = 128,
                 batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.train_ratio = train_ratio
        self.segment_size = segment_size
        self.batch_size = batch_size

    def prepare_data(self):
        # read data from file
        ds = xr.open_dataset(self.data_dir / 'otka.nc5')
        X_input = torch.from_numpy(ds['hypnotee'].values).float().permute(0, 2, 1)
        y_b2b = torch.from_numpy(ds['hypnotist'].values).float().repeat(X_input.shape[0], 1, 1).permute(0, 2, 1)
        y_class = torch.from_numpy(ds['y_class'].values)

        ds.close()

        n_subjects = X_input.shape[0]
        train_ids, val_ids = train_test_split(torch.arange(0, n_subjects),
                                              train_size=self.train_ratio,
                                              stratify=y_class)

        # normalize
        X_input = F.normalize(X_input, dim=2)
        y_b2b = F.normalize(y_b2b, dim=2)

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
        subject_ids = subject_ids.flatten(0, 1)

        train_idx = torch.where(torch.isin(subject_ids, train_ids))[0]
        val_idx = torch.where(torch.isin(subject_ids, val_ids))[0]

        # continue removing subject dimension
        X_input = X_input.flatten(0, 1)
        y_b2b = y_b2b.flatten(0, 1)
        y_class = y_class.flatten(0, 1).squeeze(dim=1)

        self.train_dataset = torch.utils.data.TensorDataset(X_input[train_idx], subject_ids[train_idx],
                                                            y_b2b[train_idx], y_class[train_idx])

        self.val_dataset = torch.utils.data.TensorDataset(X_input[val_idx], subject_ids[val_idx],
                                                          y_b2b[val_idx], y_class[val_idx])

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size)


class OtkaTimeDimSplit(pl.LightningDataModule):
    """Data module to upload input data and split it into train and validation sets on
    time dimension.
    """
    def __init__(self,
                 data_dir: Path = Path('data/'),
                 train_ratio: float = 0.7,
                 segment_size: int = 128,
                 batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.train_ratio = train_ratio
        self.segment_size = segment_size
        self.batch_size = batch_size

    def prepare_data(self):
        # read data from file
        ds = xr.open_dataset(self.data_dir / 'otka.nc5')
        X_input = torch.from_numpy(ds['hypnotee'].values).float().permute(0, 2, 1)
        y_b2b = torch.from_numpy(ds['hypnotist'].values).float().repeat(X_input.shape[0], 1, 1).permute(0, 2, 1)
        y_class = torch.from_numpy(ds['y_class'].values)

        ds.close()

        # segment
        X_input = X_input.unfold(1, self.segment_size, self.segment_size).permute(0, 1, 3, 2)
        y_b2b = y_b2b.unfold(1, self.segment_size, self.segment_size).permute(0, 1, 3, 2)

        # repeat y_class to match segmentation
        y_class = y_class.reshape(-1, 1, 1).repeat(1, X_input.shape[1], 1)

        # normalize
        X_input = F.normalize(X_input, dim=2)
        y_b2b = F.normalize(y_b2b, dim=2)

        # cleanups
        # truncate y_b2b to match X_input
        y_b2b = y_b2b[:, :X_input.shape[1], :]

        # create subject ids
        subject_ids = torch.arange(0, X_input.shape[0]).reshape(-1, 1, 1).repeat(1, X_input.shape[1], 1)

        # cut point for train/test split
        cut_point = int(X_input.shape[1] * self.train_ratio)

        self.train_dataset = torch.utils.data.TensorDataset(X_input[:, :cut_point, :, :].flatten(0, 1),
                                                            subject_ids[:, :cut_point, :].flatten(0, 1),
                                                            y_b2b[:, :cut_point, :, :].flatten(0, 1),
                                                            y_class[:, :cut_point, :].flatten(0, 1).squeeze(dim=1))

        self.val_dataset = torch.utils.data.TensorDataset(X_input[:, cut_point:, :, :].flatten(0, 1),
                                                          subject_ids[:, cut_point:, :].flatten(0, 1),
                                                          y_b2b[:, cut_point:, :, :].flatten(0, 1),
                                                          y_class[:, cut_point:, :].flatten(0, 1).squeeze(dim=1))

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size)
