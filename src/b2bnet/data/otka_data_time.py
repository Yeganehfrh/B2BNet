import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import xarray as xr
import torch.utils.data as data  # noqa
from pathlib import Path
from scipy.signal import butter, sosfiltfilt


class OtkaTimeDimSplit(pl.LightningDataModule):
    """Data module to upload input data and split it into train and validation sets on
    time dimension.
    """
    def __init__(self,
                 data_dir: Path = Path('data/'),
                 train_ratio: float = 0.7,
                 segment_size: int = 128,
                 batch_size: int = 32,
                 filter: bool = False):
        super().__init__()
        self.data_dir = data_dir
        self.train_ratio = train_ratio
        self.segment_size = segment_size
        self.batch_size = batch_size
        self.filter = filter

    def prepare_data(self):
        # read data from file
        ds = xr.open_dataset(self.data_dir / 'otka.nc5')
        X_input = torch.from_numpy(ds['hypnotee'].values).float().permute(0, 2, 1)
        # y_b2b = torch.from_numpy(ds['hypnotist'].values).float().repeat(X_input.shape[0], 1, 1).permute(0, 2, 1)

        # TODO: remove and modify the following three lines (using one of the subjects data as y_b2b)
        y_b2b = X_input[0, :, :].repeat(X_input.shape[0]-1, 1, 1)
        X_input = X_input[1:, :, :]
        y_class = torch.from_numpy(ds['y_class'].values)[1:]

        ds.close()

        if filter:
            sos = butter(4, [30, 50], 'bp', fs=128, output='sos')
            X_input = torch.from_numpy(sosfiltfilt(sos, X_input, axis=1).copy()).float()
            y_b2b = torch.from_numpy(sosfiltfilt(sos, y_b2b, axis=1).copy()).float()

        # segment
        X_input = X_input.unfold(1, self.segment_size, self.segment_size).permute(0, 1, 3, 2)
        y_b2b = y_b2b.unfold(1, self.segment_size, self.segment_size).permute(0, 1, 3, 2)

        # repeat y_class to match segmentation
        y_class = y_class.reshape(-1, 1, 1).repeat(1, X_input.shape[1], 1)

        # cleanups
        # truncate y_b2b to match X_input
        # y_b2b = y_b2b[:, :X_input.shape[1], :]

        # create subject ids
        subject_ids = torch.arange(0, X_input.shape[0]).reshape(-1, 1, 1).repeat(1, X_input.shape[1], 1)

        # cut point for train/test split
        cut_point = int(X_input.shape[1] * self.train_ratio)

        self.train_dataset = torch.utils.data.TensorDataset(
            F.normalize(X_input[:, :cut_point, :, :], dim=2).flatten(0, 1),
            subject_ids[:, :cut_point, :].flatten(0, 1),
            F.normalize(y_b2b[:, :cut_point, :, :], dim=2).flatten(0, 1),
            y_class[:, :cut_point, :].flatten(0, 1).squeeze(dim=1)
            )

        self.val_dataset = torch.utils.data.TensorDataset(
            F.normalize(X_input[:, cut_point:, :, :], dim=2).flatten(0, 1),
            subject_ids[:, cut_point:, :].flatten(0, 1),
            F.normalize(y_b2b[:, cut_point:, :, :], dim=2).flatten(0, 1),
            y_class[:, cut_point:, :].flatten(0, 1).squeeze(dim=1)
            )

    def train_dataloader(self):
        rnd_g = torch.Generator()
        rnd_g.manual_seed(42)   # TODO: remove manual seed
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, generator=rnd_g)

    def val_dataloader(self):
        rnd_g = torch.Generator()
        rnd_g.manual_seed(42)  # TODO: remove manual seed
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, generator=rnd_g)
