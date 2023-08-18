import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import xarray as xr
from torch.utils.data import DataLoader
from pathlib import Path
from scipy.signal import butter, sosfiltfilt
from ..utils.timeseries import pad, lag, mask, crop, b2b_data_handler


class OtkaTimeDimSplit(pl.LightningDataModule):
    """Data module to upload input data and split it into train and validation sets on
    time dimension.
    """
    def __init__(self,
                 data_dir: Path = Path('data/'),
                 train_ratio: float = 0.7,
                 segment_size: int = 128,
                 batch_size: int = 32,
                 filter: bool = False,
                 b2b_data: str = 'hypnotist',
                 data_mode: str = 'simple_reconstruction',
                 ):
        super().__init__()
        self.data_dir = data_dir
        self.train_ratio = train_ratio
        self.segment_size = segment_size
        self.batch_size = batch_size
        self.filter = filter

        assert self.data_mode in ['reconn', 'pad', 'lag', 'mask', 'crop']
        assert self.b2b_data in ['hypnotist', 'subject', 'random', None]

    def prepare_data(self):
        # read data from file
        ds = xr.open_dataset(self.data_dir / 'otka.nc5')
        X_input = torch.from_numpy(ds['hypnotee'].values).float().permute(0, 2, 1)
        y_class = torch.from_numpy(ds['y_class'].values)
        
        # b2b head data is None by default unless specified
        X_b2b_test = X_b2b_train = y_b2b_train = y_b2b_test = None

        # cut point for train/test split
        cut_point = int(X_input.shape[1] * self.train_ratio)

        if self.b2b_data == 'random':
            X_b2b = torch.randn(1,
                                X_input.shape[1],
                                X_input.shape[2]).repeat(X_input.shape[0], 1, 1)

        if self.b2b_data == 'subject':
            X_b2b = X_input[0, :, :].repeat(X_input.shape[0] - 1, 1, 1)
            X_input = X_input[1:, :, :]
            y_class = torch.from_numpy(ds['y_class'].values)[1:]

        if self.b2b_data == 'hypnotist':
            X_b2b = torch.from_numpy(ds['hypnotist'].values).float().repeat(X_input.shape[0], 1, 1).permute(0, 2, 1)
            X_b2b = X_b2b[:, :X_input.shape[1], :]  # truncate y_b2b to match X_input

        if self.b2b_data is not None:
            X_b2b = X_b2b.unfold(1, self.segment_size, self.segment_size).permute(0, 1, 3, 2)
            X_b2b_train, y_b2b_train, X_b2b_test, y_b2b_test = b2b_data_handler(X_b2b,
                                                                                self.data_mode,
                                                                                cut_point,
                                                                                self.segment_size)

        ds.close()

        if filter:
            sos = butter(4, [30, 50], 'bp', fs=128, output='sos')
            X_input = torch.from_numpy(sosfiltfilt(sos, X_input, axis=1).copy()).float()
            X_b2b = torch.from_numpy(sosfiltfilt(sos, X_b2b, axis=1).copy()).float()

        # segment
        X_input = X_input.unfold(1, self.segment_size, self.segment_size).permute(0, 1, 3, 2)

        # repeat y_class to match segmentation
        y_class = y_class.reshape(-1, 1, 1).repeat(1, X_input.shape[1], 1)

        # create subject ids
        subject_ids = torch.arange(0, X_input.shape[0]).reshape(-1, 1, 1).repeat(1, X_input.shape[1], 1)

        # train/test split & normalization
        # TODO: should we normalize after flatening? or instead normalize over dimension 1?
        X_train = F.normalize(X_input[:, :cut_point, :, :], dim=2)
        X_test = F.normalize(X_input[:, cut_point:, :, :], dim=2)

        if self.data_mode == 'reconn':
            X_train_in = X_train_out = X_train.flatten(0, 1)
            X_test_in = X_test_out = X_test.flatten(0, 1)

        if self.data_mode == 'pad':
            padding_length = self.segment_size // 6  # zero padding proportional to the segment length
            X_train_in, X_train_out = pad(X_train, padding_length, flatten=True)
            X_test_in, X_test_out = pad(X_test, padding_length, flatten=True)

        if self.data_mode == 'lag':
            # lagged reconstruction
            overlap = self.segment_size // 6
            X_train_in, X_train_out = lag(X_train, overlap=overlap, flatten=True)
            X_test_in, X_test_out = lag(X_test, overlap=overlap, flatten=True)

        if self.data_mode == 'mask':
            mask_length = self.segment_size // 6
            X_train_in, X_train_out = mask(X_train, mask_length=mask_length, flatten=True), X_train.flatten(0, 1)
            X_test_in, X_test_out = mask(X_test, mask_length=mask_length, flatten=True), X_test.flatten(0, 1)

        if self.data_mode == 'crop':
            X_train_in, X_train_out = crop(X_train, crop_length=1, flatten=True), X_train[:, :, -1, :].flatten(0, 1)
            X_test_in, X_test_out = crop(X_test, crop_length=1, flatten=True), X_test[:, :, -1, :].flatten(0, 1)

        self.train_dataset = torch.utils.data.TensorDataset(
            X_train_in,
            X_train_out,
            subject_ids[:, :cut_point, :].flatten(0, 1),
            X_b2b_train,
            y_b2b_train,
            y_class[:, :cut_point, :].flatten(0, 1).squeeze(dim=1)
        )

        self.val_dataset = torch.utils.data.TensorDataset(
            X_test_in,
            X_test_out,
            subject_ids[:, cut_point:, :].flatten(0, 1),
            X_b2b_test,
            y_b2b_test,
            y_class[:, cut_point:, :].flatten(0, 1).squeeze(dim=1)
        )

    def train_dataloader(self):
        rnd_g = torch.Generator()
        rnd_g.manual_seed(42)   # TODO: remove manual seed
        return DataLoader(self.train_dataset, batch_size=self.batch_size, generator=rnd_g)

    def val_dataloader(self):
        rnd_g = torch.Generator()
        rnd_g.manual_seed(42)  # TODO: remove manual seed
        return DataLoader(self.val_dataset, batch_size=self.batch_size, generator=rnd_g)

    def teardown(self, stage: str):
        # Used to clean-up when the run is finished
        ...