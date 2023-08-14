import torch


def padding(x, pad_length=64):
    return torch.nn.functional.pad(x,
                                   pad=(0, 0, pad_length, 0),
                                   mode='constant',
                                   value=0)


def lag(x, overlap=64, flatten=True):
    overlap = int(overlap/2)
    mid_point = int(x.shape[2]/2)  # the length of the time series
    if flatten:
        return x[:, :, :mid_point+overlap, :].flatten(0, 1), x[:, :, mid_point-overlap:, :].flatten(0, 1)
    else:
        return x[:, :, :mid_point+overlap, :], x[:, :, mid_point-overlap:, :]
