import torch.nn.functional as F


def pad(x, pad_length=64):
    return F.pad(x,
                 pad=(0, 0, pad_length, 0),
                 mode='constant',
                 value=0)


def lag(x, overlap=64, flatten=True):
    overlap = overlap // 2
    mid_point = x.shape[2] // 2  # the length of the time series

    if flatten:
        left = x[:, :, :mid_point + overlap, :].flatten(0, 1),
        right = x[:, :, mid_point - overlap:, :].flatten(0, 1)
    else:
        left = x[:, :, :mid_point + overlap, :]
        right = x[:, :, mid_point - overlap:, :]

    return left, right


def mask(x, mask_length=1, value=1, flatten=True):
    """Mask input tensor by setting the last `mask_length` frames to `value`.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor
    mask_length : int, optional
        length of the frame to mask, defaults to 1
    value : int, optional
        fill value, defaults to 1
    flatten : bool, optional
        flatten the output tensor, defaults to True

    Returns
    -------
    torch.Tensor
        Masked tensor
    """

    x[:, :, -mask_length:, :] = value

    if flatten:
        return x.flatten(0, 1)
    return x


def crop(x, crop_length=1, flatten=True):
    x = x[:, :, -crop_length:, :]

    if flatten:
        return x.flatten(0, 1)
    return x
