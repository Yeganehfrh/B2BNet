import torch.nn.functional as F


def pad(x, pad_length=64,
        flatten=True):
    x = F.pad(x,
              pad=(0, 0, pad_length, 0),
              mode='constant',
              value=0).flatten(0, 1)
    return x[:, :-pad_length, :], x[:, pad_length:, :]


def lag(x, overlap=64, flatten=True):
    overlap = overlap // 2
    mid_point = x.shape[2] // 2  # the length of the time series

    left = x[:, :, :mid_point + overlap, :].flatten(0, 1)
    right = x[:, :, mid_point - overlap:, :].flatten(0, 1)

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
    return x[:, :, :-crop_length, :].flatten(0, 1)


def b2b_data_handler(X_b2b, data_mode, cut_point, segment_size):
    # train/test split & normalization
    b2b_train = F.normalize(X_b2b[:, :cut_point, :, :], dim=2)
    b2b_test = F.normalize(X_b2b[:, cut_point:, :, :], dim=2)

    padding_length = overlap = mask_length = segment_size // 6  # proportional to the segment length

    if data_mode == 'reconn':
        X_b2b_train = b2b_train = b2b_train.flatten(0, 1)
        X_b2b_test = b2b_test = b2b_test.flatten(0, 1)

    if data_mode == 'pad':
        X_b2b_train, y_b2b_train = pad(b2b_train, padding_length, flatten=True)
        X_b2b_test, y_b2b_test = pad(b2b_test, padding_length, flatten=True)

    if data_mode == 'lag':
        X_b2b_train, y_b2b_train = lag(b2b_train, overlap=overlap, flatten=True)
        X_b2b_test, y_b2b_test = lag(b2b_test, overlap=overlap, flatten=True)

    if data_mode == 'mask':
        X_b2b_train, y_b2b_train = mask(b2b_train, mask_length=mask_length, flatten=True), b2b_train.flatten(0, 1)
        X_b2b_test, y_b2b_test = mask(b2b_test, mask_length=mask_length, flatten=True), b2b_test.flatten(0, 1)

    if data_mode == 'crop':
        X_b2b_train, y_b2b_train = crop(b2b_train, crop_length=1), b2b_train[:, :, -1, :].flatten(0, 1)
        X_b2b_test, y_b2b_test = crop(b2b_test, crop_length=1, flatten=True), b2b_test[:, :, -1, :].flatten(0, 1)

    return X_b2b_train, y_b2b_train, X_b2b_test, y_b2b_test
