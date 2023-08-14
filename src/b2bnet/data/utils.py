import torch


def padding(x, pad_length=64):
    return torch.nn.functional.pad(x,
                                   pad=(0, 0, pad_length, 0),
                                   mode='constant',
                                   value=0)
