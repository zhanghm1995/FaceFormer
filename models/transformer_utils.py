'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-02-27 09:46:26
Email: haimingzhang@link.cuhk.edu.cn
Description: Some useful utilites codes in Transformer network
'''

import torch
from torch import Tensor


def generate_shifted_target(target: Tensor):
    """_summary_

    Args:
        target (Tensor): (Sy, B, C)

    Returns:
        _type_: shifted target with a inserted start token
    """
    ret = torch.zeros_like(target)
    ret[1:, ...] = target[:-1, ...]
    return ret


def generate_subsequent_mask(seq_len):
    """Generate future masked matrix

    Args:
        seq_len int): sequence length

    Returns:
        Tensor: (seq_len, seq_len) dimension
    """
    mask = (torch.triu(torch.ones(seq_len, seq_len)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def generate_key_mapping_mask(tgt, lengths):
    """Generate mask matrix according valid lengths vector

    Args:
        tgt (Tensor): (Sy, B, C)
        lengths (Tensor): (B, )

    Returns:
        Tensor: (B, Sy)
    """
    if lengths is None:
        return None
    max_len, batch_size , _ = tgt.shape
    mask = torch.arange(max_len, device=lengths.device).expand(batch_size, max_len) >= lengths[:, None]
    return mask