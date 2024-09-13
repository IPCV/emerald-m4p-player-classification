from torch import Tensor
from torch.nn import functional as F
import numpy as np
import torch


def _last_dim_padding(num_dim: int, padding_length: int):
    pad = 2 * num_dim * [0]
    pad[-1] = padding_length
    return tuple(pad)


def _pad_last_dim(tensor: Tensor, padding_length: int, value: int = 0):
    pad = _last_dim_padding(tensor.dim(), padding_length)
    return F.pad(input=tensor, pad=pad, mode='constant', value=value)


class CollateFrames:
    def __init__(self, num_data_outputs: int = 2, padding_token: int = -1):        
        self.num_data_outputs = num_data_outputs
        self.num_classes = 5
        self.padding_token = padding_token

    def __call__(self, data):    
        lengths = np.asarray([len(d[0]) for d in data])
        padding_lengths = np.max(lengths) - lengths

        players, bboxes = [], []
        for datum, pad_length in zip(data, padding_lengths):
            players_batch, detections_batch = datum[:2]
            players.append(_pad_last_dim(players_batch, pad_length))
            bboxes.append(detections_batch)
        players = torch.stack(players)
        return players, bboxes, lengths


def make_seq_mask(lengths):
    batch_size, sequence_size = len(lengths), np.max(lengths)
    seq_mask = torch.zeros((batch_size, sequence_size), dtype=torch.bool)
    for i, l in enumerate(lengths):
        seq_mask[i, l:] = True
    return seq_mask

def unmask_flat_sequences(sequences, lengths):
    sequence_size = np.max(lengths)
    if isinstance(sequences, torch.Tensor):
        sequences = sequences.cpu().numpy()
    return [sequences[idx:idx + lengths[i]] for i, idx in enumerate(range(0, len(sequences), sequence_size))]
