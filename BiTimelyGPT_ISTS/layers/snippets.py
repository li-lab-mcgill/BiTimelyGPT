#! -*- coding: utf-8 -*-
import os
import logging
import numpy as np
import torch
from torch import nn, Tensor
import psutil
import subprocess
import re
from prettytable import PrettyTable


logger = logging.getLogger(__name__)

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


def get_gpu_memory_usage():
    """
    Get the current GPU memory usage.

    Returns:
        dict: A dictionary with 'total', 'used', and 'free' keys (all in MiB).
    """
    try:
        result = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.total,memory.used,memory.free", "--format=csv,noheader,nounits"])
        total, used, free = map(int, re.findall(r'\d+', result.decode('utf-8')))
        return {
            'total': total,
            'used': used,
            'free': free
        }
    except Exception as e:
        print(f"Error querying GPU memory: {e}")
        return {'total': 0, 'used': 0, 'free': 0}



class Transpose(nn.Module):
    """ Wrapper class of torch.transpose() for Sequential module. """
    def __init__(self, shape: tuple):
        super(Transpose, self).__init__()
        self.shape = shape

    def forward(self, x: Tensor) -> Tensor:
        return x.transpose(*self.shape)


class SigmoidRange(nn.Module):
    def __init__(self, low, high):
        super().__init__()
        self.low, self.high = low, high
        # self.low, self.high = ranges
    def forward(self, x):
        # return sigmoid_range(x, self.low, self.high)
        return torch.sigmoid(x) * (self.high - self.low) + self.low



def truncate_sequences(maxlen, *sequences):
    """
    Truncate sequence with maxlen, keep the last maxlen elements
    """
    sequences = [s for s in sequences if s]

    while True:
        # Calculate the lengths of each sequence
        lengths = [len(s) for s in sequences]
        # Check if the sum of lengths exceeds maxlen
        if sum(lengths) > maxlen:
            # Find the index of the longest sequence
            i = np.argmax(lengths)
            # Remove the first element of the longest sequence
            sequences[i].pop(0)
        else:
            return sequences


def sequence_padding(inputs, length=None, value=0, seq_dims=1, mode='post'):
    """
    Padding sequence with same length
    """
    if length is None:
        length = np.max([np.shape(x)[:seq_dims] for x in inputs], axis=0)
    elif not hasattr(length, '__getitem__'):
        length = [length]

    slices = [np.s_[:length[i]] for i in range(seq_dims)]
    slices = tuple(slices) if len(slices) > 1 else slices[0]
    pad_width = [(0, 0) for _ in np.shape(inputs[0])]

    outputs = []
    for x in inputs:
        x = x[slices]
        for i in range(seq_dims):
            if mode == 'post':
                pad_width[i] = (0, length[i] - np.shape(x)[i])
            elif mode == 'pre':
                pad_width[i] = (length[i] - np.shape(x)[i], 0)
            else:
                raise ValueError('"mode" argument must be "post" or "pre".')
        x = np.pad(x, pad_width, 'constant', constant_values=value)
        outputs.append(x)

    return np.array(outputs)


def insert_arguments(**arguments):
    """
    decorator, insert arguments from class
    """
    def actual_decorator(func):
        def new_func(self, *args, **kwargs):
            for k, v in arguments.items():
                if k in kwargs:
                    v = kwargs.pop(k)
                setattr(self, k, v)
            return func(self, *args, **kwargs)

        return new_func

    return actual_decorator


def delete_arguments(*arguments):
    """
    decorator, delete arguments from class
    """
    def actual_decorator(func):
        def new_func(self, *args, **kwargs):
            for k in arguments:
                if k in kwargs:
                    raise TypeError(
                        '%s got an unexpected keyword argument \'%s\'' %
                        (self.__class__.__name__, k)
                    )
            return func(self, *args, **kwargs)

        return new_func

    return actual_decorator

