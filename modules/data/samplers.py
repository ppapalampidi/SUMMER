import math

import numpy
import torch
from torch.utils.data import Sampler


def divide_chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]


class SortedSampler(Sampler):
    """
    Defines a strategy for drawing samples from the dataset,
    in ascending or descending order, based in the sample lengths.
    """

    def __init__(self, lengths, descending=False):
        self.lengths = lengths
        self.desc = descending

    def __iter__(self):

        if self.desc:
            return iter(numpy.flip(numpy.array(self.lengths).argsort(), 0))
        else:
            return iter(numpy.array(self.lengths).argsort())

    def __len__(self):
        return len(self.lengths)


class SceneBatchSampler(Sampler):
    """
    Defines a strategy for drawing batches of samples from the dataset,
    in ascending or descending order, based in the sample lengths.
    """

    def __init__(self, shuffle=False):
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            return iter(self.batches[i]
                        for i in torch.randperm(len(self.batches)))
        else:
            return iter(self.batches)

    def __len__(self):
        return len(self.batches)
