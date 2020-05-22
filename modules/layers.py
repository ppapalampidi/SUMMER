import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable

from modules.helpers import sequence_mask, masked_normalization_inf, \
    masked_normalization


class GaussianNoise(nn.Module):
    def __init__(self, stddev, mean=.0):
        """
        Additive Gaussian Noise layer
        Args:
            stddev (float): the standard deviation of the distribution
            mean (float): the mean of the distribution
        """
        super().__init__()
        self.stddev = stddev
        self.mean = mean
        self.mean = mean

    def forward(self, x):
        if self.training:
            noise = Variable(x.data.new(x.size()).normal_(self.mean,
                                                          self.stddev))
            return x + noise
        return x

    def __repr__(self):
        return '{} (mean={}, stddev={})'.format(self.__class__.__name__,
                                                str(self.mean),
                                                str(self.stddev))


class MLP(nn.Module):
    def __init__(self, input_size,
                 layers=1,
                 dropout=.0,
                 non_linearity="tanh"):
        super(MLP, self).__init__()

        if non_linearity == "relu":
            activation = nn.ReLU()
        elif non_linearity == "tanh":
            activation = nn.Tanh()
        else:
            raise ValueError("Unsupported non_linearity!")

        modules = []
        for i in range(layers - 1):
            modules.append(nn.Linear(input_size, input_size))
            modules.append(activation)
            modules.append(nn.Dropout(dropout))

        # last attention layer must output 1
        modules.append(nn.Linear(input_size, 1))
        modules.append(activation)
        modules.append(nn.Dropout(dropout))

        self.layer = nn.Sequential(*modules)

    def forward(self, x):

        y = self.layer(x)

        return y


class SelfAttention(nn.Module):
    def __init__(self, attention_size,
                 batch_first=True,
                 layers=1,
                 dropout=.0,
                 non_linearity="tanh"):
        super(SelfAttention, self).__init__()

        self.batch_first = batch_first

        if non_linearity == "relu":
            activation = nn.ReLU()
        else:
            activation = nn.Tanh()

        modules = []
        for i in range(layers - 1):
            modules.append(nn.Linear(attention_size, attention_size))
            modules.append(activation)
            modules.append(nn.Dropout(dropout))

        # last attention layer must output 1
        modules.append(nn.Linear(attention_size, 1))
        modules.append(activation)
        modules.append(nn.Dropout(dropout))

        self.attention = nn.Sequential(*modules)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sequence, lengths):

        energies = self.attention(sequence).squeeze()

        # construct a mask, based on sentence lengths
        mask = sequence_mask(lengths, energies.size(1))

        # scores = masked_normalization_inf(energies, mask)
        scores = masked_normalization(energies, mask)

        contexts = (sequence * scores.unsqueeze(-1)).sum(1)

        return contexts, scores


class ModelHelper:
    def sort_by(self, lengths):
        """
        Sort batch data and labels by length.
        Useful for variable length inputs, for utilizing PackedSequences
        Args:
            lengths (nn.Tensor): tensor containing the lengths for the data

        Returns:
            - sorted lengths Tensor
            - sort (callable) which will sort a given iterable
                according to lengths
            - unsort (callable) which will revert a given iterable to its
                original order

        """
        batch_size = lengths.size(0)

        sorted_lengths, sorted_idx = lengths.sort()
        _, original_idx = sorted_idx.sort(0, descending=True)
        reverse_idx = torch.linspace(batch_size - 1, 0, batch_size).long()

        if lengths.data.is_cuda:
            reverse_idx = reverse_idx.cuda()

        sorted_lengths = sorted_lengths[reverse_idx]

        def sort(iterable):
            if len(iterable.shape) > 1:
                return iterable[sorted_idx.data][reverse_idx]
            else:
                return iterable

        def unsort(iterable):
            if len(iterable.shape) > 1:
                return iterable[reverse_idx][original_idx][reverse_idx]
            else:
                return iterable

        return sorted_lengths, sort, unsort