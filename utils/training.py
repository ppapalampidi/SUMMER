import datetime
import numpy
import os

from sklearn.utils import compute_class_weight
import torch

from sys_config import BASE_DIR


def sort_by_lengths(lengths):
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


def save_checkpoint(state, name, path=None, timestamp=False, tag=None,
                    verbose=False, model=None):
    """
    Save a trained model, along with its optimizer, in order to be able to
    resume training
    Args:
        path (str): the directory, in which to save the checkpoints
        timestamp (bool): whether to keep only one model (latest), or keep every
            checkpoint

    Returns:

    """
    now = datetime.datetime.now().strftime("%y-%m-%d_%H:%M:%S")

    if tag is not None:
        if isinstance(tag, str):
            name += "_{}".format(tag)
        elif isinstance(tag, list):
            for t in tag:
                name += "_{}".format(t)
        else:
            raise ValueError("invalid tag type!")

    if timestamp:
        name += "_{}".format(now)

    name += ".pt"

    if path is None:
        path = os.path.join(BASE_DIR, "checkpoints")

    if not os.path.isdir(path):
        os.mkdir(path)

    file = os.path.join(path, name)

    if verbose:
        print("saving checkpoint:{} ...".format(name))

    # torch.save(state, file)

    if model != None:
        torch.save(model, file)

    return name


def load_checkpoint(name, path=None, device=None):
    """
    Load a trained model, along with its optimizer
    Args:
        name (str): the name of the model
        path (str): the directory, in which the model is saved

    Returns:
        model, optimizer

    """
    if path is None:
        path = os.path.join(BASE_DIR, "checkpoints")

    model_fname = os.path.join(path, "{}.pt".format(name))

    print("Loading checkpoint `{}` ...".format(model_fname), end=" ")

    with open(model_fname, 'rb') as f:
        state = torch.load(f, map_location="cpu")

    print("done!")

    return state


def get_class_labels(y):
    """
    Get the class labels
    :param y: list of labels, ex. ['positive', 'negative', 'positive',
                                    'neutral', 'positive', ...]
    :return: sorted unique class labels
    """
    return numpy.unique(y)


def get_class_weights(y):
    """
    Returns the normalized weights for each class
    based on the frequencies of the samples
    :param y: list of true labels (the labels must be hashable)
    :return: dictionary with the weight for each class
    """

    weights = compute_class_weight(class_weight = 'balanced', classes = numpy.unique(y), y=y)

    d = {c: w for c, w in zip(numpy.unique(y), weights)}

    return d


def class_weigths(targets, to_pytorch=False):
    w = get_class_weights(targets)
    labels = get_class_labels(targets)
    if to_pytorch:
        return torch.FloatTensor([w[l] for l in sorted(labels)])
    return labels
