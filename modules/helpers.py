import torch
from torch.nn import functional as F


def sequence_mask(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .unsqueeze(0).expand(batch_size, max_len)
            .lt(lengths.unsqueeze(1)))


def pairwise_distance(a, b, dist="euclidean"):
    if dist == "euclidean":
        return F.pairwise_distance(a, b).mean()
    elif dist == "cosine":
        return 1 - F.cosine_similarity(a, b).mean()
    else:
        raise ValueError


def masked_normalization(logits, mask):
    scores = F.softmax(logits, dim=-1)

    # apply the mask - zero out masked timesteps
    masked_scores = scores * mask.float()

    # re-normalize the masked scores
    normed_scores = masked_scores.div(masked_scores.sum(-1, keepdim=True))

    return normed_scores


def masked_mean(vecs, mask):
    masked_vecs = vecs * mask.float()

    mean = masked_vecs.sum(1) / mask.sum(1)

    return mean


def masked_normalization_inf(logits, mask):
    logits.masked_fill_(1 - mask, float('-inf'))
    # energies.masked_fill_(1 - mask, -1e18)

    scores = F.softmax(logits, dim=-1)

    return scores


def avg_vectors(vectors, mask, energies=None, position=False):
    if energies is None:
        centroid = masked_mean(vectors, mask)
        return centroid, None

    else:
        # scores = F.softmax(energies * mask, dim=1)
        masked_scores = energies * mask.float()
        normed_scores = masked_scores.div(masked_scores.sum(1, keepdim=True))

        if position:
            pos_dist = torch.arange(mask.size(1), 0, -1, device=mask.device)
            pos_dist = pos_dist.float() / pos_dist.sum().float()
            pos_dist = pos_dist.repeat(mask.size(0), 1).unsqueeze(-1)

            masked_pos_dist = pos_dist * mask.float()
            normed_pos_dist = masked_pos_dist.div(
                masked_pos_dist.sum(1, keepdim=True))

            normed_scores = normed_pos_dist + normed_scores
            normed_scores = normed_scores.div(
                normed_scores.sum(1, keepdim=True))

        centroid = (vectors * normed_scores).sum(1)
    return centroid, normed_scores


def aeq(*args):
    """
    Assert all arguments have the same value
    """
    arguments = (arg for arg in args)
    first = next(arguments)
    assert all(arg == first for arg in arguments), \
        "Not all arguments have the same value: " + str(args)
