from itertools import zip_longest

import numpy

import collections


def is_list(seq):
    _list = isinstance(seq, collections.Iterable)
    _str = isinstance(seq, (str, bytes))
    if _list and not _str:
        return True
    else:
        return False


def find_shape(seq):
    if is_list(seq):
        len_ = len(seq)
    else:
        return ()
    shapes = [find_shape(subseq) for subseq in seq]
    shape = (len_,) + tuple(max(sizes) for sizes in zip_longest(*shapes,
                                                                fillvalue=1))
    return shape


def tokenize(text, lowercase=True):
    if lowercase:
        text = text.lower()
    return text.split()


def vectorize_doc(doc, max_sents, max_length, tokenizer=None):
    # trim sentences after max_sents
    doc = doc[:max_sents]
    _doc = numpy.zeros((max_sents, max_length), dtype='int64')
    for i, sent in enumerate(doc):
        s = tokenizer.convert_tokens_to_ids(sent)
        s = pad_sequence(s, max_length)
        _doc[i] = s
    return _doc


def vectorize_doc_sentiment(doc, max_sents, max_length, tokenizer=None):
    # trim sentences after max_sents
    doc = doc[:max_sents]
    _doc = numpy.zeros((max_sents, max_length), dtype='int64')
    for i, sent in enumerate(doc):
        s = pad_sequence(sent, max_length)
        _doc[i] = s
    return _doc


def vectorize_doc_with_vocab(doc, word2idx, max_sents, max_length):
    # trim sentences after max_sents
    doc = doc[:max_sents]
    _doc = numpy.zeros((max_sents, max_length), dtype='int64')
    for i, sent in enumerate(doc):
        s = vectorize(sent, word2idx, max_length)
        _doc[i] = s
    return _doc


def vectorize(sequence, el2idx, max_length, unk_policy="random"):
    """
    Covert array of tokens, to array of ids, with a fixed length
    and zero padding at the end
    Args:
        sequence (): a list of elements
        el2idx (): dictionary of word to ids
        max_length ():
        unk_policy (): how to handle OOV words

    Returns: list of ids with zero padding at the end

    """
    words = numpy.zeros(max_length).astype(int)

    # trim tokens after max length
    sequence = sequence[:max_length]

    for i, token in enumerate(sequence):
        if token in el2idx:
            words[i] = el2idx[token]
        else:
            if unk_policy == "random":
                words[i] = el2idx["<unk>"]
            elif unk_policy == "zero":
                words[i] = 0

    return words

def pad_sequence(sequence, max_length):
    """
    Covert array of tokens, to array of ids, with a fixed length
    and zero padding at the end
    Args:
        sequence (): a list of elements
        el2idx (): dictionary of word to ids
        max_length ():
        unk_policy (): how to handle OOV words

    Returns: list of ids with zero padding at the end

    """
    words = numpy.zeros(max_length).astype(int)

    # trim tokens after max length
    sequence = sequence[:max_length]

    for i, token in enumerate(sequence):
            words[i] = sequence[i]

    return words
