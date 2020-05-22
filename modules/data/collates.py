import torch
from torch.nn.utils.rnn import pad_sequence


class MovieCollate:
    def __init__(self, *args):
        super().__init__(*args)

    def pad_samples(self, vecs):
        return pad_sequence([torch.FloatTensor(x) for x in vecs], True)

    def _collate(self, plot, script, tp_ids, scene_lengths, labels):
        """
        Important: Assumes batch_size = 1
        """
        try:
            plot = torch.FloatTensor(plot)
        except:
            print()
        script = self.pad_samples(script)
        tp_ids = torch.LongTensor(tp_ids)
        scene_lengths = torch.LongTensor(scene_lengths)
        labels = torch.FloatTensor(labels)
        return plot, script, tp_ids, scene_lengths, labels

    def __call__(self, batch):
        assert len(batch) == 1, "Works only with batch_size = 1!"
        return self._collate(*batch[0])


class MovieCollate_CSI:
    def __init__(self, *args):
        super().__init__(*args)

    def pad_samples(self, vecs):
        return pad_sequence([torch.FloatTensor(x) for x in vecs], True)

    def _collate(self, script, scene_lengths, labels):
        """
        Important: Assumes batch_size = 1
        """
        script = self.pad_samples(script)
        scene_lengths = torch.LongTensor(scene_lengths)
        labels = torch.FloatTensor(labels)
        return script, scene_lengths, labels

    def __call__(self, batch):
        assert len(batch) == 1, "Works only with batch_size = 1!"
        return self._collate(*batch[0])


class MovieCollate_tps_condition_w_chars:
    def __init__(self, *args):
        super().__init__(*args)

    def pad_samples(self, vecs):
        return pad_sequence([torch.FloatTensor(x) for x in vecs], True)

    def _collate(self, plot, script, tp_ids, scene_lengths, tps_scene_indices, primary, secondary, labels):
        """
        Important: Assumes batch_size = 1
        """
        try:
            plot = torch.FloatTensor(plot)
        except:
            print()
        script = self.pad_samples(script)
        tp_ids = torch.LongTensor(tp_ids)
        scene_lengths = torch.LongTensor(scene_lengths)
        tps_scene_indices = torch.LongTensor(tps_scene_indices)
        primary = torch.FloatTensor(primary)
        secondary = torch.FloatTensor(secondary)
        labels = torch.FloatTensor(labels)
        return plot, script, tp_ids, scene_lengths, tps_scene_indices, primary, secondary, labels

    def __call__(self, batch):
        assert len(batch) == 1, "Works only with batch_size = 1!"
        return self._collate(*batch[0])


class MovieCollate_tps_condition_multimodal:
    def __init__(self, *args):
        super().__init__(*args)

    def pad_samples(self, vecs):
        return pad_sequence([torch.FloatTensor(x) for x in vecs], True)

    def _collate(self, plot, script, tp_ids, scene_lengths, tps_scene_indices,
                 audio, vision, labels):
        """
        Important: Assumes batch_size = 1
        """
        try:
            plot = torch.FloatTensor(plot)
        except:
            print()
        script = self.pad_samples(script)
        tp_ids = torch.LongTensor(tp_ids)
        scene_lengths = torch.LongTensor(scene_lengths)
        tps_scene_indices = torch.LongTensor(tps_scene_indices)
        audio = torch.FloatTensor(audio)
        vision = torch.FloatTensor(vision)
        labels = torch.FloatTensor(labels)
        return plot, script, tp_ids, scene_lengths, tps_scene_indices, audio, \
               vision, labels

    def __call__(self, batch):
        assert len(batch) == 1, "Works only with batch_size = 1!"
        return self._collate(*batch[0])


class MovieCollate_tps_condition_aspects:
    def __init__(self, *args):
        super().__init__(*args)

    def pad_samples(self, vecs):
        return pad_sequence([torch.FloatTensor(x) for x in vecs], True)

    def _collate(self, plot, script, tp_ids, scene_lengths, tps_scene_indices, aspect_flags, labels):
        """
        Important: Assumes batch_size = 1
        """
        try:
            plot = torch.FloatTensor(plot)
        except:
            print()
        script = self.pad_samples(script)
        tp_ids = torch.LongTensor(tp_ids)
        scene_lengths = torch.LongTensor(scene_lengths)
        tps_scene_indices = torch.LongTensor(tps_scene_indices)
        aspect_flags = torch.FloatTensor(aspect_flags)
        labels = torch.FloatTensor(labels)
        return plot, script, tp_ids, scene_lengths, tps_scene_indices, aspect_flags, labels

    def __call__(self, batch):
        assert len(batch) == 1, "Works only with batch_size = 1!"
        return self._collate(*batch[0])

