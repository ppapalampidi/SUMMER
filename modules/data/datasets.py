import numpy
from torch.utils.data import Dataset


class TRIPOD(Dataset):
    def __init__(self, input, mode, binary, mu=None, sigma=None,
                 max_scene_length=50, noisy_labels=None, **kwargs):
        """
        Dataset creation
        """
        assert mode in ["unsupervised", "supervised"]

        self.mu = mu
        self.sigma = sigma

        self.data = input

        if noisy_labels != None:
            self._assign_noisy_labels(noisy_labels)
        else:
            if mode == "supervised":
                self._gold_labels()
            elif mode == "unsupervised":
                self._silver_labels()
            else:
                raise ValueError

        data = [v for k, v in sorted(self.data.items())]

        self.data = []
        for i, x in enumerate(data):
            scenes_embeddings = [_x[:max_scene_length]
                                 for _x in x["scenes_embeddings"] if len(_x) > 0]
            self.data.append(x)
            self.data[-1]["scenes_embeddings"] = scenes_embeddings
        print(len(self.data))


    def _silver_labels(self):
        for k, v in self.data.items():
            ranges = [(round((m - s) * len(v["scenes_embeddings"])),
                       round((m + s) * len(v["scenes_embeddings"])))
                      for m, s in zip(self.mu, self.sigma)]
            self.data[k]["label_ranges"] = ranges

            bin_labels = [[1 if r[0] <= i <= r[1] else 0
                           for i, s in enumerate(v["scenes_embeddings"])]
                          for j, r in enumerate(ranges)]
            self.data[k]["labels_tps"] = bin_labels


    def _gold_labels(self):
        for k, v in self.data.items():
            labels = numpy.zeros((len(v['turning_points']), len(v['scenes'])))
            for i, ids in enumerate(v['groundtruth_indices']):
                for id in ids:
                    labels[i][id] = 1
            self.data[k]["labels_tps"] = labels


    def _assign_noisy_labels(self, noisy_labels):
        cnt = 0
        for k, v in noisy_labels.items():
            labels = numpy.zeros((len(v), len(self.data[k]['scenes_embeddings'])))
            for i, ids in enumerate(v):
                for id in ids:
                    labels[i][id] = 1
            self.data[k]["labels_tps"] = labels
            cnt += 1

    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):
        sample = self.data[index]

        plot = sample["plot_embeddings"]
        script = sample["scenes_embeddings"]
        labels = sample['labels_tps']
        tp_ids = sample['turning_points']
        return plot, script, tp_ids, [len(x) for x in script], labels


class CSI(Dataset):
    def __init__(self, input, labels, max_scene_length=50,**kwargs):
        """
        CSI dataset for summarization
        """
        self.data = input
        self._assign_labels(labels)

        data = [v for k, v in sorted(self.data.items())]

        self.data = []
        for i, x in enumerate(data):
            scenes_embeddings = [_x[:max_scene_length]
                                 for _x in x["scenes_embeddings"] if
                                 ((not numpy.isnan(_x).any()) and len(_x) != 0)]
            self.data.append(x)
            self.data[-1]["scenes_embeddings"] = scenes_embeddings

    def _assign_labels(self, labels):
        for k, v in labels.items():
            assigned_labels = numpy.zeros(len(self.data[k]['scenes_embeddings']))
            for i, id in enumerate(v):
                assigned_labels[id] = 1

            self.data[k]["labels"] = assigned_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]

        script = sample["scenes_embeddings"]
        labels = sample['labels']
        return script, [len(x) for x in script], labels