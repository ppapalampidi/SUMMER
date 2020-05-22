import os
import torch

from torch import nn, optim
from torch.utils.data import DataLoader
import pickle
import numpy
import sys
sys.path.append('./')

from modules.data.collates import MovieCollate_CSI
from modules.data.datasets import CSI
from modules.models import SUMMER_unsupervised
from sys_config import MODEL_CNF_DIR, DATA_DIR
from utils.generic import number_h
from utils.opts import train_options

precision_cross = []
recall_cross = []
f1_cross = []
precision_cross_val = []
recall_cross_val = []
f1_cross_val = []

default_config = os.path.join(MODEL_CNF_DIR, "csi_unsupervised.yaml")
opts, config = train_options(default_config)


def _process_batch(model, script, scene_lengths, labels, device):
    y, extracted = model(script, scene_lengths, device)
    return y, extracted, labels


####################################################################
# Data Loading and Preprocessing
####################################################################
assert config["data"]["encoder"] in ["USE", "BERT"]

with open(os.path.join(DATA_DIR,
                       "csi_{}.pickle".format(
                           config["data"]["encoder"])),
          "rb") as f:
    data = pickle.load(f)

all_keys = list(data.keys())

with open(os.path.join(DATA_DIR, "csi_summary_labels.pickle"), "rb") as f:
    labels = pickle.load(f)

dataset = CSI(data, labels, max_scene_length=config["data"]["max_scene_length"])

loader = DataLoader(dataset, shuffle=False, collate_fn=MovieCollate_CSI())

pretrained_model = torch.load('../checkpoints/student.pt',
                              map_location='cpu')

model = SUMMER_unsupervised(config["model"],
                            window_length=config["model"]["context_window"],
                            pretrained_model=pretrained_model,
                            lambda_1=config["model"]["lambda_1"],
                            beta=config["model"]["beta"],
                            compression_rate=config["model"]["compression_rate"],
                            temperature=config["model"]["temperature"])

model.to(opts.device)

print(model)

model.eval()

pre = []
re = []
f1 = []

with torch.no_grad():
    for i_batch, batch in enumerate(loader, 1):
        if isinstance(loader, (tuple, list)):
            batch = list(map(lambda x:
                             list(map(lambda y: y.to(opts.device), x)),
                             batch))
        else:
            batch = list(map(lambda x: x.to(opts.device), batch))

        centralities, extracted, labels = _process_batch(model, *batch,
                                                         opts.device)

        gs_summary = []
        for m, label in enumerate(labels):
            if label == 1:
                gs_summary.append(m)

        right_now = len(list(set(gs_summary).intersection(extracted)))
        overall_now = len(gs_summary)
        predicted_now = len(extracted)
        try:
            recall_now = right_now / overall_now
        except:
            recall_now = 0
        re.append(recall_now)
        try:
            precision_now = right_now / predicted_now
        except:
            precision_now = 0
        pre.append(precision_now)
        try:
            f1.append(
                (2 * precision_now * recall_now) / (precision_now + recall_now))
        except:
            f1.append(0)

print("Final results:")
print(numpy.asarray(pre).mean())
print(numpy.asarray(pre).std())
print(numpy.asarray(re).mean())
print(numpy.asarray(re).std())
print(numpy.asarray(f1).mean())
print(numpy.asarray(f1).std())
