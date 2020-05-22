import os
import torch

from torch import nn, optim
from torch.utils.data import DataLoader
import pickle
import numpy
import sys
sys.path.append('../')

from models.baseline_trainer import Trainer_SUMMER
from modules.data.collates import MovieCollate_CSI
from modules.data.datasets import CSI
from modules.models import SUMMER_supervised
from sys_config import MODEL_CNF_DIR, DATA_DIR
from utils.generic import number_h
from utils.opts import train_options

####################################################################
# SETTINGS
####################################################################
from utils.training import class_weigths

precision_cross = []
recall_cross = []
f1_cross = []
precision_cross_val = []
recall_cross_val = []
f1_cross_val = []

default_config = os.path.join(MODEL_CNF_DIR,
                              "csi_supervised.yaml")
opts, config = train_options(default_config)

####################################################################
# Data Loading and Preprocessing
####################################################################
assert config["data"]["encoder"] in ["USE"]

with open(os.path.join(DATA_DIR,
                       "csi_{}.pickle".format(
                           config["data"]["encoder"])),
          "rb") as f:
    data = pickle.load(f)

all_keys = list(data.keys())

with open(os.path.join(DATA_DIR, "csi_summary_labels.pickle"), "rb") as f:
    labels = pickle.load(f)

for i in range(10):

    print("Fold: " + str(i+1))

    if i == 0 or i == 1:
        test_keys = all_keys[i * 4:(i * 4 + 4)]
        dev_keys = all_keys[(i + 1) * 4: ((i + 1) * 4 + 8)]
    elif i < 8:
        test_keys = all_keys[i * 4:(i * 4 + 4)]
        dev_keys = all_keys[(i + 1) * 4: ((i + 1) * 4 + 8)]
    elif i < 9:
        test_keys = all_keys[i * 4:(i * 4 + 4)]
        dev_keys = all_keys[(i - 1) * 4 - 4: ((i - 1) * 4 + 4)]
    else:
        test_keys = all_keys[i * 4:]
        dev_keys = all_keys[(i - 1) * 4 - 4: ((i - 1) * 4 + 4)]

    train_data = {k: data[k] for k in list(data) if
                  k not in test_keys and k not in dev_keys}
    dev_data = {k: data[k] for k in list(data) if k in dev_keys}
    test_data = {k: data[k] for k in list(data) if k in test_keys}

    train_labels = {k: labels[k] for k in list(labels) if
                    k not in test_keys and k not in dev_keys}
    dev_labels = {k: labels[k] for k in list(labels) if k in dev_keys}
    test_labels = {k: labels[k] for k in list(labels) if k in test_keys}

    names = [k for k, v in sorted(test_data.items())]

    train_set = CSI(train_data, train_labels, max_scene_length=config["data"][
                                                    "max_scene_length"])

    dev_set = CSI(dev_data, dev_labels, max_scene_length=config["data"][
                                                    "max_scene_length"])

    test_set = CSI(test_data, test_labels, max_scene_length=config["data"][
                                                    "max_scene_length"])

    train_loader = DataLoader(train_set, shuffle=True,
                              collate_fn=MovieCollate_CSI())

    val_loader = DataLoader(dev_set, shuffle=False,
                            collate_fn=MovieCollate_CSI())

    dev_loader = DataLoader(test_set, shuffle=False,
                            collate_fn=MovieCollate_CSI())

    ####################################################################
    # Model
    ####################################################################
    pretrained_model = torch.load('../checkpoints/student.pt',
                                  map_location='cpu')

    model = SUMMER_supervised(config["model"],
                              window_length=config["model"]["context_window"],
                              pretrained_model=pretrained_model,
                              temperature=config["model"]["temperature"])

    model.to(opts.device)

    print(model)

    targets = [z for x in train_set.data for z in x["labels"]]
    weights = class_weigths(targets, to_pytorch=True)
    weights = weights.to(opts.device)
    print(weights)
    pos_weight = weights[0]
    pos_weight.fill_(5)

    loss_function = nn.BCEWithLogitsLoss(reduction='none',
                                         pos_weight=pos_weight)

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(parameters)

    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(p.numel() for p in model.parameters()
                                 if p.requires_grad)

    print("Total Params:", number_h(total_params))
    print("Total Trainable Params:", number_h(total_trainable_params))

    # Trainer: responsible for managing the training process
    trainer = Trainer_SUMMER(val_loader, model,train_loader, dev_loader,
                             loss_function,[optimizer], config, opts.device)

    best_f1 = 0
    best_re = 0
    best_pre = 0
    best_f1_test = 0
    best_re_test = 0
    best_pre_test = 0
    for epoch in range(config["epochs"]):
        train_loss = trainer.train_epoch(config["data"]["mu"],
                                         config["data"]["sigma"], epoch)
        val_loss, val_f1, val_pre, val_re = trainer.eval_epoch(
            config["data"]["mu"], config["data"]["sigma"],
            config["model"]["compression_rate"])
        test_loss, test_f1, test_pre, test_re = trainer.eval_epoch_test(
            config["data"]["mu"], config["data"]["sigma"],
            config["model"]["compression_rate"])
        print("\n")
        print("Train_loss")
        print(train_loss)
        print("\n")
        print("Val_loss")
        print(val_loss)
        print("\n")
        print("Val_f1")
        print(val_f1)
        if val_f1 >= best_f1 and epoch >= 5:
            best_f1 = val_f1
            best_re = val_re
            best_pre = val_pre
            best_f1_test = test_f1
            best_re_test = test_re
            best_pre_test = test_pre
            trainer.checkpoint()

    print("Final dev scores")
    print("Precision:")
    print(best_pre)
    print("Recall:")
    print(best_re)
    print("F1:")
    print(best_f1)

    print("Final test scores")
    print("Precision:")
    print(best_pre_test)
    print("Recall:")
    print(best_re_test)
    print("F1:")
    print(best_f1_test)

    precision_cross_val.append(best_pre)
    recall_cross_val.append(best_re)
    f1_cross_val.append(best_f1)


    precision_cross.append(best_pre_test)
    recall_cross.append(best_re_test)
    f1_cross.append(best_f1_test)

    del model

print("Final cross-validation during val:")
print(numpy.asarray(precision_cross_val).mean())
print(numpy.asarray(precision_cross_val).std())
print(numpy.asarray(recall_cross_val).mean())
print(numpy.asarray(recall_cross_val).std())
print(numpy.asarray(f1_cross_val).mean())
print(numpy.asarray(f1_cross_val).std())

print("Final cross-validation:")
print(numpy.asarray(precision_cross).mean())
print(numpy.asarray(precision_cross).std())
print(numpy.asarray(recall_cross).mean())
print(numpy.asarray(recall_cross).std())
print(numpy.asarray(f1_cross).mean())
print(numpy.asarray(f1_cross).std())
