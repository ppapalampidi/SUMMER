import os
from numpy import mean
from torch import nn, optim
from torch.utils.data import DataLoader
import pickle

import sys
sys.path.append('./')
from models.baseline_trainer import BaselineTrainer
from modules.data.collates import MovieCollate
from modules.data.datasets import TRIPOD
from modules.models import TAM
from sys_config import MODEL_CNF_DIR, DATA_DIR
from utils.generic import number_h
from utils.opts import train_options

####################################################################
# SETTINGS
####################################################################
from utils.training import class_weigths

default_config = os.path.join(MODEL_CNF_DIR, "teacher.yaml")
opts, config = train_options(default_config)

assert config["data"]["encoder"] in ["USE"]

with open(os.path.join(DATA_DIR, "train_TRIPOD_{}.pickle".
        format(config["data"]["encoder"])), "rb") as f:
    train_data = pickle.load(f)


with open(os.path.join(DATA_DIR, "test_TRIPOD_{}.pickle".
        format(config["data"]["encoder"])), "rb") as f:
    test_data = pickle.load(f)

print("Building training dataset...")
train_set = TRIPOD(train_data,
                      "unsupervised",
                      config["data"]["binary"],
                      config["data"]["mu"],
                      config["data"]["sigma"],
                      max_scene_length=config["data"]["max_scene_length"],
                      pretraining=None)

print("Building evaluation dataset...")
dev_set = TRIPOD(test_data,
                     "supervised", config["data"]["binary"],
                     max_scene_length=config["data"]["max_scene_length"])

test_set = TRIPOD(test_data,
                     "supervised", config["data"]["binary"],
                     max_scene_length=config["data"]["max_scene_length"])

train_loader = DataLoader(train_set, shuffle=True, collate_fn=MovieCollate())
dev_loader = DataLoader(dev_set, shuffle=False, collate_fn=MovieCollate())
val_loader = DataLoader(test_set, shuffle=False, collate_fn=MovieCollate())

####################################################################
# Model
####################################################################

# label  weights
targets = [y for x in train_set.data for z in x["labels_tps"] for y in z]
weights = class_weigths(targets, to_pytorch=True)

model = TAM(config["model"], window_length=config["model"]["context_window"])

model.to(opts.device)
weights = weights.to(opts.device)
pos_weight = weights[1]/weights[0]
pos_weight.fill_(4)

loss_function = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight)
parameters = filter(lambda p: p.requires_grad, model.parameters())
optimizer = optim.Adam(parameters)

total_params = sum(p.numel() for p in model.parameters())
total_trainable_params = sum(p.numel() for p in model.parameters()
                             if p.requires_grad)

print("Total Params:", number_h(total_params))
print("Total Trainable Params:", number_h(total_trainable_params))

####################################################################
# Training Pipeline
####################################################################

trainer = BaselineTrainer(val_loader, model, train_loader, dev_loader,
                          loss_function, [optimizer], config, opts.device)

####################################################################
# Training Loop
####################################################################
best_loss = None
best_score = 0
final_posteriors = []
final_neighborhood = []
groundtruth = []
for epoch in range(config["epochs"]):
    train_loss = trainer.train_epoch(config["data"]["mu"],
                      config["data"]["sigma"], epoch)
    val_loss, posteriors, top_neighborhood = \
        trainer.eval_epoch(config["data"]["mu"],
                      config["data"]["sigma"])
    test_loss, posteriors_test, top_neighborhood_test = \
        trainer.eval_epoch_test(config["data"]["mu"],
                      config["data"]["sigma"])

    ground_truth = [x['groundtruth_indices'] for x in dev_set.data]
    ground_truth_test = [x['groundtruth_indices'] for x in test_set.data]

    score = [sum([len((set(z) & set(w))) for z, w in zip(x, y)]) / len(
        [v for k in x for v in k])
             for x, y in zip(ground_truth, top_neighborhood)]

    score = mean(score)

    score_test = [sum([len((set(z) & set(w))) for z, w in zip(x, y)]) / len(
        [v for k in x for v in k])
             for x, y in zip(ground_truth_test, top_neighborhood_test)]

    score_test = mean(score_test)

    if score >= best_score and epoch > 5:
        best_score = score
        final_posteriors = posteriors_test
        groundtruth = ground_truth_test
        final_neighborhood = top_neighborhood_test

        trainer.checkpoint()
    print("\n")
    print("Train loss")
    print(train_loss)
    print("\n")
    print("Val loss")
    print(val_loss)
    print("\n")
    print("Val TA")
    print(score)
    print("\n" * 2)
