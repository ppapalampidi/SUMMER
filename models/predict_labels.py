import os
import torch

from torch.utils.data import DataLoader
import pickle

import sys
sys.path.append('./')
from modules.data.collates import MovieCollate
from modules.data.datasets import TRIPOD
from utils.opts import train_options
from sys_config import MODEL_CNF_DIR, DATA_DIR


def _process_batch(plot, script, tp_ids, scene_lengths, labels, model, device):
    y = model(plot, script, tp_ids, scene_lengths, device)
    return y


if __name__ == '__main__':
    default_config = os.path.join(MODEL_CNF_DIR, "teacher.yaml")
    opts, config = train_options(default_config)

    ####################################################################
    # Data Loading and Preprocessing
    ####################################################################
    assert config["data"]["encoder"] in ["USE"]

    print("Building training dataset...")

    with open(os.path.join(DATA_DIR, "train_TRIPOD_{}.pickle".
            format(config["data"]["encoder"])), "rb") as f:
        train_data = pickle.load(f)

    names = [k for k, value in sorted(train_data.items())]

    print("Training dataset size:")

    train_set = TRIPOD(train_data,
                          "unsupervised",
                          config["data"]["binary"],
                          config["data"]["mu"],
                          config["data"]["sigma"],
                          max_scene_length=config["data"]["max_scene_length"])


    train_loader = DataLoader(train_set, shuffle=False, collate_fn=MovieCollate())

    targets = [y for x in train_set.data for z in x["labels_tps"] for y in z]

    model = torch.load('./checkpoints/teacher.pt', map_location='cpu')
    model.to(opts.device)

    print(model)

    model.eval()
    losses = []

    if isinstance(train_loader, (tuple, list)):
        iterator = zip(*train_loader)
    else:
        iterator = train_loader
    outputs = []
    with torch.no_grad():
        for i_batch, batch in enumerate(iterator, 1):
            # move all tensors in batch to the selected device
            if isinstance(train_loader, (tuple, list)):
                batch = list(map(lambda x:
                                 list(map(lambda y: y.to(opts.device), x)),
                                 batch))
            else:
                batch = list(map(lambda x: x.to(opts.device), batch))

            batch_outputs = _process_batch(*batch, model, opts.device)
            m = torch.nn.Softmax(-1)
            for j in range(batch_outputs.size(0)):
                batch_outputs[j] = m(batch_outputs[j])
            batch_outputs = batch_outputs.data.cpu().numpy()
            outputs.append(batch_outputs)

        mu = config["data"]["mu"]
        sigma = config["data"]["sigma"]
        final_posteriors = []
        final_neighborhood = []
        final_preds = {}
        final_distributions = {}
        for w, movie in enumerate(outputs):
            final_posteriors.append([])
            final_neighborhood.append([])
            posteriors = movie.tolist()
            indices = list(range(len(posteriors[0])))
            for j, tp in enumerate(posteriors):
                final_posteriors[-1].append(tp)
                indices_of_w = [x for _, x in sorted(zip(tp, list(range(len(tp)))), reverse=True)]
                top_post_index = indices_of_w[0]
                if top_post_index != 0 and top_post_index != (len(tp) - 1):
                    top_neighborhood = indices[(top_post_index-1):(top_post_index+2)]
                elif top_post_index == 0:
                    top_neighborhood = indices[top_post_index:(top_post_index+3)]
                else:
                    top_neighborhood = indices[(top_post_index-2):(top_post_index+1)]
                final_neighborhood[-1].append(top_neighborhood)
            final_preds.update({names[w]: final_neighborhood[-1]})
            final_distributions.update({names[w]: final_posteriors[-1]})

    with open('./dataset/labels_train_TRIPOD_silver.pickle', 'wb') as f:
        pickle.dump(final_preds, f, protocol=pickle.HIGHEST_PROTOCOL)
