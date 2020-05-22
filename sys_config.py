import os

import torch

print("torch:", torch.__version__)
print("Cuda:", torch.backends.cudnn.cuda)
print("CuDNN:", torch.backends.cudnn.version())

CPU_CORES = 4
RANDOM_SEED = 1618

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_CNF_DIR = os.path.join(BASE_DIR, "model_configs")

TRAINED_PATH = os.path.join(BASE_DIR, "checkpoints")

DATA_DIR = os.path.join(BASE_DIR, 'dataset')

MODEL_DIRS = ["models", "modules", "utils"]
