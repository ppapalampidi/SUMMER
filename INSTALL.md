SUMMER requires

PyTorch 1.10 or better
SciKit
UMAP

To setup with Conda, assuming CUDA 11.6:

```
conda create --name SUMMER
conda activate SUMMER
conda install pip
conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge`
conda install -c conda-forge umap-learn
pip install pyyaml
```
