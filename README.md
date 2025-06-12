# S-MolSearch: 3D Semi-supervised Contrastive Learning for Bioactive Molecule Search

[[Paper](https://openreview.net/pdf?id=wJAF8TGVUG)], [[Bohrium App](https://bohrium.dp.tech/apps/s-molsearch)]

This repository contains the official implementation of S-MolSearch, a novel semi-supervised contrastive learning framework for molecular search, as presented in our NeurIPS 2024 paper.

<p align="center"><img src="figure/overview.png" width=80%></p>
<p align="center"><b>Overview of the S-MolSearch framework</b></p>

 S-MolSearch leverages inverse optimal transport to integrate limited labeled data with extensive unlabeled data, significantly enhancing the accuracy and efficiency of molecule searches in virtual screening.

Dependencies
------------
 - [Uni-Core](https://github.com/dptech-corp/Uni-Core), check its [Installation Documentation](https://github.com/dptech-corp/Uni-Core#installation).
 - rdkit==2022.9.3, install via `pip install rdkit-pypi==2022.9.3`

Pull the docker image and install rdkit inside:

```bash
docker pull dptechnology/unicore:latest-pytorch2.1.0-cuda12.1-rdma
```

Training
------------

### Configuration

Before training, you need to set the following environment variables or modify the paths in `train.sh`:

```bash
export DATA_PATH="path/to/your/supervised/data"          # sup data directory
export WEIGHT_PATH="path/to/pretrained/weights.pt"     # Pre-trained model weights
export UNSUP_DATA_PATH="path/to/unsupervised/data.lmdb" # unsup data
export LOG_DIR="path/to/logs"                          # Log and checkpoint directory
```

### Start training

Use the provided training script:

```bash
bash train.sh
```


Data and Scripts
------------
Training data, inference and retrieval scripts will be released ASAP.

Citation
------------
If you find this work useful, please cite our paper:
```bibtex
@inproceedings{
zhou2024smolsearch,
title={S-MolSearch: 3D Semi-supervised Contrastive Learning for Bioactive Molecule Search},
author={Gengmo Zhou and Zhen Wang and Feng Yu and Guolin Ke and Zhewei Wei and Zhifeng Gao},
booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
year={2024},
url={https://openreview.net/forum?id=wJAF8TGVUG}
}
```

