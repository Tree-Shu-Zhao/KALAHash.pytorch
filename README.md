# 🔬 Code and Datasets

## 🎯 Project Overview

This repository contains the code and resources for our paper.

## 💻 System Information

- Debian 12
- Python 3.10
- PyTorch 2.3.1
- CUDA 12.5
- Intel(R) Core(TM) i7-9700F CPU @ 3.00GHz
- NVIDIA GeForce RTX 2070 super

## 🛠 Installation

```bash
conda create -n kalahash python=3.10
conda activate kalahash
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
```

## 📊 Dataset

We use the following folder structure.

```bash
data
├── cifar-10
│   ├── images
│   ├── train.txt
│   ├── query.txt
│   ├── gallery.txt
│   └── classnames.txt
├── nus-wide
│   ├── images
│   ├── train.txt
│   ├── query.txt
│   ├── gallery.txt
│   └── classnames.txt
└── mscoco
    ├── images
    ├── train.txt
    ├── query.txt
    ├── gallery.txt
    └── classnames.txt
```

For CIFAR-10, we download the [dataset](https://www.cs.toronto.edu/~kriz/cifar.html) and convert it to RGB images.

For NUS-WIDE, we obtain it from [DeepHash-pytorch
](https://github.com/swuxyj/DeepHash-pytorch).

For MSCOCO, we use the [MS-COCO2014](https://cocodataset.org/#download).

To ensure reproducibility, we provide low-resource splits with `1shot_split_cache.pkl`, `2shot_split_cache.pkl`, `4shot_split_cache.pkl`, and `8shot_split_cache.pkl` in each dataset folder.

## 🚀 Usage

To train the model:

```bash
python src/main.py experiment=kalahash_cifar10_1s_16b
```

Experiment configurations can be found in `configs/experiment`. Results will be saved in `outputs/<DATE>`.

To evaluate the model:

```bash
python src/main.py experiment=kalahash_cifar10_1s_16b test.EVAL_ONLY=True test.CHECKPOINT_PATH=<CHECKPOINT_PATH>
```


## 🤖 Model

To ensure reproducibility, we provide model checkpoints of KALAHash: https://drive.google.com/drive/folders/1AsQUJ0o3kAKi0a9LBrHcdDVGlKT4X2Zp?usp=sharing

## 📄 Citation

TBD

## Acknowledgement

[CLIP-LoRA](https://github.com/MaxZanella/CLIP-LoRA)
