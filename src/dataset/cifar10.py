import os
import pickle
import random
from collections import defaultdict

import numpy as np
import torch
from loguru import logger
from PIL import Image
from torch.utils.data import Dataset


def split_few_shot(train_items, num_shots):
    assert num_shots and num_shots > 0 and isinstance(num_shots, int), "num_shots must be an integer and greater than 0"

    # Group data by label
    grouped_data_by_label = defaultdict(list)
    for item in train_items:
        one_hot_label = item[1]

        # Convert one-hot label to discrete labels
        labels = one_hot_label.nonzero().squeeze()

        if labels.numel() == 1:
            grouped_data_by_label[labels.item()].append(item)
        else:
            for label in labels:
                grouped_data_by_label[label.item()].append(item)

    sampled_data = []
    for label, items in grouped_data_by_label.items():
        if len(items) >= num_shots:
            # If data with a specific label is enough to sample
            samples = random.sample(items, num_shots)
        else:
            # Else add all available data
            samples = items
            logger.error(f"The number of label {label} is less than num_shots {num_shots}!")
        sampled_data.extend(samples)

    return sampled_data


class Cifar10Dataset(Dataset):
    DATA = None
    CLASS_NAMES = None
    DATA_DIR = None

    @staticmethod
    def load_data(data_dir):
        splits = ("train", "query", "gallery")
        data = {}
        for split in splits:
            annotations = []
            annotation_file = f"{split}.txt"
            with open(os.path.join(data_dir, annotation_file), "r") as f:
                for line in f:
                    # Line format: lorry_s_001519.png 0 0 0 0 0 0 0 0 0 1
                    # The first item is an image name
                    # Remains are formed as a one-hot label
                    items = line.strip().split(" ")
                    image_filename = items[0]
                    image_path = os.path.join(data_dir, "images", image_filename)
                    label = " ".join(items[1:])
                    onehot_label = torch.from_numpy(np.fromstring(label, dtype=float, sep=' '))  # Strange bug. Cannot use torch.from_numpy
                    annotations.append((image_path, onehot_label))
            data[split] = annotations
        with open(os.path.join(data_dir, "classnames.txt"), "r") as f:
            class_names = f.read().splitlines()

        Cifar10Dataset.DATA = data
        Cifar10Dataset.CLASS_NAMES = class_names
        Cifar10Dataset.DATA_DIR = data_dir

    def __init__(self, split, preprocess, num_shots=None):
        super(Cifar10Dataset, self).__init__()
        assert Cifar10Dataset.DATA or Cifar10Dataset.CLASS_NAMES, "Call load_data first!"
        self.preprocess = preprocess

        # Split few-shot data
        if num_shots:
            assert split == "train", "We only build a few-shot data on the train split"
            cache_pth = os.path.join(Cifar10Dataset.DATA_DIR, f"{num_shots}shot_split_cache.pkl")
            if os.path.exists(cache_pth):
                # If the cache file exists, load it
                with open(cache_pth, 'rb') as f:
                    items = pickle.load(f)
            else:
                # Else split and store it
                items = split_few_shot(Cifar10Dataset.DATA["train"], num_shots)
                with open(cache_pth, 'wb') as f:
                    pickle.dump(items, f)
        else:
            # Return full data
            items = Cifar10Dataset.DATA[split]

        self.items = items
        self.onehot_labels = torch.stack([item[1] for item in self.items])

    def __getitem__(self, index):
        item = self.items[index]
        image_filename, onehot_label = item
        image = self.preprocess(Image.open(image_filename))
        return {
            "images": image,
            "onehot_labels": onehot_label,
            "indices": index,
        }

    def __len__(self):
        return len(self.items)