import os
import pickle
import random
from collections import defaultdict

import numpy as np
import torch
from loguru import logger
from PIL import Image
from torch.utils.data import Dataset


def split_few_shot_multi_label(train_items, num_shots, num_classes):
    assert num_shots > 0 and isinstance(num_shots, int), "num_shots must be an integer and greater than 0"
    
    # Initialize dictionaries to keep track of samples per label
    label_sample_count = {label: 0 for label in range(num_classes)}
    sampled_data = []
    
    for item in train_items:
        one_hot_label = item[1]
        labels = one_hot_label.nonzero().squeeze()
        
        # Handle both single-label and multi-label cases
        if labels.dim() == 0:
            labels = [labels.item()]
        else:
            labels = labels.tolist()
        
        # Check if this item can be added without exceeding num_shots for any label
        can_add = True
        for label in labels:
            if label_sample_count[label] >= num_shots:
                can_add = False
                break
        
        if can_add:
            for label in labels:
                label_sample_count[label] += 1
            sampled_data.append(item)
        
        # Check if we have exactly num_shots samples for all labels
        if all(count == num_shots for count in label_sample_count.values()):
            break
    
    # Check if any label doesn't have enough samples
    for label, count in label_sample_count.items():
        if count < num_shots:
            logger.error(f"Unable to find {num_shots} samples for label {label}. Only found {count}.")
    
    return sampled_data

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


class CocoDataset(Dataset):
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

        CocoDataset.DATA = data
        CocoDataset.CLASS_NAMES = class_names
        CocoDataset.DATA_DIR = data_dir

    def __init__(self, split, preprocess, num_shots=None):
        super(CocoDataset, self).__init__()
        assert CocoDataset.DATA or CocoDataset.CLASS_NAMES, "Call load_data first!"
        self.preprocess = preprocess

        # Split few-shot data
        if num_shots:
            assert split == "train", "We only build a few-shot data on the train split"
            cache_pth = os.path.join(CocoDataset.DATA_DIR, f"{num_shots}shot_split_cache.pkl")
            if os.path.exists(cache_pth):
                # If the cache file exists, load it
                with open(cache_pth, 'rb') as f:
                    items = pickle.load(f)
            else:
                # Else split and store it
                items = split_few_shot_multi_label(CocoDataset.DATA["train"], num_shots, 80)
                with open(cache_pth, 'wb') as f:
                    pickle.dump(items, f)
        else:
            # Return full data
            items = CocoDataset.DATA[split]

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