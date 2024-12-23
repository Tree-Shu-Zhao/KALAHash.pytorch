from loguru import logger
from torch.utils.data import DataLoader

from clip import clip

from .cifar10 import Cifar10Dataset
from .mscoco import CocoDataset
from .nus_wide import NusWideDataset


def build_dataloaders(cfg):
    name = cfg.NAME.lower()
    _, preprocess = clip.load(cfg.CLIP_BACKBONE)
    if name == "cifar-10" or name == "cifar10":
        Cifar10Dataset.load_data(cfg.DATA_DIR)
        train_dataset = Cifar10Dataset("train", preprocess, num_shots=cfg.NUM_SHOTS)
        query_dataset = Cifar10Dataset("query", preprocess)
        gallery_dataset = Cifar10Dataset("gallery", preprocess)
    elif name == "nus-wide":
        NusWideDataset.load_data(cfg.DATA_DIR)
        train_dataset = NusWideDataset("train", preprocess, num_shots=cfg.NUM_SHOTS)
        query_dataset = NusWideDataset("query", preprocess)
        gallery_dataset = NusWideDataset("gallery", preprocess)
    elif name == "mscoco" or name == "coco" or name == "ms-coco":
        CocoDataset.load_data(cfg.DATA_DIR)
        train_dataset = CocoDataset("train", preprocess, num_shots=cfg.NUM_SHOTS)
        query_dataset = CocoDataset("query", preprocess)
        gallery_dataset = CocoDataset("gallery", preprocess)
    else:
        raise ValueError(f"Unknown dataset name: {name}")

    logger.info(f"num_train: {len(train_dataset)}")
    logger.info(f"num_query: {len(query_dataset)}")
    logger.info(f"num_gallery: {len(gallery_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN_BATCH_SIZE,
        num_workers=cfg.NUM_WORKERS,
        shuffle=True,
        pin_memory=True,
        drop_last=True, # When batch size is 1, BN will raise an error
    )
    query_loader = DataLoader(
        query_dataset,
        batch_size=cfg.TEST_BATCH_SIZE,
        num_workers=cfg.NUM_WORKERS,
        shuffle=False,
        pin_memory=True,
    )
    gallery_loader = DataLoader(
        gallery_dataset,
        batch_size=cfg.TEST_BATCH_SIZE,
        num_workers=cfg.NUM_WORKERS,
        shuffle=False,
        pin_memory=True,
    )

    return {
        "train": train_loader,
        "query": query_loader,
        "gallery": gallery_loader,
    }