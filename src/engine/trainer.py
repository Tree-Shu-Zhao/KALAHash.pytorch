import os
import time

import torch
from loguru import logger

from src.criterion import build_criterion
from src.utils import AverageMeter, DictAverageMeter, ProgressMeter

from .evaluator import HashingEvaluator


class Trainer:
    def __init__(self, cfg, dataloaders, model):
        super().__init__()

        self.cfg = cfg
        self.model = model
        self.train_dataloader = dataloaders["train"]
        self.query_dataloader = dataloaders["query"]
        self.gallery_dataloader = dataloaders["gallery"]

        self.current_epoch = 0
        self.current_iter = 0
        self.best_map = 0.
        self.epochs = cfg.train.EPOCHS

        # Setup devices
        if cfg.train.GPU is None:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(f"cuda:{cfg.train.GPU[0]}") if isinstance(cfg.train.GPU, list) else torch.device(f"cuda:{cfg.train.GPU}")
        self.model.to(self.device)

        # Build criterion
        self.criterion = build_criterion(
            cfg.criterion, 
            num_train=len(self.train_dataloader.dataset),
            train_onehot_labels = self.train_dataloader.dataset.onehot_labels
            ).to(self.device)

        # Build optimizer
        # Other optimizers will lead to NAN
        self.optimizer = torch.optim.SGD(
            self.model.get_learnable_params() + list(self.criterion.parameters()),
            lr=cfg.train.optimizer.LR,
            momentum=cfg.train.optimizer.MOMENTUM,
            weight_decay=cfg.train.optimizer.WEIGHT_DECAY,
        )

        # Build evaluator
        self.evaluator = HashingEvaluator(cfg.model.hash_head.BITS, cfg.test.TOPK, self.device)

    def train(self):
        for epoch in range(self.epochs):
            self.current_epoch = epoch
            self.model.set_model_mode("train")

            # Training loop
            self.train_loop()

            # Evaluate
            if (epoch + 1) % self.cfg.train.EVAL_FREQ == 0:
                result = self.evaluate()
                mAP = result["mAP"]
                self.best_map = max(self.best_map, mAP)
                logger.info(
                    "epoch: {}\t mAP: {:.4f} best_map: {:.4f}".format(epoch, mAP, self.best_map)
                )

        # Save checkpoints
        torch.save(self.model.state_dict(), os.path.join(self.cfg.train.CHECKPOINT_DIR, "model.pth"))
        
    def evaluate(self, checkpoint_path=None):
        self.model.set_model_mode("eval")
        if checkpoint_path is not None:
            self.model.load_state_dict(torch.load(checkpoint_path))
            self.model.to(self.device)
        result = self.evaluator.evaluate(self.model, self.query_dataloader, self.gallery_dataloader)
        return result
    
    def train_loop(self):
        # Build meters
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        loss_meter = DictAverageMeter(':6.3f')
        progress = ProgressMeter(
            len(self.train_dataloader),
            [batch_time, data_time, loss_meter],
            prefix="Epoch: [{}/{}]".format(self.current_epoch+1, self.epochs),
        )

        end = time.perf_counter()
        for it, batch in enumerate(self.train_dataloader):
            self.current_iter += 1
            data_time.update(time.perf_counter() - end)

            # Forward pass and backward pass
            self.optimizer.zero_grad()
            batch = move_to_device(batch, self.device)
            outputs = self.model(batch)
            outputs.update({"current_epoch": self.current_epoch})
            loss_dict = self.criterion(outputs, batch)
            loss_dict["loss"].backward()
            self.optimizer.step()

            # Update loss meters
            loss_meter.update(loss_dict)

            batch_time.update(time.perf_counter() - end)
            end = time.perf_counter()

            if (it+1) % self.cfg.train.PRINT_FREQ == 0:
                progress.display(it+1)
        
        if hasattr(self.criterion, "DCC"):
            self.criterion.DCC()

def move_to_device(batch, device):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = batch[k].to(device)
    return batch
