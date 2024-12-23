import torch.nn as nn

from .similarity_loss import build_similarity_criterion


class HashNetCriterion(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.sim_criterion = build_similarity_criterion(cfg.similarity_criterion)
    
    def forward(self, outputs, batch):
        U_batch = outputs["image_hash_features"]
        dtype = U_batch.dtype
        Y_batch = batch["onehot_labels"].to(dtype)

        sim_loss = self.sim_criterion(U_batch, Y_batch, outputs["current_epoch"])
        loss = sim_loss

        return {
            "loss": loss,
            "sim_loss": sim_loss,
        }