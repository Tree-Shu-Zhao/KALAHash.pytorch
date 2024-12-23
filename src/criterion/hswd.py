import torch.nn as nn

from .quantization_loss import build_quantization_criterion
from .similarity_loss import build_similarity_criterion


class HSWDCriterion(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.lambda_ = cfg.LAMBDA

        self.sim_criterion = build_similarity_criterion(cfg.similarity_criterion)
        self.qua_criterion = build_quantization_criterion(cfg.quantization_criterion)
    
    def forward(self, outputs, batch):
        U_batch = outputs["image_hash_features"]
        dtype = U_batch.dtype
        Y_batch = batch["onehot_labels"].to(dtype)

        sim_loss = self.sim_criterion(U_batch, Y_batch, outputs["current_epoch"])
        qua_loss = self.qua_criterion(U_batch, None)
        loss = sim_loss + self.lambda_ * qua_loss

        return {
            "loss": loss,
            "sim_loss": sim_loss,
            "qua_loss": qua_loss,
        }