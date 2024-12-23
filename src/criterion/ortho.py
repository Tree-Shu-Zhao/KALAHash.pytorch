import torch.nn as nn

from .quantization_loss import build_quantization_criterion
from .similarity_loss import build_similarity_criterion


class OrthoHashCriterion(nn.Module):
    def __init__(self, cfg):
        super(OrthoHashCriterion, self).__init__()

        self.sim_criterion = build_similarity_criterion(cfg.similarity_criterion)
        self.qua_criterion = build_quantization_criterion(cfg.quantization_criterion)
        self.quan = cfg.QUAN
    
    def forward(self, outputs, batch):
        U_batch = outputs["image_hash_features"]
        dtype = U_batch.dtype
        Y_batch = batch["onehot_labels"].to(dtype)

        sim_loss = self.sim_criterion(U_batch, Y_batch)
        qua_loss = self.qua_criterion(U_batch, U_batch)
        loss = sim_loss + self.quan * qua_loss
        return {
            "loss": loss,
            "sim_loss": sim_loss,
            "qua_loss": qua_loss,
        }
