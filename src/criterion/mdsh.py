import torch
import torch.nn as nn

from .quantization_loss import build_quantization_criterion
from .similarity_loss import build_similarity_criterion


class MdshCriterion(nn.Module):
    def __init__(self, cfg, num_train, train_onehot_labels):
        super(MdshCriterion, self).__init__()
        self.bit = cfg.BITS
        self.num_classes = cfg.NUM_CLASSES

        self.sim_criterion = build_similarity_criterion(cfg.similarity_criterion)
        self.qua_criterion = build_quantization_criterion(cfg.quantization_criterion)
        self.lambda_ = cfg.LAMBDA

        self.register_buffer("U", torch.zeros(num_train, self.bit))
        self.register_buffer("Y", train_onehot_labels)
    
    def forward(self, outputs, batch):
        dtype = self.U.dtype
        U_batch = outputs["image_hash_features"].to(dtype)
        image_features = outputs["image_features"].to(dtype)
        Y_batch = batch["onehot_labels"].to(dtype)
        idx = batch["indices"]
        self.Y = self.Y.to(dtype)

        self.U[idx, :] = U_batch.clone().detach()

        sim_loss = self.sim_criterion(U_batch, Y_batch, self.U, self.Y, idx, image_features, outputs["current_epoch"])
        qua_loss = self.qua_criterion(U_batch)
        loss = sim_loss + self.lambda_ * qua_loss
        return {
            "loss": loss,
            "sim_loss": sim_loss,
            "qua_loss": qua_loss,
        }
