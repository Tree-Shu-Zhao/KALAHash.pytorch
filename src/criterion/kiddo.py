import torch
import torch.nn as nn

from .quantization_loss import build_quantization_criterion
from .similarity_loss import build_similarity_criterion


class KiddoCriterion(nn.Module):
    def __init__(self, cfg, num_train, train_onehot_labels):
        super().__init__()
        self.cfg = cfg

        # Build similarity and quantization criterion
        self.sim_criterion_name = cfg.similarity_criterion.NAME.lower()
        self.sim_criterion = build_similarity_criterion(cfg.similarity_criterion)
        self.qua_criterion = build_quantization_criterion(cfg.quantization_criterion)

        self.alpha = cfg.ALPHA
        self.beta = cfg.BETA
        self.num_dcc = cfg.NUM_DCC
        self.bits = cfg.BITS

        # Initialize variables
        self.register_buffer("B", torch.randn(cfg.BITS, num_train).sign().float())
        self.dtype = self.B.dtype
        self.register_buffer("U", torch.zeros(cfg.BITS, num_train, dtype=self.dtype))
        self.register_buffer("Y", train_onehot_labels.t().to(self.dtype))

    def forward(self, outputs, batch_data):
        dtype = self.dtype
        image_hash_features = outputs["image_hash_features"].to(dtype)
        textual_knowledge = outputs["textual_knowledge"].to(dtype)
        onehot_labels = batch_data["onehot_labels"].to(dtype)
        idx = batch_data["indices"]

        self.W = textual_knowledge.t()
        self.U[:, idx] = image_hash_features.clone().detach().t()

        sim_loss = self.sim_criterion(image_hash_features, onehot_labels, outputs["current_epoch"])
        qua_loss = self.qua_criterion(image_hash_features, self.B[:, idx].t())
        logits = self.B[:, idx].t() @ self.W
        W_cls_loss = (onehot_labels - logits).pow(2).mean()

        loss = (
            self.cfg.GAMMA * sim_loss
            + self.alpha * W_cls_loss
            + self.beta * qua_loss
        )

        return {
            "loss": loss,
            "sim_loss": sim_loss,
            "cls_loss": W_cls_loss,
            "qua_loss": qua_loss,
        }

    def DCC(self):
        with torch.no_grad():
            B = self.B
            W = self.W
            for _ in range(self.num_dcc):
                # W-step

                for i in range(B.shape[0]):
                    P = W @ self.Y + self.beta / self.alpha * self.U
                    p = P[i, :]
                    w = W[i, :]
                    W_prime = torch.cat((W[:i, :], W[i + 1 :, :]))
                    B_prime = torch.cat((B[:i, :], B[i + 1 :, :]))
                    B[i, :] = (p - B_prime.t() @ W_prime @ w).sign()

            self.B = B
