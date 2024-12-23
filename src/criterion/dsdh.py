import copy

import torch
import torch.nn as nn

from .quantization_loss import build_quantization_criterion
from .similarity_loss import build_similarity_criterion


class DsdhCriterion(nn.Module):
    def __init__(self, cfg, num_train, train_onehot_labels):
        super(DsdhCriterion, self).__init__()

        self.bit = cfg.BITS
        self.mu = cfg.DSDH_MU
        self.nu = cfg.DSDH_NU
        self.eta = cfg.DSDH_ETA
        self.dcc_iter = cfg.DCC_ITER
        self.W = None

        self.sim_criterion = build_similarity_criterion(cfg.similarity_criterion)
        self.qua_criterion = build_quantization_criterion(cfg.quantization_criterion)
        self.cls_criterion = copy.deepcopy(self.qua_criterion) # Use L2 regression loss for classification

        self.register_buffer("B", torch.zeros(self.bit, num_train).float())
        self.dtype = self.B.dtype
        self.register_buffer("U", torch.zeros(self.bit, num_train, dtype=self.dtype))
        self.register_buffer("Y", train_onehot_labels.t().to(self.dtype))

    def forward(self, outputs, batch):
        dtype = self.dtype
        if self.W is None:
            self.DCC()
        
        U_batch = outputs["image_hash_features"].to(dtype)
        Y_batch = batch["onehot_labels"].to(dtype)
        idx = batch["indices"]

        self.U[:, idx] = U_batch.clone().detach().t()
        self.Y[:, idx] = Y_batch.t()

        sim_loss = self.sim_criterion(U_batch, Y_batch)
        cls_loss = self.cls_criterion(Y_batch.t(), self.W.t() @ self.B[:, idx])
        qua_loss = self.qua_criterion(U_batch, self.B[:, idx].t())
        loss = sim_loss + self.mu * cls_loss + self.eta * qua_loss

        return {
            "loss": loss,
            "sim_loss": sim_loss,
            "cls_loss": cls_loss,
            "qua_loss": qua_loss,
        }

    def DCC(self):
        with torch.no_grad():
            B = self.B
            for _ in range(self.dcc_iter):
                # W-step
                W = torch.inverse(B @ B.t() + self.nu / self.mu * torch.eye(self.bit).to(B.device)) @ B @ self.Y.t()

                for i in range(B.shape[0]):
                    P = W @ self.Y + self.eta / self.mu * self.U
                    p = P[i, :]
                    w = W[i, :]
                    W_prime = torch.cat((W[:i, :], W[i + 1 :, :]))
                    B_prime = torch.cat((B[:i, :], B[i + 1 :, :]))
                    B[i, :] = (p - B_prime.t() @ W_prime @ w).sign()

            self.B = B
            self.W = W