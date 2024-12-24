import torch
import torch.nn as nn


class KiddoCriterion(nn.Module):
    def __init__(self, cfg, num_train, train_onehot_labels):
        super().__init__()
        self.cfg = cfg

        # Build similarity and quantization criterion
        self.sim_criterion = SimLoss()
        self.qua_criterion = QuantizationLoss()

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
        ali_loss = (onehot_labels - logits).pow(2).mean()

        loss = (
            self.cfg.GAMMA * sim_loss
            + self.alpha * ali_loss
            + self.beta * qua_loss
        )

        return {
            "loss": loss,
            "sim_loss": sim_loss,
            "ali_loss": ali_loss,
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

class SimLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, feats, labels, current_epoch=None):
        theta = feats @ feats.t() / 2
        sim_matrix = (labels @ labels.t() > 0).float()
        sim_loss = ((1 + (-theta.abs()).exp()).log() + theta.clamp(min=0) - sim_matrix * theta).mean()

        return sim_loss

class QuantizationLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, U, B):
        return (U - B.sign()).pow(2).mean()
