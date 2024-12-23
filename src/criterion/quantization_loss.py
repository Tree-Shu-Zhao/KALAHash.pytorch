import torch
import torch.nn as nn


def build_quantization_criterion(cfg):
    name = cfg.NAME.lower()
    if name == "l2":
        return L2QuantizationLoss(cfg)
    elif name == "l1":
        return L1QuantizationLoss(cfg)
    elif name == "cauchy":
        return CauchyQuantizationLoss(cfg)
    elif name == "greedy":
        return GreedyHashQuantizationLoss(cfg)
    elif name == "swd":
        return SwdQuantizationLoss(cfg)
    elif name == "hswd":
        return HswdQuantizationLoss(cfg)
    else:
        raise ValueError(f"Quantization loss {name} not found")

class L2QuantizationLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
    
    def forward(self, U, B):
        return (U - B.sign()).pow(2).mean()

class L1QuantizationLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
    
    def forward(self, U):
        return (U.abs() - 1).pow(2).mean()

class CauchyQuantizationLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.gamma = cfg.GAMMA
        self.K = cfg.BITS
    
    def d(self, hi, hj):
        inner_product = hi @ hj.t()
        norm = hi.pow(2).sum(dim=1, keepdim=True).pow(0.5) @ hj.pow(2).sum(dim=1, keepdim=True).pow(0.5).t()
        cos = inner_product / norm.clamp(min=0.0001)
        # formula 6
        return (1 - cos.clamp(max=0.99)) * self.K / 2
    
    def forward(self, U):
        ones = torch.ones_like(U)
        return torch.log(1 + self.d(U.abs(), ones) / self.gamma).mean()

class GreedyHashQuantizationLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.alpha = cfg.ALPHA
    
    def forward(self, U):
        return self.alpha * (U.abs() - 1).pow(3).abs().mean()

class SwdQuantizationLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_projections = cfg.NUM_PROJECTIONS
    
    def forward(self, U, _, aggregate=True):
        device = U.device
        U = U.float()
        dim = U.size(1)
        real_b = torch.randn(U.shape, device=device).sign()
        
        theta = torch.randn((dim, self.n_projections), requires_grad=False, device=device)
        theta = theta/torch.norm(theta, dim=0)[None, :]
        
        xgen_1d = U.view(-1, dim)@theta
        xreal_1d = real_b.view(-1, dim)@theta
        
        if aggregate:
            gloss = wasserstein1d(xreal_1d, xgen_1d) / self.n_projections
        else:
            gloss = wasserstein1d(xreal_1d, xgen_1d, aggregate=False)
        
        return gloss

class HswdQuantizationLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
    
    def forward(self, U, _, aggregate=True):
        device = U.device
        real_b = torch.randn(U.shape, device=device).sign()
        bsize, dim = U.size()

        if aggregate:
            gloss = wasserstein1d(real_b, U) / dim
        else:
            gloss = wasserstein1d(real_b, U, aggregate=False)

        return gloss


def wasserstein1d(x, y, aggregate=True):
    """Compute wasserstein loss in 1D"""
    x1, _ = torch.sort(x, dim=0)
    y1, _ = torch.sort(y, dim=0)
    n = x.size(0)
    if aggregate:
        z = (x1-y1).view(-1)
        return torch.dot(z, z)/n
    else:
        return (x1-y1).square().sum(0)/n
