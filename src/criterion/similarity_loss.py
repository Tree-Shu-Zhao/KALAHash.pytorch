import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.linalg import hadamard
from scipy.special import comb


def build_similarity_criterion(cfg):
    name = cfg.NAME.lower()
    if name == "dpsh":
        return DpshLoss(cfg)
    elif name == "dch":
        return DchLoss(cfg)
    elif name == "hashnet":
        return HashNetLoss(cfg)
    elif name == "greedy":
        return GreedyHashLoss(cfg)
    elif name == "csq":
        return CsqLoss(cfg)
    elif name == "ortho":
        return OrthoHashLoss(cfg)
    elif name == "mdsh":
        return MdshLoss(cfg)
    else:
        raise ValueError(f"Similarity loss {name} not found")


class DpshLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
    
    def forward(self, feats, labels, current_epoch=None):
        theta = feats @ feats.t() / 2
        sim_matrix = (labels @ labels.t() > 0).float()
        sim_loss = ((1 + (-theta.abs()).exp()).log() + theta.clamp(min=0) - sim_matrix * theta).mean()

        return sim_loss

class DchLoss(torch.nn.Module):
    def __init__(self, cfg):
        super(DchLoss, self).__init__()
        self.gamma = cfg.GAMMA
        self.K = cfg.BITS

    def d(self, hi, hj):
        inner_product = hi @ hj.t()
        norm = hi.pow(2).sum(dim=1, keepdim=True).pow(0.5) @ hj.pow(2).sum(dim=1, keepdim=True).pow(0.5).t()
        cos = inner_product / norm.clamp(min=0.0001)
        # formula 6
        return (1 - cos.clamp(max=0.99)) * self.K / 2

    def forward(self, U, Y, current_epoch=None):
        s = (Y @ Y.t() > 0).float()

        if (1 - s).sum() != 0 and s.sum() != 0:
            # formula 2
            positive_w = s * s.numel() / s.sum()
            negative_w = (1 - s) * s.numel() / (1 - s).sum()
            w = positive_w + negative_w
        else:
            # maybe |S1|==0 or |S2|==0
            w = 1

        d_hi_hj = self.d(U, U)

        # formula 8
        cauchy_loss = w * (s * torch.log(d_hi_hj / self.gamma) + torch.log(1 + self.gamma / d_hi_hj))
        cauchy_loss = cauchy_loss.mean()

        return cauchy_loss

class HashNetLoss(torch.nn.Module):
    def __init__(self, cfg):
        super(HashNetLoss, self).__init__()

        self.alpha = cfg.HASHNET_ALPHA
        self.step = cfg.HASHNET_STEP

    def forward(self, U, Y, current_epoch):
        scale = (current_epoch // self.step + 1) ** 0.5
        U = torch.tanh(scale * U)

        similarity = (Y @ Y.t() > 0).float()
        dot_product = self.alpha * U @ U.t()

        mask_positive = similarity.data > 0
        mask_negative = similarity.data <= 0

        exp_loss = (1 + (-dot_product.abs()).exp()).log() + dot_product.clamp(min=0) - similarity * dot_product

        # weight
        S1 = mask_positive.float().sum()
        S0 = mask_negative.float().sum()
        S = S0 + S1
        exp_loss[mask_positive] = exp_loss[mask_positive] * (S / S1)
        exp_loss[mask_negative] = exp_loss[mask_negative] * (S / S0)

        loss = exp_loss.sum() / S

        return loss

class GreedyHashLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        bit = cfg.BITS
        n_class = cfg.NUM_CLASSES
        self.fc = nn.Linear(bit, n_class, bias=False)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, u, onehot_y, current_epoch=None):
        b = GreedyHashLoss.Hash.apply(u)
        # one-hot to label
        y = onehot_y.argmax(axis=1)
        b = b.to(self.fc.weight.dtype)
        y_pre = self.fc(b)
        loss = self.criterion(y_pre, y)
        return loss
    
    class Hash(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            # ctx.save_for_backward(input)
            return input.sign()

        @staticmethod
        def backward(ctx, grad_output):
            # input,  = ctx.saved_tensors
            # grad_output = grad_output.data
            return grad_output

class CsqLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        dataset_name = cfg.DATASET_NAME.lower()
        if dataset_name == "nus-wide" or dataset_name == "coco" or dataset_name == "ms-coco" or dataset_name == "mscoco":
            self.is_single_label = False
        elif dataset_name == "cifar10" or dataset_name == "cifar-10":
            self.is_single_label = True
        else:
            raise ValueError(f"Cannot determine the type of the dataset: {dataset_name}")

        self.register_buffer("hash_targets", self.get_hash_targets(cfg.NUM_CLASSES, cfg.BITS))
        self.register_buffer("multi_label_random_center", torch.randint(2, (cfg.BITS,)))
        self.criterion = torch.nn.BCELoss()

    def forward(self, u, y, current_epoch=None):
        hash_center = self.label2center(y)
        u = u.to(hash_center.dtype)
        center_loss = self.criterion(0.5 * (u + 1), 0.5 * (hash_center + 1))
        return center_loss

    def label2center(self, y):
        self.hash_targets = self.hash_targets.to(y.dtype)
        self.multi_label_random_center = self.multi_label_random_center.to(y.dtype)
        if self.is_single_label:
            hash_center = self.hash_targets[y.argmax(axis=1)]
        else:
            # to get sign no need to use mean, use sum here
            center_sum = y @ self.hash_targets
            random_center = self.multi_label_random_center.repeat(center_sum.shape[0], 1)
            center_sum[center_sum == 0] = random_center[center_sum == 0]
            hash_center = 2 * (center_sum > 0).float() - 1
        return hash_center

    # use algorithm 1 to generate hash centers
    def get_hash_targets(self, n_class, bit):
        H_K = hadamard(bit)
        H_2K = np.concatenate((H_K, -H_K), 0)
        hash_targets = torch.from_numpy(H_2K[:n_class]).float()

        if H_2K.shape[0] < n_class:
            hash_targets.resize_(n_class, bit)
            for k in range(20):
                for index in range(H_2K.shape[0], n_class):
                    ones = torch.ones(bit)
                    # Bernouli distribution
                    sa = random.sample(list(range(bit)), bit // 2)
                    ones[sa] = -1
                    hash_targets[index] = ones
                # to find average/min  pairwise distance
                c = []
                for i in range(n_class):
                    for j in range(n_class):
                        if i < j:
                            TF = sum(hash_targets[i] != hash_targets[j])
                            c.append(TF)
                c = np.array(c)

                # choose min(c) in the range of K/4 to K/3
                # see in https://github.com/yuanli2333/Hadamard-Matrix-for-hashing/issues/1
                # but it is hard when bit is  small
                if c.min() > bit / 4 and c.mean() >= bit / 2:
                    print(c.min(), c.mean())
                    break
        return hash_targets

class OrthoHashLoss(nn.Module):
    def __init__(self, cfg):
        super(OrthoHashLoss, self).__init__()
        self.losses = {}
        self.s = cfg.SCALE
        self.m = cfg.MARGIN
        self.m_type = cfg.MARGIN_TYPE
        self.bit = cfg.BITS
        self.n_class = cfg.NUM_CLASSES

        dataset_name = cfg.DATASET_NAME.lower()
        if dataset_name == "nus-wide" or dataset_name == "coco" or dataset_name == "ms-coco" or dataset_name == "mscoco":
            self.multiclass = True
        elif dataset_name == "cifar10" or dataset_name == "cifar-10":
            self.multiclass = False
        else:
            raise ValueError(f"Cannot determine the type of the dataset: {dataset_name}")

        self.multiclass_loss = cfg.MULTICLASS_LOSS
        
        if cfg.CODEBOOK == 'normal':
            codebook = torch.randn(cfg.NUM_CLASSES, cfg.BITS)
        elif cfg.CODEBOOK == 'bernoulli':
            prob = torch.ones(cfg.NUM_CLASSES, cfg.BITS) * 0.5
            codebook = torch.bernoulli(prob) * 2. - 1.
        elif cfg.CODEBOOK == "mdsh":
            center_path = os.path.join(cfg.CENTER_PATH, f"CSQ_init_True_{self.n_class}_{self.bit}.npy")
            hash_centers = np.load(center_path)
            codebook = torch.from_numpy(hash_centers).float()
        else:  # O: optim
            codebook = self.get_codebook(cfg.NUM_CLASSES, cfg.BITS)

        codebook = codebook.sign().cuda()

        if codebook is None:  # usual CE
            self.ce_fc = nn.Linear(cfg.BITS, cfg.NUM_CLASSES)
        else:
            # not learning cent, we are doing codebook learning
            self.ce_fc = CosSim(
                cfg.BITS,
                cfg.NUM_CLASSES,
                codebook,
                learn_cent=False,
            )
        self.ce_fc = self.ce_fc.cuda()
    
    def compute_margin_logits(self, logits, labels):
        if self.m_type == 'cos':
            if self.multiclass:
                y_onehot = labels * self.m
                margin_logits = self.s * (logits - y_onehot)
            else:
                y_onehot = torch.zeros_like(logits)
                y_onehot.scatter_(1, torch.unsqueeze(labels, dim=-1), self.m)
                margin_logits = self.s * (logits - y_onehot)
        else:
            if self.multiclass:
                y_onehot = labels * self.m
                arc_logits = torch.acos(logits.clamp(-0.99999, 0.99999))
                logits = torch.cos(arc_logits + y_onehot)
                margin_logits = self.s * logits
            else:
                y_onehot = torch.zeros_like(logits)
                y_onehot.scatter_(1, torch.unsqueeze(labels, dim=-1), self.m)
                arc_logits = torch.acos(logits.clamp(-0.99999, 0.99999))
                logits = torch.cos(arc_logits + y_onehot)
                margin_logits = self.s * logits

        return margin_logits

    def forward(self, hash_features, labels, current_epoch=None):
        logits = self.ce_fc(hash_features.float())

        if self.multiclass:
            labels = labels.float()

            margin_logits = self.compute_margin_logits(logits, labels)

            if self.multiclass_loss in ['bce', 'imbalance']:
                loss_ce = F.binary_cross_entropy_with_logits(
                    margin_logits, labels, reduction='none')
                if self.multiclass_loss == 'imbalance':
                    imbalance_mask, multiclass_acc = self.get_imbalance_mask(
                        torch.sigmoid(margin_logits), labels, labels.size(1))
                    loss_ce = loss_ce * imbalance_mask
                    loss_ce = loss_ce.sum() / (imbalance_mask.sum() + 1e-7)
                    self.losses['multiclass_acc'] = multiclass_acc
                else:
                    loss_ce = loss_ce.mean()
            elif self.multiclass_loss in ['label_smoothing']:
                log_logits = F.log_softmax(margin_logits, dim=1)
                labels_scaled = labels / labels.sum(dim=1, keepdim=True)
                loss_ce = - (labels_scaled * log_logits).sum(dim=1)
                loss_ce = loss_ce.mean()
            else:
                raise NotImplementedError(
                    f'unknown method: {self.multiclass_loss}')
        else:
            labels = labels.argmax(1)

            margin_logits = self.compute_margin_logits(logits, labels)
            loss_ce = F.cross_entropy(margin_logits, labels)
        return loss_ce

    def get_imbalance_mask(self, sigmoid_logits, labels, nclass,
                       threshold=0.7, imbalance_scale=-1):
        if imbalance_scale == -1:
            imbalance_scale = 1 / nclass

        mask = torch.ones_like(sigmoid_logits) * imbalance_scale

        # wan to activate the output
        mask[labels == 1] = 1

        # if predicted wrong, and not the same as labels, minimize it
        correct = (sigmoid_logits >= threshold) == (labels == 1)
        mask[~correct] = 1

        multiclass_acc = correct.float().mean()

        # the rest maintain "imbalance_scale"
        return mask, multiclass_acc
    
    def get_codebook(self,
        nclass, nbit, maxtries=10000, initdist=0.61, mindist=0.2,
        reducedist=0.01):
        """
        brute force to find centroid with furthest distance
        :param nclass:
        :param nbit:
        :param maxtries:
        :param initdist:
        :param mindist:
        :param reducedist:
        :return:
        """
        codebook = torch.zeros(nclass, nbit)
        i = 0
        count = 0
        currdist = initdist
        while i < nclass:
            print(i, end='\r')
            c = torch.randn(nbit).sign()
            nobreak = True
            for j in range(i):
                if self.get_hd(c, codebook[j]) < currdist:
                    i -= 1
                    nobreak = False
                    break
            if nobreak:
                codebook[i] = c
            else:
                count += 1

            if count >= maxtries:
                count = 0
                currdist -= reducedist
                print('reduce', currdist, i)
                if currdist < mindist:
                    raise ValueError('cannot find')

            i += 1
        codebook = codebook[torch.randperm(nclass)]
        return codebook
    
    def get_hd(self, a, b):
        return 0.5 * (a.size(0) - a @ b.t()) / a.size(0)

class CosSim(nn.Module):
    def __init__(self, nfeat, nclass, codebook=None, learn_cent=True):
        super(CosSim, self).__init__()
        self.nfeat = nfeat
        self.nclass = nclass
        self.learn_cent = learn_cent

        if codebook is None:  # if no centroids, by default just usual weight
            codebook = torch.randn(nclass, nfeat)

        self.centroids = nn.Parameter(codebook.clone()).cuda()
        if not learn_cent:
            self.centroids.requires_grad_(False)

    def forward(self, x):
        norms = torch.norm(x, p=2, dim=-1, keepdim=True)
        nfeat = torch.div(x, norms)

        norms_c = torch.norm(self.centroids, p=2, dim=-1, keepdim=True)
        ncenters = torch.div(self.centroids, norms_c)
        logits = torch.matmul(nfeat, torch.transpose(ncenters, 0, 1))

        return logits

    def extra_repr(self) -> str:
        return 'in_features={}, n_class={}, learn_centroid={}'.format(
            self.nfeat, self.nclass, self.learn_cent
        )

class MdshLoss(nn.Module):
    def __init__(self, cfg):
        """
        :param config: in paper, the hyper-parameter lambda is chose 0.0001
        :param bit:
        """
        super(MdshLoss, self).__init__()
        self.bit = cfg.BITS
        self.n_class = cfg.NUM_CLASSES
        self.epoch_change = cfg.EPOCH_CHANGE
        self.beta = cfg.BETA
        self.center_path = os.path.join(cfg.CENTER_PATH, f"CSQ_init_True_{self.n_class}_{self.bit}.npy")

        self.alpha_pos, self.alpha_neg, self.beta_neg, self.d_min, self.d_max = self.get_margin()
        self.hash_center = self.generate_center(self.bit, self.n_class, list(range(self.n_class)))
        # np.save(config['save_center'], self.hash_center.cpu().numpy())
        self.BCEloss = nn.BCELoss()
        
        self.register_buffer("label_center", torch.from_numpy(
            np.eye(self.n_class, dtype=np.float32)[np.array([i for i in range(self.n_class)])]))
        self.tanh = nn.Tanh()

    def forward(self, hash_features, labels, U, Y, idx, image_features, current_epoch):
        if current_epoch < self.epoch_change:
            pair_loss = 0
        else:
            last_u = U
            last_y = Y
            pair_loss = self.moco_pairloss(hash_features, labels, last_u, last_y, idx)
        cos_loss = self.cos_eps_loss(hash_features, labels, idx)
        
        loss = cos_loss + self.beta * pair_loss
        return loss

    def moco_pairloss(self, u, y, last_u, last_y, ind):
        u = F.normalize(u)
        last_u = F.normalize(last_u)
        label_sim = ((y @ y.t()) > 0).float()
        cos_sim = u @ u.t()
        last_sim = ((y @ last_y.t()) > 0).float()
        last_cos = u @ last_u.t()

        loss = torch.sum(last_sim * torch.log(1 + torch.exp(1/2 *(1 - last_cos))))/torch.sum(last_sim) # only the positive pair 
        return loss
    
    def cos_eps_loss(self, u, y, ind):
        K = self.bit
        m = 0.0
        l = 1 - 2 * self.d_max / K
        u_norm = F.normalize(u)
        centers_norm = F.normalize(self.hash_center)
        cos_sim = torch.matmul(u_norm, torch.transpose(centers_norm, 0, 1)) # batch x n_class
        s = (y @ self.label_center.t()).float() # batch x n_class
        cos_sim = K ** 0.5 * cos_sim
        p = torch.softmax(cos_sim, dim=1)
        loss = s * torch.log(p) + (1-s) * torch.log(1-p)
        loss = torch.mean(loss)
        return -loss
            
    def get_margin(self):
        L = self.bit
        n_class = self.n_class
        right = (2 ** L) / n_class
        d_min = 0
        d_max = 0
        for j in range(2 * L + 4):
            dim = j
            sum_1 = 0
            sum_2 = 0
            for i in range((dim - 1) // 2 + 1):
                sum_1 += comb(L, i)
            for i in range((dim) // 2 + 1):
                sum_2 += comb(L, i)
            if sum_1 <= right and sum_2 > right:
                d_min = dim
        for i in range(2 * L + 4):
            dim = i
            sum_1 = 0
            sum_2 = 0
            for j in range(dim):
                sum_1 += comb(L, j)
            for j in range(dim - 1):
                sum_2 += comb(L, j)
            if sum_1 >= right and sum_2 < right:
                d_max = dim
        alpha_neg = L - 2 * d_max
        beta_neg = L - 2 * d_min
        alpha_pos = L
        return alpha_pos, alpha_neg, beta_neg, d_min, d_max

    def generate_center(self, bit, n_class, l):
        hash_centers = np.load(self.center_path)
        self.evaluate_centers(hash_centers)
        hash_centers = hash_centers[l]
        Z = torch.from_numpy(hash_centers).float().cuda()
        return Z
    
    def evaluate_centers(self, H):
        dist = []
        for i in range(H.shape[0]):
            for j in range(i+1, H.shape[0]):
                    TF = np.sum(H[i] != H[j])
                    dist.append(TF)
        dist = np.array(dist)
        st = dist.mean() - dist.var() + dist.min()
        print(f"mean is {dist.mean()}; min is {dist.min()}; var is {dist.var()}; max is {dist.max()}")


class ADSHSimLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.bits = cfg.BITS
    
    def forward(self, feats, labels):
        S = (labels @ labels.t() > 0).float()
        S = torch.where(S == 1, torch.full_like(S, 1), torch.full_like(S, -1))
        # Soft similarity matrix, benefit to converge
        r = S.sum() / (1 - S).sum()
        S = S * (1 + r) - r
        sim_loss = ((self.bits * S - feats @ feats.t()) ** 2).mean()

        return sim_loss

