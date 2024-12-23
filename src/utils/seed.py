import random

import numpy as np
import torch


def set_seed(seed):
    if seed:
        # For reproductivity
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
